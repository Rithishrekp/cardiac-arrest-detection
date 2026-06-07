from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

_MODULE_DIR = os.path.dirname(__file__)
_SAVED_MODELS_DIR = os.path.abspath(os.path.join(_MODULE_DIR, "..", "saved_models"))
_PROCESSED_DIR = os.path.abspath(os.path.join(_MODULE_DIR, "..", "datasets", "processed"))


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance.

    ``FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)``

    Parameters
    ----------
    gamma : float
        Focusing parameter; higher values down-weight easy examples.
    alpha : float | torch.Tensor | None
        Weighting factor for each class.  If a scalar, it weights the
        positive class; if a tensor, per-class weights.
    reduction : str
        ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self, gamma: float = 2.0, alpha: float | torch.Tensor | None = None, reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1.0 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (int, float)):
                alpha_t = self.alpha * targets.float() + (1.0 - self.alpha) * (1.0 - targets.float())
                focal = alpha_t * focal
            else:
                alpha_t = self.alpha.to(logits.device).gather(0, targets)
                focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


# ---------------------------------------------------------------------------
# CNN1D + BiLSTM hybrid
# ---------------------------------------------------------------------------

class CNNBiLSTM(nn.Module):
    """1-D CNN → Bidirectional LSTM for ECG waveform classification.

    Architecture
    ------------
    1. Three 1-D convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
       extract spatial features from the raw temporal signal.
    2. The CNN output (``batch, channels, reduced_timesteps``) is
       reshaped to ``(batch, reduced_timesteps, channels)`` and fed
       into a Bidirectional LSTM that models rhythmic decay.
    3. The final hidden states are concatenated and passed through a
       fully-connected classifier head.
    """

    def __init__(
        self,
        input_length: int = 9000,
        n_channels: int = 1,
        num_classes: int = 2,
        conv_filters: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        pool_sizes: list[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_units: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        conv_filters = conv_filters or [64, 128, 256]
        kernel_sizes = kernel_sizes or [7, 5, 3]
        pool_sizes = pool_sizes or [4, 4, 2]
        fc_units = fc_units or [64]

        conv_blocks: list[nn.Module] = []
        in_ch = n_channels
        for filters, kernel, pool in zip(conv_filters, kernel_sizes, pool_sizes):
            conv_blocks.extend([
                nn.Conv1d(in_ch, filters, kernel, padding=kernel // 2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(pool),
                nn.Dropout(dropout),
            ])
            in_ch = filters
        self.conv = nn.Sequential(*conv_blocks)

        with torch.inference_mode():
            dummy = torch.zeros(1, n_channels, input_length)
            out = self.conv(dummy)
        cnn_out_timesteps = out.shape[-1]
        cnn_out_channels = out.shape[1]

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        lstm_out_dim = lstm_hidden * 2
        fc_layers: list[nn.Module] = []
        in_feat = lstm_out_dim
        for units in fc_units:
            fc_layers.extend([nn.Linear(in_feat, units), nn.ReLU(), nn.Dropout(dropout)])
            in_feat = units
        fc_layers.append(nn.Linear(in_feat, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# Standalone BiLSTM (kept for backward compat)
# ---------------------------------------------------------------------------

class BiLSTM(nn.Module):
    """Bidirectional LSTM for waveform classification (standalone)."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
        all_probs.extend(probs.cpu().numpy())

    from sklearn.metrics import f1_score, precision_score, recall_score

    return {
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Public training API
# ---------------------------------------------------------------------------

def prepare_ecg_windows(
    ecg_path: str | None = None,
    window_size: int = 1000,
    stride: int = 200,
) -> np.ndarray:
    """Create overlapping windows from the filtered ECG recording.

    Parameters
    ----------
    ecg_path : str | None
        Path to ``ecg_filtered.npy``.  Defaults to
        ``ml/datasets/processed/ecg_filtered.npy``.
    window_size : int
        Number of samples per window (default ``1000``).
    stride : int
        Step between consecutive windows (default ``200``).

    Returns
    -------
    np.ndarray
        Shape ``(n_windows, n_channels, window_size)``.
    """
    if ecg_path is None:
        ecg_path = os.path.join(_PROCESSED_DIR, "ecg_filtered.npy")
    ecg = np.load(ecg_path)
    if ecg.ndim == 1:
        ecg = ecg[np.newaxis, :]
    n_channels, total_samples = ecg.shape
    windows: list[np.ndarray] = []
    for start in range(0, total_samples - window_size + 1, stride):
        windows.append(ecg[:, start:start + window_size])
    return np.stack(windows, axis=0).astype(np.float32)


def _auto_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def train_cnnbilstm(
    waveforms: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    save_model: bool = True,
    model_name: str = "best_cnnbilstm",
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Train the CNN + BiLSTM hybrid on ECG waveform windows.

    Uses :class:`FocalLoss` to handle class imbalance.

    Parameters
    ----------
    waveforms : np.ndarray
        Shape ``(n_samples, n_channels, n_timesteps)``.
    labels : np.ndarray
        1-D integer class labels.
    test_size : float
        Validation split fraction (default ``0.2``).
    batch_size : int
        Dataloader batch size (default ``32``).
    epochs : int
        Number of training epochs (default ``50``).
    lr : float
        Learning rate (default ``1e-3``).
    weight_decay : float
        L2 regularisation (default ``1e-4``).
    device : str
        ``"auto"``, ``"cuda"``, or ``"cpu"``.
    focal_gamma : float
        Focusing parameter for FocalLoss (default ``2.0``).
    focal_alpha : float | None
        Class-weight for FocalLoss (default ``None``).
    save_model : bool
        Save model weights to ``ml/saved_models/`` (default ``True``).
    model_name : str
        Stem for saved weight file (default ``"best_cnnbilstm"``).
    **model_kwargs
        Forwarded to :class:`CNNBiLSTM`.

    Returns
    -------
    dict
        ``"model"``, ``"history"``, ``"test_metrics"``, ``"best_epoch"``.
    """
    device = _auto_device(device)

    if waveforms.ndim == 2:
        waveforms = waveforms[:, np.newaxis, :]

    X_train, X_test, y_train, y_test = train_test_split(
        waveforms, labels, test_size=test_size, stratify=labels, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size,
    )

    model = CNNBiLSTM(
        input_length=waveforms.shape[-1],
        n_channels=waveforms.shape[1],
        num_classes=len(np.unique(labels)),
        **model_kwargs,
    ).to(device)

    criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    history = {"train_loss": [], "test_f1": [], "test_precision": [], "test_recall": []}
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = _evaluate(model, test_loader, device)
        history["train_loss"].append(loss)
        history["test_f1"].append(metrics["f1"])
        history["test_precision"].append(metrics["precision"])
        history["test_recall"].append(metrics["recall"])

        scheduler.step(metrics["f1"])

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            if save_model:
                os.makedirs(_SAVED_MODELS_DIR, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(_SAVED_MODELS_DIR, f"{model_name}.pt"),
                )

    if save_model:
        model.load_state_dict(
            torch.load(os.path.join(_SAVED_MODELS_DIR, f"{model_name}.pt"))
        )

    return {
        "model": model,
        "history": history,
        "test_metrics": {"f1": best_f1, "epoch": best_epoch},
        "best_epoch": best_epoch,
    }


def train_bilstm(
    waveforms: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    save_model: bool = True,
    model_name: str = "best_bilstm",
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Train a standalone BiLSTM on waveform segments.

    API matches :func:`train_cnnbilstm`.
    """
    device = _auto_device(device)

    if waveforms.ndim == 2:
        waveforms = waveforms[:, :, np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        waveforms, labels, test_size=test_size, stratify=labels, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size,
    )

    model = BiLSTM(
        input_size=waveforms.shape[-1],
        num_classes=len(np.unique(labels)),
        **model_kwargs,
    ).to(device)

    criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    history = {"train_loss": [], "test_f1": [], "test_precision": [], "test_recall": []}
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = _evaluate(model, test_loader, device)
        history["train_loss"].append(loss)
        history["test_f1"].append(metrics["f1"])
        history["test_precision"].append(metrics["precision"])
        history["test_recall"].append(metrics["recall"])

        scheduler.step(metrics["f1"])

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            if save_model:
                os.makedirs(_SAVED_MODELS_DIR, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(_SAVED_MODELS_DIR, f"{model_name}.pt"),
                )

    if save_model:
        model.load_state_dict(
            torch.load(os.path.join(_SAVED_MODELS_DIR, f"{model_name}.pt"))
        )

    return {
        "model": model,
        "history": history,
        "test_metrics": {"f1": best_f1, "epoch": best_epoch},
        "best_epoch": best_epoch,
    }
