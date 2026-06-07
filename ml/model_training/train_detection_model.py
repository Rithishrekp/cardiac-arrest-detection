from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class CNN1D(nn.Module):
    """1-D Convolutional Neural Network for waveform classification."""

    def __init__(
        self,
        input_length: int,
        n_channels: int = 1,
        num_classes: int = 3,
        conv_filters: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        fc_units: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        conv_filters = conv_filters or [64, 128, 256]
        kernel_sizes = kernel_sizes or [7, 5, 3]
        fc_units = fc_units or [128, 64]

        layers: list[nn.Module] = []
        in_channels = n_channels
        for filters, kernel in zip(conv_filters, kernel_sizes):
            layers.extend(
                [
                    nn.Conv1d(in_channels, filters, kernel, padding=kernel // 2),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = filters

        self.conv = nn.Sequential(*layers)

        with torch.inference_mode():
            dummy = torch.zeros(1, n_channels, input_length)
            out = self.conv(dummy)
        flattened = out.view(1, -1).shape[1]

        fc_layers: list[nn.Module] = []
        in_features = flattened
        for units in fc_units:
            fc_layers.extend(
                [
                    nn.Linear(in_features, units),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = units
        fc_layers.append(nn.Linear(in_features, num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class BiLSTM(nn.Module):
    """Bidirectional LSTM for sequential waveform classification."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
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
    all_preds, all_labels = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
    }


def train_cnn1d(
    waveforms: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "auto",
    **model_kwargs: Any,
) -> dict:
    """Train the 1-D CNN on waveform segments.

    Parameters
    ----------
    waveforms : np.ndarray
        Shape ``(n_samples, n_timesteps)`` or ``(n_samples, 1, n_timesteps)``.
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
    device : str
        ``"auto"``, ``"cuda"``, or ``"cpu"``.
    **model_kwargs
        Passed to :class:`CNN1D`.

    Returns
    -------
    dict
        ``"model"``, ``"history"``, ``"test_metrics"``.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if waveforms.ndim == 2:
        waveforms = waveforms[:, np.newaxis, :]

    X_train, X_test, y_train, y_test = train_test_split(
        waveforms, labels, test_size=test_size, stratify=labels, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size,
    )

    model = CNN1D(
        input_length=waveforms.shape[-1],
        n_channels=waveforms.shape[1],
        num_classes=len(np.unique(labels)),
        **model_kwargs,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "test_accuracy": [], "test_f1": []}
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = _evaluate(model, test_loader, device)
        history["train_loss"].append(loss)
        history["test_accuracy"].append(metrics["accuracy"])
        history["test_f1"].append(metrics["f1"])

    return {"model": model, "history": history, "test_metrics": history["test_f1"][-1]}


def train_bilstm(
    waveforms: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "auto",
    **model_kwargs: Any,
) -> dict:
    """Train a BiLSTM on waveform segments.

    API matches :func:`train_cnn1d`.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if waveforms.ndim == 2:
        waveforms = waveforms[:, :, np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        waveforms, labels, test_size=test_size, stratify=labels, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size,
    )

    model = BiLSTM(
        input_size=waveforms.shape[-1],
        num_classes=len(np.unique(labels)),
        **model_kwargs,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "test_accuracy": [], "test_f1": []}
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = _evaluate(model, test_loader, device)
        history["train_loss"].append(loss)
        history["test_accuracy"].append(metrics["accuracy"])
        history["test_f1"].append(metrics["f1"])

    return {"model": model, "history": history, "test_metrics": history["test_f1"][-1]}
