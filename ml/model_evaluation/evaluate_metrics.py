from __future__ import annotations

from typing import Any

import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute macro-averaged Area Under the ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_prob : np.ndarray
        Predicted class probabilities, shape ``(n_samples, n_classes)``.

    Returns
    -------
    float
        Macro-averaged AUROC.
    """
    if y_prob.ndim == 1 or y_prob.shape[1] == 1:
        return roc_auc_score(y_true, y_prob)
    return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute macro-averaged Area Under the Precision-Recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_prob : np.ndarray
        Predicted class probabilities, shape ``(n_samples, n_classes)``.

    Returns
    -------
    float
        Macro-averaged AUPRC.
    """
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    scores = []
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_onehot[:, i], y_prob[:, i])
        scores.append(auc(recall, precision))
    return float(np.mean(scores))


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal Loss for imbalanced classification.

    ``FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)``

    Parameters
    ----------
    logits : torch.Tensor
        Raw class scores, shape ``(batch, n_classes)``.
    targets : torch.Tensor
        Integer class indices, shape ``(batch,)``.
    gamma : float
        Focusing parameter; higher values down-weight easy examples.
    alpha : float | None
        Weighting factor for class imbalance.  If ``None``, no weighting.
    reduction : str
        ``"mean"`` or ``"sum"``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    loss = (1.0 - pt) ** gamma * ce_loss

    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean() if reduction == "mean" else loss.sum()


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    y_prob : np.ndarray | None
        Predicted probabilities (required for AUROC / AUPRC).
    prefix : str
        Optional prefix for metric keys.

    Returns
    -------
    dict[str, Any]
        Metrics including accuracy, precision, recall, F1, AUROC,
        AUPRC, confusion matrix, and per-class counts.
    """
    metrics: dict[str, Any] = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        metrics[f"{prefix}auroc_macro"] = compute_auroc(y_true, y_prob)
        metrics[f"{prefix}auprc_macro"] = compute_auprc(y_true, y_prob)

    return metrics
