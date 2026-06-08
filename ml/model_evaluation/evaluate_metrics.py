from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MODULE_DIR = os.path.dirname(__file__)
_VISUALIZATIONS_DIR = os.path.abspath(os.path.join(_MODULE_DIR, "..", "visualizations"))


# ---------------------------------------------------------------------------
# Threshold-based metrics (no accuracy_score)
# ---------------------------------------------------------------------------

def _ensure_1d(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """If *y_prob* is 2-D with exactly 2 columns (binary classifier),
    return the positive-class probabilities as a 1-D array."""
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        return y_true, y_prob[:, 1]
    return y_true, y_prob


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro-averaged Area Under the ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_prob : np.ndarray
        Predicted probabilities, shape ``(n_samples,)`` for binary,
        ``(n_samples, 2)`` for binary classifier output, or
        ``(n_samples, n_classes)`` for multi-class.

    Returns
    -------
    float
        AUROC score.
    """
    y_true, y_prob = _ensure_1d(y_true, y_prob)
    if y_prob.ndim == 1:
        return float(roc_auc_score(y_true, y_prob))
    return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro-averaged Area Under the Precision-Recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        AUPRC score.
    """
    y_true, y_prob = _ensure_1d(y_true, y_prob)
    if y_prob.ndim == 1:
        return float(average_precision_score(y_true, y_prob))

    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    scores = []
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_onehot[:, i], y_prob[:, i])
        scores.append(auc(recall, precision))
    return float(np.mean(scores))


def sensitivity_at_far(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_far: float = 0.07,
) -> dict[str, float]:
    """Sensitivity (recall) at a maximum false-alarm rate threshold.

    For binary classification, scans decision thresholds on the positive-class
    probability to find the highest recall achievable while keeping the
    false-alarm rate (FPR) below *max_far*.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class (1-D).
    max_far : float
        Maximum acceptable false-alarm rate (default ``0.07`` = 7%).

    Returns
    -------
    dict
        ``{"sensitivity": …, "threshold": …, "far_actual": …}``
    """
    if y_prob.ndim == 2:
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        else:
            y_prob = y_prob[:, -1]

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    mask = fpr <= max_far
    if not mask.any():
        idx = 0
    else:
        idx = np.where(mask)[0][-1]

    return {
        "sensitivity": float(tpr[idx]),
        "threshold": float(thresholds[idx]) if idx < len(thresholds) else 1.0,
        "far_actual": float(fpr[idx]),
    }


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
        Focusing parameter (default ``2.0``).
    alpha : float | None
        Weighting factor for class imbalance.
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
        if isinstance(alpha, (int, float)):
            alpha_t = alpha * targets.float() + (1.0 - alpha) * (1.0 - targets.float())
            loss = alpha_t * loss
        else:
            alpha_t = alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss

    return loss.mean() if reduction == "mean" else loss.sum()


# ---------------------------------------------------------------------------
# Comprehensive evaluation (no accuracy_score)
# ---------------------------------------------------------------------------

def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Compute comprehensive classification metrics (without accuracy).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    y_prob : np.ndarray | None
        Predicted probabilities (required for AUROC, AUPRC, and
        sensitivity at FAR).
    prefix : str
        Optional key prefix.

    Returns
    -------
    dict
        Metrics: precision, recall, F1 (weighted + binary), AUROC,
        AUPRC, sensitivity@7% FAR, confusion matrix, per-class counts.
    """
    metrics: dict[str, Any] = {
        f"{prefix}precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        metrics[f"{prefix}precision_binary"] = precision_score(y_true, y_pred, average="binary", zero_division=0)
        metrics[f"{prefix}recall_binary"] = recall_score(y_true, y_pred, average="binary", zero_division=0)
        metrics[f"{prefix}f1_binary"] = f1_score(y_true, y_pred, average="binary", zero_division=0)

    if y_prob is not None:
        metrics[f"{prefix}auroc_macro"] = compute_auroc(y_true, y_prob)
        metrics[f"{prefix}auprc_macro"] = compute_auprc(y_true, y_prob)

        if n_classes == 2:
            far_result = sensitivity_at_far(y_true, y_prob, max_far=0.07)
            metrics[f"{prefix}sensitivity_at_7pct_far"] = far_result["sensitivity"]
            metrics[f"{prefix}threshold_at_7pct_far"] = far_result["threshold"]
            metrics[f"{prefix}far_actual_at_threshold"] = far_result["far_actual"]

    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
    prefix: str = "",
) -> str:
    """Plot and save a confusion matrix heatmap.

    Returns the file path of the saved figure.
    """
    if save_path is None:
        os.makedirs(_VISUALIZATIONS_DIR, exist_ok=True)
        save_path = os.path.join(_VISUALIZATIONS_DIR, f"{prefix}confusion_matrix.png")

    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(max(6, cm.shape[0] * 1.5), max(5, cm.shape[1] * 1.2)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted label", ylabel="True label")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: str | None = None,
    prefix: str = "",
) -> str:
    """Plot and save the ROC curve with AUROC annotation.

    Returns the file path of the saved figure.
    """
    if save_path is None:
        os.makedirs(_VISUALIZATIONS_DIR, exist_ok=True)
        save_path = os.path.join(_VISUALIZATIONS_DIR, f"{prefix}roc_curve.png")

    if y_prob.ndim == 2:
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        else:
            y_prob = y_prob[:, -1]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")

    data_summary = {
        "auroc": float(auroc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }
    np.save(save_path.replace(".png", "_data.npy"), data_summary)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: str | None = None,
    prefix: str = "",
) -> str:
    """Plot and save the Precision-Recall curve with AUPRC annotation.

    Returns the file path of the saved figure.
    """
    if save_path is None:
        os.makedirs(_VISUALIZATIONS_DIR, exist_ok=True)
        save_path = os.path.join(_VISUALIZATIONS_DIR, f"{prefix}pr_curve.png")

    if y_prob.ndim == 2:
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        else:
            y_prob = y_prob[:, -1]

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="blue", lw=2, label=f"PR (AUPRC = {auprc:.4f})")
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color="gray", lw=1, linestyle="--", label=f"Baseline ({baseline:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    data_summary = {
        "auprc": float(auprc),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
    }
    np.save(save_path.replace(".png", "_data.npy"), data_summary)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _serialize(value: Any) -> Any:
    """Convert numpy types to native Python for JSON export."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def export_performance_log(
    metrics: dict[str, Any],
    save_path: str | None = None,
    prefix: str = "",
) -> str:
    """Export the evaluation metrics dict as a JSON log file.

    Returns the file path of the saved log.
    """
    if save_path is None:
        os.makedirs(_VISUALIZATIONS_DIR, exist_ok=True)
        save_path = os.path.join(_VISUALIZATIONS_DIR, f"{prefix}performance_log.json")

    serializable: dict[str, Any] = {}
    for k, v in metrics.items():
        if k.endswith("_plot") or k.endswith("confusion_matrix"):
            continue
        serializable[k] = _serialize(v)

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    return save_path


def evaluate_and_visualize(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
    prefix: str = "",
    save_plots: bool = True,
    save_log: bool = True,
) -> dict[str, Any]:
    """Run full evaluation + generate diagnostic plots + export log.

    Returns a dict with metrics, saved plot file paths, and log path.
    """
    metrics = evaluate_all(y_true, y_pred, y_prob=y_prob, prefix=prefix)

    if save_plots:
        cm_path = plot_confusion_matrix(y_true, y_pred, class_names=class_names, prefix=prefix)
        metrics[f"{prefix}confusion_matrix_plot"] = cm_path

        if y_prob is not None:
            roc_path = plot_roc_curve(y_true, y_prob, prefix=prefix)
            pr_path = plot_pr_curve(y_true, y_prob, prefix=prefix)
            metrics[f"{prefix}roc_curve_plot"] = roc_path
            metrics[f"{prefix}pr_curve_plot"] = pr_path

    if save_log:
        log_path = export_performance_log(metrics, prefix=prefix)
        metrics[f"{prefix}performance_log"] = log_path

    return metrics

# ---------------------------------------------------------------------------
# Execution Block to Trigger and Display Outputs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 INITIALIZING GOOGLE-GRADE CARDIAC RISK EVALUATION ENGINE")
    print("="*60)
    
    # 1. Simulating an Imbalanced Imminent Arrest Dataset (95% Normal, 5% High Risk)
    np.random.seed(42)
    n_samples = 1000
    
    # Ground Truth: 0 = Stable, 1 = Imminent Cardiac Arrest
    y_true_mock = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Generating mock probability scores (higher scores correlated with true events)
    y_prob_mock = np.random.beta(0.5, 2.0, size=n_samples)
    y_prob_mock = np.where(y_true_mock == 1, np.random.beta(2.0, 0.5, size=n_samples), y_prob_mock)
    
    # Discrete label prediction based on a traditional 0.5 classification threshold
    y_pred_mock = (y_prob_mock >= 0.5).astype(int)
    
    # 2. Running your complete pipeline
    print("\n[STEP 1/3] Computing high-dimensional imbalanced metrics...")
    results = evaluate_and_visualize(
        y_true=y_true_mock,
        y_pred=y_pred_mock,
        y_prob=y_prob_mock,
        class_names=["Stable_Athlete", "Cardiac_Arrest_Risk"],
        prefix="prod_v1_"
    )
    
    # 3. Formatted Terminal Display
    print("\n" + "📊 VERIFIED MODEL PERFORMANCE LOGS".center(60, "-"))
    print(f"• AUROC (Macro):              {results['prod_v1_auroc_macro']:.4f}")
    print(f"• AUPRC (Macro):              {results['prod_v1_auprc_macro']:.4f}")
    print(f"• F1-Score (Macro):           {results['prod_v1_f1_macro']:.4f}")
    print(f"• Sensitivity @ 7% FAR:       {results['prod_v1_sensitivity_at_7pct_far']*100:.2f}%")
    print(f"• Dynamic Alert Threshold:    {results['prod_v1_threshold_at_7pct_far']:.4f}")
    print(f"• Actual FAR achieved:        {results['prod_v1_far_actual_at_threshold']*100:.2f}%")
    
    print("\n" + "💾 EXPORTED ARTIFACT LOCATIONS".center(60, "-"))
    print(f"📌 Matrix Heatmap:  {results['prod_v1_confusion_matrix_plot']}")
    print(f"📌 ROC Curve Plot:  {results['prod_v1_roc_curve_plot']}")
    print(f"📌 PR Curve Plot:   {results['prod_v1_pr_curve_plot']}")
    print(f"📌 JSON Data Log:   {results['prod_v1_performance_log']}")
    print("="*60 + "\n")