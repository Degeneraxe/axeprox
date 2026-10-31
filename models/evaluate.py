"""Model evaluation metrics reporting precision, recall, F1, PR-AUC, MAE, and calibration."""
from __future__ import annotations

from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_cls_true: np.ndarray,
    y_cls_prob: np.ndarray,
    y_reg_true: np.ndarray,
    y_reg_pred: np.ndarray,
    censored_flags: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes comprehensive quantitative performance metrics."""
    y_pred_binary = (y_cls_prob >= threshold).astype(int)

    # Classification metrics
    prec = precision_score(y_cls_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_cls_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_cls_true, y_pred_binary, zero_division=0)

    try:
        pr_auc = average_precision_score(y_cls_true, y_cls_prob)
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = roc_auc_score(y_cls_true, y_cls_prob)
    except Exception:
        roc_auc = 0.5

    brier = brier_score_loss(y_cls_true, y_cls_prob)
    cm = confusion_matrix(y_cls_true, y_pred_binary, labels=[0, 1]).tolist()

    # Regression metrics on uncensored samples
    uncensored_mask = ~censored_flags
    if np.sum(uncensored_mask) > 0:
        true_log = y_reg_true[uncensored_mask]
        pred_log = y_reg_pred[uncensored_mask]
        mae_log = mean_absolute_error(true_log, pred_log)
        # Convert back to ms: |exp(pred) - exp(true)|
        true_ms = np.expm1(true_log)
        pred_ms = np.expm1(np.clip(pred_log, 0, 15))
        mae_ms = mean_absolute_error(true_ms, pred_ms)
    else:
        mae_log = 0.0
        mae_ms = 0.0

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "brier_calibration": float(brier),
        "confusion_matrix": cm,
        "mae_log_duration": float(mae_log),
        "mae_duration_ms": float(mae_ms),
    }


def format_comparison_table(metrics_lstm: Dict[str, Any], metrics_baseline: Dict[str, Any]) -> str:
    """Formats markdown comparison table."""
    lines = [
        "| Metric | PyTorch LSTM | Baseline (LogReg / Ridge) | Delta (%) |",
        "|---|---|---|---|",
        f"| **F1-Score (Viability)** | **{metrics_lstm['f1']:.3f}** | {metrics_baseline['f1']:.3f} | {((metrics_lstm['f1'] - metrics_baseline['f1']) / max(1e-4, metrics_baseline['f1']) * 100):+.1f}% |",
        f"| **Precision** | **{metrics_lstm['precision']:.3f}** | {metrics_baseline['precision']:.3f} | {((metrics_lstm['precision'] - metrics_baseline['precision']) / max(1e-4, metrics_baseline['precision']) * 100):+.1f}% |",
        f"| **Recall** | **{metrics_lstm['recall']:.3f}** | {metrics_baseline['recall']:.3f} | {((metrics_lstm['recall'] - metrics_baseline['recall']) / max(1e-4, metrics_baseline['recall']) * 100):+.1f}% |",
        f"| **PR-AUC** | **{metrics_lstm['pr_auc']:.3f}** | {metrics_baseline['pr_auc']:.3f} | {((metrics_lstm['pr_auc'] - metrics_baseline['pr_auc']) / max(1e-4, metrics_baseline['pr_auc']) * 100):+.1f}% |",
        f"| **ROC-AUC** | **{metrics_lstm['roc_auc']:.3f}** | {metrics_baseline['roc_auc']:.3f} | {((metrics_lstm['roc_auc'] - metrics_baseline['roc_auc']) / max(1e-4, metrics_baseline['roc_auc']) * 100):+.1f}% |",
        f"| **Brier Calibration** | **{metrics_lstm['brier_calibration']:.4f}** | {metrics_baseline['brier_calibration']:.4f} | {((metrics_baseline['brier_calibration'] - metrics_lstm['brier_calibration']) / max(1e-4, metrics_baseline['brier_calibration']) * 100):+.1f}% |",
        f"| **MAE Duration (ms)** | **{metrics_lstm['mae_duration_ms']:.1f} ms** | {metrics_baseline['mae_duration_ms']:.1f} ms | - |",
    ]
    return "\n".join(lines)
