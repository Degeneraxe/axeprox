"""Model training pipeline for ArbSense.

Implements:
- Chronological time-series walk-forward dataset construction with purge gaps
- PyTorch multi-task training loop (Weighted BCE for classification + Smooth L1 for regression)
- Class imbalance weighting (pos_weight = N_neg / N_pos)
- Early stopping with validation checkpointing (models/checkpoints/lstm_best.pt)
- Evaluation against baseline and reporting of F1, Precision, Recall, PR-AUC, Brier score, and MAE.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AppConfig
from data.storage.db_manager import DatabaseManager
from models.baseline import BaselineModel
from models.dataset import DatasetSplits, create_walk_forward_splits
from models.evaluate import evaluate_predictions, format_comparison_table
from models.features import FEATURE_NAMES, extract_step_features
from models.lstm_model import MultiHeadSpreadLSTM

logger = logging.getLogger(__name__)


def build_dataset_from_db(
    db_path: str = "data/arbsense.db",
    seq_len: int = 20,
    min_events: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[int], List[float], List[bool]]:
    """Constructs sequential feature vectors and targets directly from persisted spread events in DB.
    
    Requirement 1:
    - Reads persisted 20-step point-in-time feature windows (feature_window_json) directly from spread_events.
    - Does NOT query raw orderbook_ticks for training samples.
    - Preserves training data permanently even when raw ticks are pruned.
    - If fewer than min_events persisted feature windows exist, raises ValueError (Zero Fabrication Guarantee).
    """
    config = AppConfig.load()
    if min_events is None:
        min_events = int(config.raw.get("training", {}).get("min_events", 20))

    db = DatabaseManager(sqlite_path=db_path)
    db.initialize()
    cur = db._sqlite_conn.cursor()

    # Query completed real spread events from tracker with persisted feature windows
    cur.execute("""
        SELECT * FROM spread_events 
        WHERE duration_ms IS NOT NULL AND feature_window_json IS NOT NULL
        ORDER BY open_time ASC
    """)
    events = [dict(r) for r in cur.fetchall()]
    db.close()

    samples_X: List[np.ndarray] = []
    labels_cls: List[int] = []
    targets_reg: List[float] = []
    censored_flags: List[bool] = []

    import json
    for evt in events:
        win_str = evt.get("feature_window_json")
        if not win_str:
            continue
        try:
            raw_seq = json.loads(win_str)
        except Exception:
            continue

        if not raw_seq or len(raw_seq) < seq_len:
            continue

        seq = raw_seq[-seq_len:]
        dur = float(evt["duration_ms"])
        log_dur = float(evt.get("log_duration") or math.log(1.0 + max(1.0, dur)))
        is_viable = int(evt.get("is_viable", 0))
        is_censored = bool(evt.get("is_right_censored", 0))

        samples_X.append(np.array(seq, dtype=np.float32))
        labels_cls.append(is_viable)
        targets_reg.append(log_dur)
        censored_flags.append(is_censored)

    # Zero fabrication check: strictly fail if fewer than min_events exist
    if len(samples_X) < min_events:
        msg = (
            f"Insufficient real events with persisted feature windows in DB "
            f"(found {len(samples_X)} complete sample windows, required minimum {min_events}). "
            f"Never fabricating synthetic or stride samples."
        )
        logger.error("[Training Pipeline] %s", msg)
        raise ValueError(msg)

    return samples_X, labels_cls, targets_reg, censored_flags




def train_models(
    db_path: str = "data/arbsense.db",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    checkpoint_dir: str = "models/checkpoints",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs end-to-end model training, validation, early stopping, and comparative evaluation."""
    print("=" * 80)
    print("ArbSense AI Engine Training Pipeline (PyTorch LSTM vs Baseline)")
    print("=" * 80)

    # 1. Load Data
    samples_X, labels_cls, targets_reg, censored_flags = build_dataset_from_db(db_path=db_path)
    print(f"[*] Total dataset samples extracted from real ticks/events: {len(samples_X)}")

    # 2. Chronological Walk-Forward Split with Purge Gap read from config.yaml (Defect 6)
    cfg = AppConfig.load()
    train_cfg = cfg.raw.get("training", {})
    dataset_cfg = cfg.raw.get("dataset", {})
    train_ratio = float(train_cfg.get("train_ratio", dataset_cfg.get("train_ratio", 0.70)))
    val_ratio = float(train_cfg.get("val_ratio", dataset_cfg.get("val_ratio", 0.15)))
    purge_gap = int(train_cfg.get("purge_gap_events", dataset_cfg.get("purge_gap_events", 5)))

    splits = create_walk_forward_splits(
        samples_X=samples_X,
        labels_cls=labels_cls,
        targets_reg=targets_reg,
        censored_flags=censored_flags,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        purge_gap=purge_gap,
    )
    print(f"    - Train set size:      {len(splits.X_train)} samples")
    print(f"    - Validation set size: {len(splits.X_val)} samples")
    print(f"    - Test set size:       {len(splits.X_test)} samples (Unseen evaluation holdout)")

    # 3. Class Imbalance Weighting (Weighted BCE)
    n_pos = float(np.sum(splits.y_cls_train == 1.0))
    n_neg = float(np.sum(splits.y_cls_train == 0.0))
    pos_weight_val = (n_neg / max(1.0, n_pos)) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    print(f"    - Class Distribution (Train): {int(n_pos)} viable / {int(n_neg)} non-viable (pos_weight = {pos_weight_val:.2f})")

    # 4. Prepare PyTorch DataLoaders
    train_ds = TensorDataset(
        torch.tensor(splits.X_train, dtype=torch.float32),
        torch.tensor(splits.y_cls_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(splits.y_reg_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(splits.censored_train, dtype=torch.bool).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(splits.X_val, dtype=torch.float32),
        torch.tensor(splits.y_cls_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(splits.y_reg_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(splits.censored_val, dtype=torch.bool).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 5. Initialize LSTM Model
    input_dim = splits.X_train.shape[-1]
    model = MultiHeadSpreadLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 6. Training Loop with Early Stopping
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, "lstm_best.pt")
    scaler_path = os.path.join(checkpoint_dir, "scaler_params.json")
    import json
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({
            "means": splits.scaler.means.tolist(),
            "stds": splits.scaler.stds.tolist(),
        }, f)
    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0

    print("\n[Step 1] Training PyTorch Multi-Head LSTM Network...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y_cls, batch_y_reg, batch_cens in train_loader:
            optimizer.zero_grad()
            pred_cls, pred_reg = model(batch_x)

            # Classification loss (Weighted BCE)
            loss_cls = F.binary_cross_entropy(pred_cls, batch_y_cls, weight=torch.where(batch_y_cls == 1.0, pos_weight, 1.0))

            # Regression loss on uncensored samples
            uncens_mask = ~batch_cens
            if uncens_mask.sum() > 0:
                loss_reg = F.smooth_l1_loss(pred_reg[uncens_mask], batch_y_reg[uncens_mask])
            else:
                loss_reg = torch.tensor(0.0)

            total_loss = loss_cls + 0.5 * loss_reg
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += total_loss.item()

        # Validation pass
        model.eval()
        val_x = torch.tensor(splits.X_val, dtype=torch.float32)
        val_y_cls = torch.tensor(splits.y_cls_val, dtype=torch.float32).unsqueeze(1)
        val_y_reg = torch.tensor(splits.y_reg_val, dtype=torch.float32).unsqueeze(1)
        val_cens = torch.tensor(splits.censored_val, dtype=torch.bool).unsqueeze(1)

        with torch.no_grad():
            v_pred_cls, v_pred_reg = model(val_x)
            v_loss_cls = F.binary_cross_entropy(v_pred_cls, val_y_cls)
            v_loss_reg = F.smooth_l1_loss(v_pred_reg[~val_cens], val_y_reg[~val_cens])
            val_loss = (v_loss_cls + 0.5 * v_loss_reg).item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    [Early Stopping] Triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()

    # Calibrate decision threshold STRICTLY on the validation set (never test set)
    with torch.no_grad():
        v_cls_prob, _ = model(torch.tensor(splits.X_val, dtype=torch.float32))
        v_probs = v_cls_prob.squeeze().numpy()

    best_thresh = 0.5
    best_val_f1 = 0.0
    for t_cand in np.linspace(0.2, 0.8, 13):
        from sklearn.metrics import f1_score
        f1_cand = f1_score(splits.y_cls_val, (v_probs >= t_cand).astype(int), zero_division=0)
        if f1_cand > best_val_f1:
            best_val_f1 = f1_cand
            best_thresh = float(t_cand)

    print(f"    - Decision threshold calibrated on Validation set: {best_thresh:.2f} (Val F1: {best_val_f1:.3f})")

    # Save complete unified checkpoint (Defect 5: weights + decision_threshold + scaler + feature_names)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "decision_threshold": float(best_thresh),
        "scaler": {
            "means": splits.scaler.means.tolist(),
            "stds": splits.scaler.stds.tolist(),
        },
        "feature_names": FEATURE_NAMES,
    }
    torch.save(checkpoint_payload, best_ckpt_path)


    # 7. Train Baseline Model (Amendment: Logistic Regression & Ridge on pooled features)
    print("\n[Step 2] Training Baseline Model (Logistic Regression & Ridge)...")
    baseline = BaselineModel()
    baseline.fit(splits.X_train, splits.y_cls_train, splits.y_reg_train, splits.censored_train)

    # 8. Unseen Test Set Evaluation (Strict no test-set tuning rule)
    print("\n[Step 3] Evaluating on Unseen Chronological Test Set...")
    test_x = torch.tensor(splits.X_test, dtype=torch.float32)
    with torch.no_grad():
        lstm_test_cls_prob, lstm_test_reg_log = model(test_x)
        lstm_cls_prob = lstm_test_cls_prob.squeeze().numpy()
        lstm_reg_pred = lstm_test_reg_log.squeeze().numpy()

    base_cls_prob, base_reg_pred = baseline.predict(splits.X_test)

    # Compute metrics using the threshold calibrated on validation fold
    metrics_lstm = evaluate_predictions(
        y_cls_true=splits.y_cls_test,
        y_cls_prob=lstm_cls_prob,
        y_reg_true=splits.y_reg_test,
        y_reg_pred=lstm_reg_pred,
        censored_flags=splits.censored_test,
        threshold=best_thresh,
    )
    metrics_baseline = evaluate_predictions(
        y_cls_true=splits.y_cls_test,
        y_cls_prob=base_cls_prob,
        y_reg_true=splits.y_reg_test,
        y_reg_pred=base_reg_pred,
        censored_flags=splits.censored_test,
        threshold=best_thresh,
    )

    print("\n" + "=" * 80)
    print("EVIDENCE: Comparative Evaluation Report (Test Set)")
    print("=" * 80)
    print(format_comparison_table(metrics_lstm, metrics_baseline))
    print(f"\nBest Checkpoint saved: {best_ckpt_path}")

    return metrics_lstm, metrics_baseline


if __name__ == "__main__":
    try:
        train_models()
    except ValueError as err:
        print(f"\n[Training Halted - Real Data Integrity Gate] {err}")
        sys.exit(0)
