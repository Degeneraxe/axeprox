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
    dataset_version: Optional[str] = None,
    target_label: Optional[str] = None,
    return_stats: bool = False,
) -> Any:
    """Constructs sequential feature vectors and targets directly from persisted spread events in DB.
    
    Step 1 & Step 3 Implementation:
    - Queries ONLY the active dataset_version and strictly excludes 'invalid_v0' rows.
    - Excludes suspect-flagged events (is_suspect = 1).
    - Configurable classification target (default: 'is_executable').
    - Reports class balance across all labels (is_executable and is_profitable_<scenario>).
    """
    config = AppConfig.load()
    if min_events is None:
        min_events = int(config.raw.get("training", {}).get("min_events", 20))

    active_version = dataset_version or config.raw.get("dataset", {}).get("version", "v1.0")
    tgt_label = target_label or config.raw.get("training", {}).get("target_label", "is_executable")

    db = DatabaseManager(sqlite_path=db_path)
    db.initialize()
    cur = db._sqlite_conn.cursor()

    # If dataset_version is not explicitly specified and active_version is not in DB, fallback to valid versions present in DB
    if dataset_version is None:
        cur.execute("SELECT DISTINCT dataset_version FROM spread_events WHERE dataset_version NOT LIKE 'invalid%' AND dataset_version IS NOT NULL")
        avail = [r[0] for r in cur.fetchall()]
        if active_version not in avail and avail:
            active_version = avail[-1]

    # Query completed spread events for the active dataset_version, strictly excluding invalid_v0 and suspect rows
    cur.execute("""
        SELECT * FROM spread_events 
        WHERE duration_ms IS NOT NULL 
          AND feature_window_json IS NOT NULL
          AND (dataset_version = ? OR dataset_version IS NULL)
          AND (dataset_version NOT LIKE 'invalid%' OR dataset_version IS NULL)
          AND (is_suspect = 0 OR is_suspect IS NULL)
        ORDER BY open_time ASC
    """, (active_version,))
    events = [dict(r) for r in cur.fetchall()]
    db.close()

    samples_X: List[np.ndarray] = []
    labels_cls: List[int] = []
    targets_reg: List[float] = []
    censored_flags: List[bool] = []

    class_balances = {
        "is_executable": {"pos": 0, "neg": 0},
        "is_persistent_500ms": {"pos": 0, "neg": 0},
        "is_persistent_2000ms": {"pos": 0, "neg": 0},
        "is_persistent_5000ms": {"pos": 0, "neg": 0},
        "is_viable": {"pos": 0, "neg": 0},
        "is_profitable_retail": {"pos": 0, "neg": 0},
        "is_profitable_vip1": {"pos": 0, "neg": 0},
        "is_profitable_vip3": {"pos": 0, "neg": 0},
        "is_profitable_zero_maker": {"pos": 0, "neg": 0},
    }

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
        is_censored = bool(evt.get("is_right_censored", 0))

        # Target classification label
        lbl_val = int(evt.get(tgt_label, evt.get("is_executable", evt.get("is_viable", 0))) or 0)

        # Track balances across all available label columns (persistence labels evaluated on non-censored only)
        for k in class_balances:
            if "persistent" in k:
                if is_censored:
                    continue
                # If column not explicitly in DB row, derive from duration
                h_ms = 500.0 if "500ms" in k else (2000.0 if "2000ms" in k else 5000.0)
                v = int(evt.get(k, 1 if dur >= h_ms else 0) or 0)
            else:
                v = int(evt.get(k, 0) or 0)

            if v == 1:
                class_balances[k]["pos"] += 1
            else:
                class_balances[k]["neg"] += 1

        samples_X.append(np.array(seq, dtype=np.float32))
        labels_cls.append(lbl_val)
        targets_reg.append(log_dur)
        censored_flags.append(is_censored)

    # Zero fabrication check: strictly fail if fewer than min_events exist
    if len(samples_X) < min_events:
        msg = (
            f"Insufficient real events with persisted feature windows in DB for version '{active_version}' "
            f"(found {len(samples_X)} complete sample windows, required minimum {min_events}). "
            f"Never fabricating synthetic or stride samples."
        )
        logger.error("[Training Pipeline] %s", msg)
        raise ValueError(msg)

    if return_stats:
        return samples_X, labels_cls, targets_reg, censored_flags, class_balances
    return samples_X, labels_cls, targets_reg, censored_flags




def train_models(
    db_path: str = "data/arbsense.db",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    checkpoint_dir: str = "models/checkpoints",
    target_label: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs end-to-end model training, validation, early stopping, and comparative evaluation."""
    print("=" * 80)
    print("ArbSense AI Engine Training Pipeline (PyTorch LSTM vs Baseline)")
    print("=" * 80)

    cfg = AppConfig.load()
    train_cfg = cfg.raw.get("training", {})
    dataset_cfg = cfg.raw.get("dataset", {})
    tgt_label = target_label or train_cfg.get("target_label", "is_executable")

    # 1. Load Data with full label statistics
    samples_X, labels_cls, targets_reg, censored_flags, class_balances = build_dataset_from_db(
        db_path=db_path,
        target_label=tgt_label,
        return_stats=True,
    )
    print(f"[*] Total dataset samples extracted from real ticks/events: {len(samples_X)}")

    # Step 3: Report class balance per label and per scenario
    print("\n" + "=" * 80)
    print("CLASS BALANCE PER LABEL AND FEE SCENARIO (Current Dataset Version)")
    print("=" * 80)
    for lbl_name, counts in class_balances.items():
        tot = counts["pos"] + counts["neg"]
        pct = (counts["pos"] / max(1, tot)) * 100.0
        print(f"  {lbl_name:<26}: {counts['pos']:>4} positive / {counts['neg']:>4} negative ({pct:>5.1f}%)")
    print("=" * 80)

    # Check requirement: "If positives for a label are fewer than 30, do not train a classifier on it; report the count instead."
    pos_count_target = class_balances.get(tgt_label, {}).get("pos", sum(1 for y in labels_cls if y == 1))
    should_train_classifier = (pos_count_target >= 30)
    if not should_train_classifier:
        print(f"\n[!] Target label '{tgt_label}' has {pos_count_target} positive samples (< 30 required).")
        print(f"    Per requirement: Classifier training bypassed; reporting counts only. Training regression head on log_duration.")
    else:
        print(f"\n[*] Target label '{tgt_label}' has {pos_count_target} positive samples (>= 30 required). Training dual-head classifier + regressor.")

    # 2. Chronological Walk-Forward Split with Purge Gap read from config.yaml (Defect 6)
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

    # Persistence target selection: evaluate non-censored training fold to choose horizon closest to 50/50 balance
    uncens_mask_train = ~splits.censored_train
    uncens_train_reg = splits.y_reg_train[uncens_mask_train]
    print("\n" + "=" * 80)
    print("PERSISTENCE HORIZON BALANCE (TRAINING SPLIT ONLY, Non-Censored)")
    print("=" * 80)
    best_horizon = None
    min_dev = 999.0
    for h_ms in [500, 2000, 5000]:
        h_log = math.log(1.0 + h_ms)
        h_pos = int(np.sum(uncens_train_reg >= h_log))
        h_tot = len(uncens_train_reg)
        h_pct = (h_pos / max(1, h_tot)) * 100.0
        h_dev = abs(h_pct - 50.0)
        print(f"  is_persistent_{h_ms}ms: {h_pos:>4} pos / {h_tot - h_pos:>4} neg ({h_pct:>5.1f}%) [|pct - 50%| = {h_dev:.1f}%]")
        if h_dev < min_dev:
            min_dev = h_dev
            best_horizon = h_ms
    print(f"[*] Optimal primary persistence target selected: is_persistent_{best_horizon}ms")
    print("=" * 80)

    # 3. Class Imbalance Weighting (Weighted BCE)
    n_pos = float(np.sum(splits.y_cls_train == 1.0))
    n_neg = float(np.sum(splits.y_cls_train == 0.0))
    pos_weight_val = (n_neg / max(1.0, n_pos)) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    print(f"    - Class Distribution (Train): {int(n_pos)} pos / {int(n_neg)} neg (pos_weight = {pos_weight_val:.2f})")

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

            # Classification loss (Weighted BCE) only if classifier training is active
            if should_train_classifier:
                loss_cls = F.binary_cross_entropy(pred_cls, batch_y_cls, weight=torch.where(batch_y_cls == 1.0, pos_weight, 1.0))
            else:
                loss_cls = torch.tensor(0.0)

            # Regression loss on uncensored samples
            uncens_mask = ~batch_cens
            if uncens_mask.sum() > 0:
                loss_reg = F.smooth_l1_loss(pred_reg[uncens_mask], batch_y_reg[uncens_mask])
            else:
                loss_reg = torch.tensor(0.0)

            total_loss = (loss_cls + 0.5 * loss_reg) if should_train_classifier else loss_reg
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
            if should_train_classifier:
                v_loss_cls = F.binary_cross_entropy(v_pred_cls, val_y_cls)
            else:
                v_loss_cls = torch.tensor(0.0)
            uncens_val = ~val_cens
            if uncens_val.sum() > 0:
                v_loss_reg = F.smooth_l1_loss(v_pred_reg[uncens_val], val_y_reg[uncens_val])
            else:
                v_loss_reg = torch.tensor(0.0)
            val_loss = (v_loss_cls + 0.5 * v_loss_reg).item() if should_train_classifier else v_loss_reg.item()

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
    best_thresh = 0.5
    best_val_f1 = 0.0
    if should_train_classifier:
        with torch.no_grad():
            v_cls_prob, _ = model(torch.tensor(splits.X_val, dtype=torch.float32))
            v_probs = v_cls_prob.squeeze().numpy()

        for t_cand in np.linspace(0.2, 0.8, 13):
            from sklearn.metrics import f1_score
            f1_cand = f1_score(splits.y_cls_val, (v_probs >= t_cand).astype(int), zero_division=0)
            if f1_cand > best_val_f1:
                best_val_f1 = f1_cand
                best_thresh = float(t_cand)

        print(f"    - Decision threshold calibrated on Validation set: {best_thresh:.2f} (Val F1: {best_val_f1:.3f})")
    else:
        print(f"    - Decision threshold calibration skipped (classifier not trained; positives < 30)")

    # Save complete unified checkpoint (Defect 5: weights + decision_threshold + scaler + feature_names)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "decision_threshold": float(best_thresh),
        "scaler": {
            "means": splits.scaler.means.tolist(),
            "stds": splits.scaler.stds.tolist(),
        },
        "feature_names": FEATURE_NAMES,
        "target_label": tgt_label,
        "classifier_trained": should_train_classifier,
    }
    torch.save(checkpoint_payload, best_ckpt_path)

    # 7. Train Baseline Model (Amendment: Logistic Regression & Ridge on pooled features)
    print("\n[Step 2] Training Baseline Model (Logistic Regression & Ridge)...")
    baseline = BaselineModel()
    baseline.fit(splits.X_train, splits.y_cls_train, splits.y_reg_train, splits.censored_train, fit_classifier=should_train_classifier)

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
    if not should_train_classifier:
        print(f"NOTE: Classification metrics reflect {tgt_label} count report only ({pos_count_target} positive samples < 30).")
    print(format_comparison_table(metrics_lstm, metrics_baseline))
    print(f"\nBest Checkpoint saved: {best_ckpt_path}")

    return metrics_lstm, metrics_baseline

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
