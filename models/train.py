"""Single Entry Point Model Training & Evaluation Pipeline for ArbSense (Item 2).

Integrates:
1. Non-overlapping expanding-window walk-forward validation (stride >= test_window).
2. 60s time purge + >=5 events between all partitions.
3. Held-out final period never used during fold training or threshold tuning.
4. Unified feature engineering via models.features (reused by predictor, tracker, and training).
5. MultiHeadSpreadLSTM production model training with early stopping & threshold tuning.
6. Checkpoint persistence with complete provenance metadata (dataset_version, git_commit, feature_names, threshold, scaler).
7. Results JSON output for the Dashboard (models/checkpoints/model_results.json).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import AppConfig, get_git_commit
from models.features import (
    FEATURE_NAMES,
    TABULAR_FEATURE_NAMES,
    get_feature_names,
    get_tabular_feature_names,
    parse_feature_window,
)
from models.lstm_model import MultiHeadSpreadLSTM
from models.walk_forward import (
    run_walkforward_pipeline,
    set_seed,
    compute_metrics,
    tune_mcc_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainPipeline")


def build_dataset_from_db(
    db_path: str = "data/arbsense.db",
    seq_len: int = 20,
    min_events: Optional[int] = None,
    dataset_version: Optional[str] = None,
    target_label: Optional[str] = None,
    return_stats: bool = False,
) -> Any:
    """Constructs sequential feature vectors and targets directly from spread events in DB."""
    config = AppConfig.load()
    if min_events is None:
        min_events = int(config.raw.get("training", {}).get("min_events", 20))

    active_version = dataset_version or config.raw.get("dataset", {}).get("version", "v2.1")
    tgt_label = target_label or config.raw.get("training", {}).get("target_label", "is_persistent_500ms")

    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(f"file:{p.resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(spread_events);")
    cols = {r[1] for r in cur.fetchall()}

    has_version = "dataset_version" in cols
    has_suspect = "is_suspect" in cols

    where_clauses = ["duration_ms IS NOT NULL", "feature_window_json IS NOT NULL"]
    params: List[Any] = []

    if has_version:
        cur.execute("SELECT DISTINCT dataset_version FROM spread_events WHERE dataset_version NOT LIKE 'invalid%' AND dataset_version IS NOT NULL")
        avail = [r[0] for r in cur.fetchall()]
        if dataset_version is None and active_version not in avail and avail:
            active_version = avail[-1]
        where_clauses.append("(dataset_version = ? OR dataset_version IS NULL)")
        where_clauses.append("(dataset_version NOT LIKE 'invalid%' OR dataset_version IS NULL)")
        params.append(active_version)

    if has_suspect:
        where_clauses.append("(is_suspect = 0 OR is_suspect IS NULL)")

    execution_target = tgt_label in ("is_viable", "is_executable") or tgt_label.startswith("is_profitable_")
    if execution_target and (active_version == "v2.2" or dataset_version is None):
        if "execution_evidence_json" not in cols:
            conn.close()
            raise ValueError("Arrival-based execution labels require recorded arrival evidence")
        where_clauses.append("json_extract(execution_evidence_json, '$.status') IN ('priced','insufficient_depth')")

    query = f"SELECT * FROM spread_events WHERE {' AND '.join(where_clauses)} ORDER BY open_time ASC"
    cur.execute(query, tuple(params))
    events = [dict(r) for r in cur.fetchall()]
    conn.close()

    samples_X: List[np.ndarray] = []
    labels_cls: List[int] = []
    targets_reg: List[float] = []
    censored_flags: List[bool] = []
    class_balances: Dict[str, Any] = {
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

    use_latency = bool(config.raw.get("features", {}).get("use_latency", False))
    imputed_lat_ms = float(config.raw.get("dataset", {}).get("viability_latency_threshold_ms", 85.0))

    for ev in events:
        win_str = ev.get("feature_window_json")
        if not win_str:
            continue
        seq_arr = parse_feature_window(win_str, use_latency=use_latency, imputed_latency_ms=imputed_lat_ms)
        if len(seq_arr) < seq_len:
            continue

        samples_X.append(seq_arr[-seq_len:])

        dur = float(ev.get("duration_ms", 0.0))
        is_cens = bool(ev.get("is_right_censored", 0))
        censored_flags.append(is_cens)

        log_dur = float(ev.get("log_duration", math.log1p(max(0.0, dur))))
        targets_reg.append(log_dur)

        evidence = json.loads(ev.get("execution_evidence_json") or "{}")
        for lbl_k in class_balances:
            if (ev.get("dataset_version") == "v2.2" and not lbl_k.startswith("is_persistent_")
                    and evidence.get("status") not in ("priced", "insufficient_depth")):
                continue
            val = ev.get(lbl_k)
            if val == 1 or val is True:
                class_balances[lbl_k]["pos"] += 1
            else:
                class_balances[lbl_k]["neg"] += 1

        if tgt_label in ev and ev[tgt_label] is not None:
            c_val = int(ev[tgt_label])
        elif tgt_label == "is_persistent_500ms":
            c_val = 1 if dur >= 500.0 else 0
        elif tgt_label == "is_executable":
            c_val = int(ev.get("is_executable", 1 if dur >= 85.0 else 0))
        elif tgt_label == "is_viable":
            c_val = int(ev.get("is_viable", 1 if dur >= 85.0 else 0))
        else:
            c_val = 1 if dur >= 500.0 else 0

        labels_cls.append(c_val)

    if len(samples_X) < min_events:
        raise ValueError(f"Insufficient real events with persisted feature windows in DB: found {len(samples_X)}, required >= {min_events}")

    if return_stats:
        return samples_X, labels_cls, targets_reg, censored_flags, class_balances
    return samples_X, labels_cls, targets_reg, censored_flags


def train_production_model(
    events: List[Dict[str, Any]],
    held_out_events: List[Dict[str, Any]],
    dataset_version: str,
    target_label: str,
    checkpoint_path: str = "models/checkpoints/lstm_best.pt",
    val_ratio: float = 0.20,
) -> Dict[str, Any]:
    """Trains final production MultiHeadSpreadLSTM on active events and evaluates on held-out test events."""
    set_seed(42)
    device = torch.device("cpu")

    # Filter uncensored events
    uncens_active = [e for e in events if not e["is_censored"]]
    uncens_held = [e for e in held_out_events if not e["is_censored"]]

    if len(uncens_active) < 15:
        raise ValueError(f"Insufficient active events for production training (found {len(uncens_active)})")

    n_active = len(uncens_active)
    split_idx = int(n_active * (1.0 - val_ratio))
    train_pool = uncens_active[:split_idx]
    val_pool = uncens_active[split_idx:]

    X_tr_seq = np.array([e["seq"] for e in train_pool], dtype=np.float32)
    y_tr = np.array([e["y_cls"] for e in train_pool], dtype=np.float32)
    log_dur_tr = np.array([e["log_duration"] for e in train_pool], dtype=np.float32)

    X_va_seq = np.array([e["seq"] for e in val_pool], dtype=np.float32) if val_pool else X_tr_seq[-5:]
    y_va = np.array([e["y_cls"] for e in val_pool], dtype=np.float32) if val_pool else y_tr[-5:]
    log_dur_va = np.array([e["log_duration"] for e in val_pool], dtype=np.float32) if val_pool else log_dur_tr[-5:]

    # Normalization statistics computed strictly on training portion
    seq_means = np.mean(X_tr_seq, axis=(0, 1))
    seq_stds = np.std(X_tr_seq, axis=(0, 1))
    seq_stds[seq_stds < 1e-6] = 1.0

    X_tr_norm = (X_tr_seq - seq_means) / seq_stds
    X_va_norm = (X_va_seq - seq_means) / seq_stds

    # Initialize model using config hyperparameters (Item 3)
    use_latency = bool(cfg.raw.get("features", {}).get("use_latency", False))
    feat_names = get_feature_names(use_latency=use_latency)
    input_dim = X_tr_seq.shape[2] if len(X_tr_seq.shape) == 3 and X_tr_seq.shape[2] > 0 else len(feat_names)

    lstm_cfg = cfg.raw.get("model", {}).get("lstm", {})
    hidden_dim = int(lstm_cfg.get("hidden_dim", 64))
    num_layers = int(lstm_cfg.get("layers", 2))
    dropout = float(lstm_cfg.get("dropout", 0.2))
    lr = float(lstm_cfg.get("lr", 1e-3))
    epochs = int(lstm_cfg.get("epochs", 25))

    model = MultiHeadSpreadLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Class weighting
    pos_count = np.sum(y_tr)
    neg_count = len(y_tr) - pos_count
    pos_weight = torch.tensor([neg_count / max(1.0, pos_count)])
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    smooth_l1 = nn.SmoothL1Loss()

    t_x_tr = torch.tensor(X_tr_norm, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    t_d_tr = torch.tensor(log_dur_tr, dtype=torch.float32).unsqueeze(1)

    t_x_va = torch.tensor(X_va_norm, dtype=torch.float32)
    t_y_va = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)

    best_val_loss = float("inf")
    best_weights = model.state_dict()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        p_raw, d_pred = model(t_x_tr)
        loss = F.binary_cross_entropy(p_raw, t_y_tr) + 0.5 * smooth_l1(d_pred, t_d_tr)
        loss.backward()
        optimizer.step()

        # Validation check
        model.eval()
        with torch.no_grad():
            p_va, _ = model(t_x_va)
            va_loss = F.binary_cross_entropy(p_va, t_y_va).item()
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.train()

    model.load_state_dict(best_weights)
    model.eval()

    # Threshold calibration strictly on validation set using MCC
    with torch.no_grad():
        p_va_probs, _ = model(t_x_va)
        val_probs_np = p_va_probs.squeeze().numpy()
    calibrated_threshold = tune_mcc_threshold(y_va.astype(int), val_probs_np)

    # Evaluate on held-out test events (unseen final period)
    held_out_metrics = {}
    if uncens_held:
        X_held_seq = np.array([e["seq"] for e in uncens_held], dtype=np.float32)
        y_held = np.array([e["y_cls"] for e in uncens_held], dtype=int)
        dur_held = np.array([e["duration_ms"] for e in uncens_held], dtype=float)

        X_held_norm = (X_held_seq - seq_means) / seq_stds
        with torch.no_grad():
            p_held, d_held = model(torch.tensor(X_held_norm, dtype=torch.float32))
            p_held_np = p_held.squeeze().numpy()
            d_held_np = d_held.squeeze().numpy()

        p_tr_prior = float(np.mean(y_tr))
        held_out_metrics = compute_metrics(
            y_true=y_held,
            y_prob=p_held_np,
            y_pred_mcc=(p_held_np >= calibrated_threshold).astype(int),
            p_train_prior=p_tr_prior,
        )
        held_dur_mae = float(mean_absolute_error(dur_held, np.expm1(np.clip(d_held_np, 0, 15))))
        held_out_metrics["mae_ms"] = held_dur_mae

    # Save complete production checkpoint with all required provenance metadata (Item 1)
    ckpt_dir = Path(checkpoint_path).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    git_hash = get_git_commit()
    checkpoint_payload = {
        "dataset_version": dataset_version,
        "git_commit": git_hash,
        "git": git_hash,
        "feature_names": feat_names,
        "decision_threshold": float(calibrated_threshold),
        "threshold": float(calibrated_threshold),
        "scaler": {
            "means": seq_means.tolist(),
            "stds": seq_stds.tolist(),
        },
        "model_state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "target_label": target_label,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    torch.save(checkpoint_payload, checkpoint_path)
    logger.info("Saved complete production checkpoint to %s (Dataset: %s, Threshold: %.2f)", checkpoint_path, dataset_version, calibrated_threshold)

    return {
        "threshold": calibrated_threshold,
        "held_out_metrics": held_out_metrics,
        "checkpoint_path": checkpoint_path,
    }


def main():
    parser = argparse.ArgumentParser(description="ArbSense Unified Walk-Forward & Production Training Pipeline.")
    parser.add_argument("--db", type=str, default="data/arbsense.db", help="Path to SQLite database")
    parser.add_argument("--dataset-version", type=str, default=None, help="Dataset version (e.g. v2.1)")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/lstm_best.pt", help="Checkpoint output path")
    parser.add_argument("--results-json", type=str, default="models/checkpoints/model_results.json", help="Results JSON output path")
    args = parser.parse_args()

    cfg = AppConfig.load()
    ds_ver = args.dataset_version or cfg.raw.get("dataset", {}).get("version", "v2.1")
    tgt_label = cfg.raw.get("training", {}).get("target_label", "is_persistent_500ms")

    # Step 1: Run complete Walk-Forward validation
    wf_results = run_walkforward_pipeline(
        db_path=args.db,
        dataset_version=ds_ver,
        target_label=tgt_label,
        verbose=True,
    )

    # Step 2: Train production model and evaluate on held-out final period
    events = wf_results["events"]
    held_out = wf_results["held_out_events"]

    prod_results = train_production_model(
        events=events,
        held_out_events=held_out,
        dataset_version=ds_ver,
        target_label=tgt_label,
        checkpoint_path=args.checkpoint,
    )

    # Combine into comprehensive model results JSON for Dashboard (Item 7)
    final_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_version": ds_ver,
        "git_commit": get_git_commit(),
        "target_label": tgt_label,
        "checkpoint_path": args.checkpoint,
        "decision_threshold": prod_results["threshold"],
        "n_events": wf_results["n_events"],
        "n_folds": wf_results["n_folds"],
        "n_held_out": wf_results["n_held_out"],
        "walkforward_summary": wf_results["results_summary"],
        "duration_summary": wf_results["duration_summary"],
        "held_out_metrics": prod_results["held_out_metrics"],
    }

    out_json = Path(args.results_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info("Saved unified model results to %s", out_json)
    print("\n" + "=" * 90)
    print("TRAINING PIPELINE SUMMARY")
    print(f"Dataset Version:     {ds_ver}")
    print(f"Calibrated Threshold: {prod_results['threshold']:.2f}")
    print(f"Checkpoint Saved:    {args.checkpoint}")
    print(f"Results JSON Saved:  {args.results_json}")
    print("=" * 90)


if __name__ == "__main__":
    main()
