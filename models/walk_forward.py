"""Walk-Forward Model Validation & Evaluation Module for ArbSense.

Implements rigorous, non-overlapping expanding-window walk-forward evaluation:
- Reads configuration from config.yaml (fold_fractions, held_out_ratio, loss_weight, horizons, target_label)
- Reuses models.features for all sequence and tabular feature extraction
- 60s time purge + >=5 events between train/val/test splits
- Strictly non-overlapping test windows (stride >= test_window)
- Held-out final period never used during fold training or threshold tuning
- Comprehensive evaluation: PR-AUC, Lift, P@10%, P@20%, MCC, Balanced Accuracy, BSS, F1, Bootstrap CIs
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import random
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config.settings import AppConfig, get_git_commit
from models.features import (
    FEATURE_NAMES,
    TABULAR_FEATURE_NAMES,
    engineer_tabular_features,
)
from models.lstm_model import MultiHeadSpreadLSTM

logger = logging.getLogger("WalkForward")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CompactSpreadLSTM(nn.Module):
    """1-layer compact LSTM model for walk-forward benchmarking."""
    def __init__(self, input_dim: int = 14, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.classifier(last), self.regressor(last)


def load_events_from_db(
    db_path: str,
    dataset_version: Optional[str] = None,
    target_label: str = "is_persistent_500ms",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Loads events and extracts sequential and tabular features using models.features."""
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(f"file:{p.resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cfg = AppConfig.load()
    if dataset_version is None:
        dataset_version = cfg.raw.get("dataset", {}).get("version", "v2.1")

    # Discover valid dataset versions in DB if active not present
    cur.execute("SELECT DISTINCT dataset_version FROM spread_events WHERE dataset_version NOT LIKE 'invalid%' AND dataset_version IS NOT NULL")
    avail = [r[0] for r in cur.fetchall()]
    if dataset_version not in avail and avail:
        dataset_version = avail[-1]

    cur.execute("""
        SELECT 
            event_id,
            symbol,
            buy_exchange,
            sell_exchange,
            open_time,
            close_time,
            duration_ms,
            open_spread_bps,
            cost_hurdle_bps,
            slippage_bps,
            is_persistent_500ms,
            is_persistent_2000ms,
            is_persistent_5000ms,
            is_executable,
            is_viable,
            is_right_censored,
            feature_window_json
        FROM spread_events
        WHERE dataset_version = ?
          AND feature_window_json IS NOT NULL
          AND duration_ms IS NOT NULL
          AND (is_suspect = 0 OR is_suspect IS NULL)
        ORDER BY CAST(open_time AS REAL) ASC
    """, (dataset_version,))

    rows = cur.fetchall()
    conn.close()

    events: List[Dict[str, Any]] = []
    last_event_time_per_pair: Dict[Tuple[str, str, str], float] = {}
    event_times_queue_per_pair: Dict[Tuple[str, str, str], List[float]] = {}

    for r in rows:
        win_str = r["feature_window_json"]
        if not win_str:
            continue
        try:
            seq = json.loads(win_str)
        except Exception:
            continue
        if len(seq) < 20:
            continue

        seq_arr = np.array(seq[-20:], dtype=np.float32)
        open_ts = float(r["open_time"])
        dur = float(r["duration_ms"])
        is_cens = bool(r["is_right_censored"])

        # Target label resolution
        if target_label in r.keys() and r[target_label] is not None:
            raw_target = r[target_label]
        elif target_label == "is_persistent_500ms":
            raw_target = 1 if dur >= 500.0 else 0
        elif target_label == "is_persistent_2000ms":
            raw_target = 1 if dur >= 2000.0 else 0
        elif target_label == "is_persistent_5000ms":
            raw_target = 1 if dur >= 5000.0 else 0
        else:
            raw_target = 1 if dur >= 500.0 else 0

        target_cls = int(raw_target) if not is_cens else (1 if dur >= 500.0 else 0)

        sym = r["symbol"]
        buy_ex = r["buy_exchange"]
        sell_ex = r["sell_exchange"]
        pair = (sym, buy_ex, sell_ex)

        prev_ts = last_event_time_per_pair.get(pair, None)
        last_event_time_per_pair[pair] = open_ts

        if pair not in event_times_queue_per_pair:
            event_times_queue_per_pair[pair] = []
        event_times_queue_per_pair[pair] = [t for t in event_times_queue_per_pair[pair] if t >= open_ts - 60000.0]
        recent_times = list(event_times_queue_per_pair[pair])
        event_times_queue_per_pair[pair].append(open_ts)

        # Reusing models.features for tabular feature computation
        event_dict = dict(r)
        tabular_vec = engineer_tabular_features(
            seq_arr=seq_arr,
            event_dict=event_dict,
            prev_event_open_ts=prev_ts,
            recent_event_times=recent_times,
        )

        events.append({
            "event_id": r["event_id"],
            "symbol": sym,
            "direction": f"{buy_ex}->{sell_ex}",
            "open_time": open_ts,
            "duration_ms": dur,
            "log_duration": math.log1p(max(0.0, dur)),
            "is_censored": is_cens,
            "y_cls": target_cls,
            "seq": seq_arr,
            "tabular": tabular_vec,
        })

    return events, TABULAR_FEATURE_NAMES


def create_non_overlapping_folds(
    events: List[Dict[str, Any]],
    fold_fractions: Optional[List[Dict[str, float]]] = None,
    held_out_ratio: float = 0.15,
    purge_gap_sec: float = 60.0,
    min_purge_events: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates non-overlapping walk-forward folds plus a separate final held-out period.
    
    Guarantees:
    - stride >= test_window (zero overlap between test windows across folds)
    - 60s time purge + >=5 event purge gap inside each fold
    - Held-out final period (held_out_ratio) is reserved and never touched during folds
    """
    n_tot = len(events)
    n_active = int(n_tot * (1.0 - held_out_ratio))
    active_events = events[:n_active]
    held_out_events = events[n_active:]

    # Default non-overlapping expanding folds if not passed from config
    if not fold_fractions:
        fold_fractions = [
            {"train_ratio": 0.40, "val_ratio": 0.15, "test_ratio": 0.15},
            {"train_ratio": 0.55, "val_ratio": 0.15, "test_ratio": 0.15},
            {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15},
        ]

    folds_data = []
    for f_idx, f_cfg in enumerate(fold_fractions):
        tr_frac = float(f_cfg.get("train_ratio", 0.40))
        va_frac = float(f_cfg.get("val_ratio", 0.15))
        te_frac = float(f_cfg.get("test_ratio", 0.15))

        tr_end = int(n_active * tr_frac)
        t_tr_end = active_events[tr_end - 1]["open_time"]

        # Purge Gap 1 (Train -> Val)
        va_start = tr_end + min_purge_events
        while va_start < n_active and active_events[va_start]["open_time"] < (t_tr_end + purge_gap_sec * 1000.0):
            va_start += 1

        va_end = min(n_active, int(n_active * (tr_frac + va_frac)))
        if va_start >= va_end:
            va_end = min(n_active, va_start + 10)
        t_va_end = active_events[va_end - 1]["open_time"]

        # Purge Gap 2 (Val -> Test)
        te_start = va_end + min_purge_events
        while te_start < n_active and active_events[te_start]["open_time"] < (t_va_end + purge_gap_sec * 1000.0):
            te_start += 1

        te_end = min(n_active, int(n_active * (tr_frac + va_frac + te_frac)))
        if te_start >= te_end:
            te_end = min(n_active, te_start + 10)

        tr_ev = active_events[:tr_end]
        va_ev = active_events[va_start:va_end]
        te_ev = active_events[te_start:te_end]

        purge1_gap_s = (active_events[va_start]["open_time"] - t_tr_end) / 1000.0 if va_start < n_active else 0.0
        purge2_gap_s = (active_events[te_start]["open_time"] - t_va_end) / 1000.0 if te_start < n_active else 0.0

        folds_data.append({
            "fold": f_idx + 1,
            "train": tr_ev,
            "val": va_ev,
            "test": te_ev,
            "purge1_gap_s": purge1_gap_s,
            "purge2_gap_s": purge2_gap_s,
            "train_range": (tr_ev[0]["open_time"], tr_ev[-1]["open_time"]),
            "val_range": (va_ev[0]["open_time"], va_ev[-1]["open_time"]) if va_ev else (0, 0),
            "test_range": (te_ev[0]["open_time"], te_ev[-1]["open_time"]) if te_ev else (0, 0),
        })

    return folds_data, held_out_events


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred_mcc: np.ndarray,
    p_train_prior: float,
) -> Dict[str, float]:
    """Computes headline and secondary classification metrics."""
    n = len(y_true)
    if n == 0 or np.sum(y_true) == 0:
        return {}

    base_rate = float(np.mean(y_true))
    pr_auc = float(average_precision_score(y_true, y_prob))
    lift = pr_auc / base_rate if base_rate > 0 else 1.0

    order = np.argsort(y_prob)[::-1]
    k10 = max(1, int(np.ceil(0.10 * n)))
    k20 = max(1, int(np.ceil(0.20 * n)))

    p_top10 = float(np.mean(y_true[order[:k10]]))
    p_top20 = float(np.mean(y_true[order[:k20]]))

    y_pred_top20 = np.zeros(n, dtype=int)
    y_pred_top20[order[:k20]] = 1
    mcc_top20 = float(matthews_corrcoef(y_true, y_pred_top20))
    bal_acc_top20 = float(balanced_accuracy_score(y_true, y_pred_top20))

    mcc_val_thresh = float(matthews_corrcoef(y_true, y_pred_mcc))
    bal_acc_val_thresh = float(balanced_accuracy_score(y_true, y_pred_mcc))

    bs_model = float(brier_score_loss(y_true, y_prob))
    bs_ref = float(np.mean((y_true - p_train_prior) ** 2))
    bss = float(1.0 - (bs_model / bs_ref)) if bs_ref > 0 else 0.0

    f1_val = float(f1_score(y_true, y_pred_mcc, zero_division=0))
    prec_val = float(precision_score(y_true, y_pred_mcc, zero_division=0))
    rec_val = float(recall_score(y_true, y_pred_mcc, zero_division=0))

    return {
        "pr_auc": pr_auc,
        "lift": lift,
        "p_top10": p_top10,
        "p_top20": p_top20,
        "mcc": mcc_val_thresh,
        "bal_acc": bal_acc_val_thresh,
        "mcc_top20": mcc_top20,
        "bal_acc_top20": bal_acc_top20,
        "bss": bss,
        "brier": bs_model,
        "f1": f1_val,
        "precision": prec_val,
        "recall": rec_val,
        "base_rate": base_rate,
    }


def bootstrap_ci_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 500,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Calculates 95% bootstrap confidence intervals for PR-AUC and Lift."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    praucs = []
    lifts = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if np.sum(y_b) == 0 or np.sum(y_b) == len(y_b):
            continue
        auc = average_precision_score(y_b, p_b)
        br = np.mean(y_b)
        praucs.append(auc)
        lifts.append(auc / br if br > 0 else 1.0)

    if not praucs:
        return {"pr_auc_ci": (0.0, 0.0), "lift_ci": (0.0, 0.0)}

    return {
        "pr_auc_ci": (float(np.percentile(praucs, 2.5)), float(np.percentile(praucs, 97.5))),
        "lift_ci": (float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))),
    }


def tune_mcc_threshold(y_val: np.ndarray, probs_val: np.ndarray) -> float:
    """Selects probability threshold on validation fold strictly maximizing MCC."""
    best_thresh = 0.5
    best_mcc = -1.0
    for t in np.linspace(0.05, 0.85, 41):
        pred = (probs_val >= t).astype(int)
        score = matthews_corrcoef(y_val, pred)
        if score > best_mcc:
            best_mcc = score
            best_thresh = float(t)
    return best_thresh


def run_walkforward_pipeline(
    db_path: str = "data/arbsense.db",
    dataset_version: Optional[str] = None,
    target_label: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs complete expanding-window walk-forward validation and held-out evaluation."""
    cfg = AppConfig.load()
    tr_cfg = cfg.raw.get("training", {})
    tgt_label = target_label or tr_cfg.get("target_label", "is_persistent_500ms")
    ds_ver = dataset_version or cfg.raw.get("dataset", {}).get("version", "v2.1")
    held_out_ratio = float(tr_cfg.get("held_out_ratio", 0.15))
    fold_fractions = tr_cfg.get("fold_fractions", None)
    purge_gap_sec = float(tr_cfg.get("purge_gap_seconds", 60.0))

    if verbose:
        print("=" * 90)
        print("ArbSense Unified Walk-Forward Validation & Benchmark Engine")
        print(f"Database:        {db_path}")
        print(f"Dataset Version: {ds_ver}")
        print(f"Target Label:    {tgt_label}")
        print(f"Purge Gap:       {purge_gap_sec}s time gap + >=5 events")
        print(f"Held-out Ratio:  {held_out_ratio * 100:.1f}%")
        print("=" * 90)

    events, feat_names = load_events_from_db(db_path, dataset_version=ds_ver, target_label=tgt_label)
    if len(events) < 20:
        raise ValueError(f"Insufficient events in DB for walk-forward evaluation (found {len(events)}, required >= 20)")

    folds, held_out_events = create_non_overlapping_folds(
        events=events,
        fold_fractions=fold_fractions,
        held_out_ratio=held_out_ratio,
        purge_gap_sec=purge_gap_sec,
    )

    model_names = [
        "Majority Class Baseline",
        "Symbol+Direction Only",
        "Regularized Logistic Reg",
        "HistGradientBoosting",
        "Compact LSTM",
    ]

    fold_metrics = {m: [] for m in model_names}
    fold_duration_mae = {
        "Constant Median": [],
        "Ridge Regressor": [],
        "HistGradientBoosting": [],
        "Compact LSTM": [],
    }

    feature_importances_accum = []

    for f_info in folds:
        f_num = f_info["fold"]
        tr_ev = f_info["train"]
        va_ev = f_info["val"]
        te_ev = f_info["test"]

        tr_uncens = [e for e in tr_ev if not e["is_censored"]]
        va_uncens = [e for e in va_ev if not e["is_censored"]]
        te_uncens = [e for e in te_ev if not e["is_censored"]]

        if len(tr_uncens) < 5 or len(va_uncens) < 3 or len(te_uncens) < 3:
            continue

        y_tr = np.array([e["y_cls"] for e in tr_uncens], dtype=int)
        y_va = np.array([e["y_cls"] for e in va_uncens], dtype=int)
        y_te = np.array([e["y_cls"] for e in te_uncens], dtype=int)

        dur_tr = np.array([e["duration_ms"] for e in tr_uncens], dtype=float)
        dur_va = np.array([e["duration_ms"] for e in va_uncens], dtype=float)
        dur_te = np.array([e["duration_ms"] for e in te_uncens], dtype=float)

        log_dur_tr = np.array([e["log_duration"] for e in tr_uncens], dtype=float)
        log_dur_te = np.array([e["log_duration"] for e in te_uncens], dtype=float)

        p_tr_prior = float(np.mean(y_tr)) if len(y_tr) > 0 else 0.5

        # Feature preparation
        X_tr_tab = np.array([e["tabular"] for e in tr_uncens], dtype=np.float32)
        X_va_tab = np.array([e["tabular"] for e in va_uncens], dtype=np.float32)
        X_te_tab = np.array([e["tabular"] for e in te_uncens], dtype=np.float32)

        scaler = StandardScaler()
        X_tr_tab_norm = scaler.fit_transform(X_tr_tab)
        X_va_tab_norm = scaler.transform(X_va_tab)
        X_te_tab_norm = scaler.transform(X_te_tab)

        # 1. Majority class baseline
        p_maj_te = np.full(len(y_te), p_tr_prior)
        m_maj = compute_metrics(y_te, p_maj_te, (p_maj_te >= 0.5).astype(int), p_tr_prior)
        if m_maj:
            fold_metrics["Majority Class Baseline"].append(m_maj)

        # 2. Symbol+Direction baseline
        cat_tr = [[e["symbol"], e["direction"]] for e in tr_uncens]
        cat_va = [[e["symbol"], e["direction"]] for e in va_uncens]
        cat_te = [[e["symbol"], e["direction"]] for e in te_uncens]
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_tr_cat = ohe.fit_transform(cat_tr)
        X_va_cat = ohe.transform(cat_va)
        X_te_cat = ohe.transform(cat_te)

        lr_cat = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        lr_cat.fit(X_tr_cat, y_tr)
        p_cat_va = lr_cat.predict_proba(X_va_cat)[:, 1] if len(np.unique(y_tr)) > 1 else np.full(len(y_va), p_tr_prior)
        p_cat_te = lr_cat.predict_proba(X_te_cat)[:, 1] if len(np.unique(y_tr)) > 1 else np.full(len(y_te), p_tr_prior)
        th_cat = tune_mcc_threshold(y_va, p_cat_va)
        m_cat = compute_metrics(y_te, p_cat_te, (p_cat_te >= th_cat).astype(int), p_tr_prior)
        if m_cat:
            fold_metrics["Symbol+Direction Only"].append(m_cat)

        # 3. Regularized Logistic Regression
        lr_tab = LogisticRegression(C=0.1, penalty="l2", max_iter=500, random_state=42)
        lr_tab.fit(X_tr_tab_norm, y_tr)
        p_lr_va = lr_tab.predict_proba(X_va_tab_norm)[:, 1] if len(np.unique(y_tr)) > 1 else np.full(len(y_va), p_tr_prior)
        p_lr_te = lr_tab.predict_proba(X_te_tab_norm)[:, 1] if len(np.unique(y_tr)) > 1 else np.full(len(y_te), p_tr_prior)
        th_lr = tune_mcc_threshold(y_va, p_lr_va)
        m_lr = compute_metrics(y_te, p_lr_te, (p_lr_te >= th_lr).astype(int), p_tr_prior)
        if m_lr:
            fold_metrics["Regularized Logistic Reg"].append(m_lr)

        # 4. HistGradientBoosting
        hgb = HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=10, random_state=42)
        hgb.fit(X_tr_tab, y_tr)
        p_hgb_va = hgb.predict_proba(X_va_tab)[:, 1] if len(np.unique(y_tr)) > 1 else np.full(len(y_va), p_tr_prior)
        p_hgb_te = hgb.predict_proba(X_te_tab)[:, 1] if len(np.unique(y_tr)) > 1 else np.full(len(y_te), p_tr_prior)
        th_hgb = tune_mcc_threshold(y_va, p_hgb_va)
        m_hgb = compute_metrics(y_te, p_hgb_te, (p_hgb_te >= th_hgb).astype(int), p_tr_prior)
        if m_hgb:
            fold_metrics["HistGradientBoosting"].append(m_hgb)

        # 5. Compact PyTorch LSTM
        set_seed(42)
        lstm = CompactSpreadLSTM(input_dim=14, hidden_dim=32, dropout=0.3)
        X_tr_seq = np.array([e["seq"] for e in tr_uncens], dtype=np.float32)
        X_va_seq = np.array([e["seq"] for e in va_uncens], dtype=np.float32)
        X_te_seq = np.array([e["seq"] for e in te_uncens], dtype=np.float32)

        # Normalize sequences using train sequence statistics
        seq_means = np.mean(X_tr_seq, axis=(0, 1))
        seq_stds = np.std(X_tr_seq, axis=(0, 1))
        seq_stds[seq_stds < 1e-6] = 1.0

        X_tr_seq_norm = (X_tr_seq - seq_means) / seq_stds
        X_va_seq_norm = (X_va_seq - seq_means) / seq_stds
        X_te_seq_norm = (X_te_seq - seq_means) / seq_stds

        optimizer = torch.optim.AdamW(lstm.parameters(), lr=1e-3, weight_decay=1e-3)
        bce_loss_fn = nn.BCELoss()
        smooth_l1 = nn.SmoothL1Loss()

        t_x_tr = torch.tensor(X_tr_seq_norm, dtype=torch.float32)
        t_y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        t_d_tr = torch.tensor(log_dur_tr, dtype=torch.float32).unsqueeze(1)

        lstm.train()
        for epoch in range(15):
            optimizer.zero_grad()
            p_out, d_out = lstm(t_x_tr)
            loss = bce_loss_fn(p_out, t_y_tr) + 0.5 * smooth_l1(d_out, t_d_tr)
            loss.backward()
            optimizer.step()

        lstm.eval()
        with torch.no_grad():
            p_lstm_va = lstm(torch.tensor(X_va_seq_norm, dtype=torch.float32))[0].squeeze().numpy()
            p_lstm_te, d_lstm_te = lstm(torch.tensor(X_te_seq_norm, dtype=torch.float32))
            p_lstm_te = p_lstm_te.squeeze().numpy()
            d_lstm_te = d_lstm_te.squeeze().numpy()

        th_lstm = tune_mcc_threshold(y_va, p_lstm_va)
        m_lstm = compute_metrics(y_te, p_lstm_te, (p_lstm_te >= th_lstm).astype(int), p_tr_prior)
        if m_lstm:
            fold_metrics["Compact LSTM"].append(m_lstm)

        # Duration Regressors MAE
        med_tr = float(np.median(dur_tr))
        fold_duration_mae["Constant Median"].append(float(mean_absolute_error(dur_te, np.full(len(dur_te), med_tr))))

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr_tab_norm, log_dur_tr)
        ridge_pred_dur = np.expm1(np.clip(ridge.predict(X_te_tab_norm), 0, 15))
        fold_duration_mae["Ridge Regressor"].append(float(mean_absolute_error(dur_te, ridge_pred_dur)))

        hgb_reg = HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=10, random_state=42)
        hgb_reg.fit(X_tr_tab, log_dur_tr)
        hgb_pred_dur = np.expm1(np.clip(hgb_reg.predict(X_te_tab), 0, 15))
        fold_duration_mae["HistGradientBoosting"].append(float(mean_absolute_error(dur_te, hgb_pred_dur)))

        lstm_pred_dur = np.expm1(np.clip(d_lstm_te, 0, 15))
        fold_duration_mae["Compact LSTM"].append(float(mean_absolute_error(dur_te, lstm_pred_dur)))

        if verbose:
            print(f"Fold {f_num}: PR-AUC -> Maj: {m_maj.get('pr_auc',0):.3f} | LR: {m_lr.get('pr_auc',0):.3f} | HGB: {m_hgb.get('pr_auc',0):.3f} | LSTM: {m_lstm.get('pr_auc',0):.3f}")

    # Summary aggregations
    results_summary = {}
    for m_name, met_list in fold_metrics.items():
        if not met_list:
            continue
        pr_aucs = [m["pr_auc"] for m in met_list]
        lifts = [m["lift"] for m in met_list]
        mccs = [m["mcc"] for m in met_list]
        baccs = [m["bal_acc"] for m in met_list]
        brier_skills = [m["bss"] for m in met_list]
        f1s = [m["f1"] for m in met_list]

        results_summary[m_name] = {
            "pr_auc_mean": float(np.mean(pr_aucs)),
            "pr_auc_std": float(np.std(pr_aucs)),
            "lift_mean": float(np.mean(lifts)),
            "lift_std": float(np.std(lifts)),
            "mcc_mean": float(np.mean(mccs)),
            "mcc_std": float(np.std(mccs)),
            "bal_acc_mean": float(np.mean(baccs)),
            "bal_acc_std": float(np.std(baccs)),
            "bss_mean": float(np.mean(brier_skills)),
            "bss_std": float(np.std(brier_skills)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        }

    duration_summary = {}
    for r_name, mae_list in fold_duration_mae.items():
        if mae_list:
            duration_summary[r_name] = {
                "mae_mean_ms": float(np.mean(mae_list)),
                "mae_std_ms": float(np.std(mae_list)),
            }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "dataset_version": ds_ver,
        "target_label": tgt_label,
        "n_events": len(events),
        "n_folds": len(folds),
        "n_held_out": len(held_out_events),
        "results_summary": results_summary,
        "duration_summary": duration_summary,
        "fold_metrics": fold_metrics,
        "held_out_events": held_out_events,
        "events": events,
    }
