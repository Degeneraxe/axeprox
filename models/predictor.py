"""Real-time AI inference predictor for ArbSense.

Loads trained PyTorch LSTM checkpoint (models/checkpoints/lstm_best.pt)
and evaluates live order book spreads in real time to predict:
1. Viability Probability (Will the spread stay open longer than execution latency?)
2. Predicted Spread Duration (in milliseconds)
"""
from __future__ import annotations

import math
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from config.settings import AppConfig
from models.features import extract_step_features, parse_feature_window
from models.lstm_model import MultiHeadSpreadLSTM

logger = logging.getLogger(__name__)


class LiveSpreadPredictor:
    """Performs sub-millisecond AI inference on live streaming order books."""

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/lstm_best.pt",
        seq_len: int = 20,
        expected_dataset_version: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.seq_len = seq_len
        try:
            self.config = AppConfig.load()
        except Exception:
            self.config = AppConfig()

        if expected_dataset_version is not None:
            self.expected_dataset_version = expected_dataset_version
        else:
            self.expected_dataset_version = self.config.raw.get("dataset", {}).get("version", "v2.1")

        self.use_latency = bool(self.config.raw.get("features", {}).get("use_latency", False))

        self.model: Optional[MultiHeadSpreadLSTM] = None
        self.device = torch.device("cpu")
        self.history: Dict[Tuple[str, str, str], List[List[float]]] = {}
        self.is_loaded = False
        self.dataset_version: str = ""
        self.git_commit: str = ""
        self.decision_threshold: float = 0.50
        self.feature_names: List[str] = []
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads trained PyTorch LSTM checkpoint requiring complete provenance metadata.
        
        Strictness guarantee (Item 1):
        - File must exist.
        - Checkpoint dictionary must contain complete metadata:
          'dataset_version', 'git_commit', 'feature_names', 'threshold' (or 'decision_threshold'),
          and 'scaler' (with 'means' and 'stds').
        - Refuses to load on missing metadata or dataset_version mismatch.
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.checkpoint_path}. "
                f"Real trained checkpoint required."
            )

        try:
            payload = torch.load(self.checkpoint_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed loading checkpoint from {self.checkpoint_path}: {e}") from e

        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid checkpoint format at {self.checkpoint_path}. Expected dict with metadata, got {type(payload)}."
            )

        # Check required metadata
        required_keys = ["dataset_version", "feature_names", "model_state_dict"]
        missing = [k for k in required_keys if k not in payload]

        # git_commit check
        if "git_commit" not in payload and "git" not in payload:
            missing.append("git_commit")

        # threshold check
        threshold_val = payload.get("threshold", payload.get("decision_threshold"))
        if threshold_val is None:
            missing.append("threshold")

        # scaler check
        scaler_dict = payload.get("scaler")
        if not isinstance(scaler_dict, dict) or "means" not in scaler_dict or "stds" not in scaler_dict:
            missing.append("scaler")

        if missing:
            raise ValueError(
                f"Checkpoint at {self.checkpoint_path} is missing required metadata: {', '.join(missing)}"
            )

        # Validate dataset_version mismatch
        ckpt_version = str(payload["dataset_version"])
        if ckpt_version != self.expected_dataset_version:
            raise ValueError(
                f"Dataset version mismatch: checkpoint is for '{ckpt_version}', but expected '{self.expected_dataset_version}'. Refusing to load."
            )

        input_dim = len(payload["feature_names"]) if payload.get("feature_names") else 14
        hidden_dim = int(payload.get("hidden_dim", 64))
        num_layers = int(payload.get("num_layers", 2))
        self.model = MultiHeadSpreadLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

        self.dataset_version = ckpt_version
        self.git_commit = str(payload.get("git_commit", payload.get("git", "")))
        self.feature_names = list(payload["feature_names"])
        self.decision_threshold = float(threshold_val)

        self.means = np.array(scaler_dict["means"], dtype=np.float32)
        self.stds = np.array(scaler_dict["stds"], dtype=np.float32)
        self.stds[self.stds < 1e-6] = 1.0

        self.is_loaded = True
        logger.info(
            "[AI Predictor] Loaded PyTorch LSTM checkpoint from %s (Dataset: %s, Threshold: %.2f, Features: %d)",
            self.checkpoint_path, self.dataset_version, self.decision_threshold, len(self.feature_names)
        )

    def predict_opportunity(
        self,
        symbol: str,
        buy_snap: Any,
        sell_snap: Any,
        gross_spread_bps: float,
        vol_buy_bps: float = 1.0,
        vol_sell_bps: float = 1.0,
        feature_window: Optional[List[List[float]]] = None,
    ) -> Tuple[bool, float, float]:
        """Infers viability and duration for a candidate cross-exchange opportunity.
        
        Uses the shared 50-tick rolling volatility function values and in-memory as-of window (Requirements 1, 2, 3).
        
        Returns:
            Tuple of:
            - is_viable_predicted: bool (conf >= decision_threshold)
            - predicted_duration_ms: float
            - viability_confidence: float (0.0 to 1.0)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"Predictor model is not loaded from {self.checkpoint_path}.")

        key = (symbol, buy_snap.exchange, sell_snap.exchange)
        if key not in self.history:
            self.history[key] = []

        step_feats = extract_step_features(
            snap_buy=buy_snap,
            snap_sell=sell_snap,
            gross_spread_bps=gross_spread_bps,
            vol_buy_bps=vol_buy_bps,
            vol_sell_bps=vol_sell_bps,
            use_latency=self.use_latency,
        )
        self.history[key].append(step_feats)
        if len(self.history[key]) > self.seq_len:
            self.history[key].pop(0)

        if feature_window is not None and len(feature_window) == self.seq_len:
            current_seq = feature_window
        else:
            current_seq = list(self.history[key])
            while len(current_seq) < self.seq_len:
                current_seq.insert(0, step_feats)

        arr_in = parse_feature_window(current_seq, use_latency=self.use_latency)

        if self.means is not None and self.stds is not None:
            arr_in = (arr_in - self.means) / self.stds

        tensor_in = torch.from_numpy(np.expand_dims(arr_in, axis=0))
        with torch.no_grad():
            pred_prob, pred_log_dur = self.model(tensor_in)
            conf = float(pred_prob.item())
            log_dur = float(pred_log_dur.item())
            dur_ms = max(50.0, float(math.expm1(np.clip(log_dur, 0, 15))))
            is_viable = conf >= self.decision_threshold

        return is_viable, dur_ms, conf
