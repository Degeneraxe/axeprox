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

from models.features import extract_step_features
from models.lstm_model import MultiHeadSpreadLSTM

logger = logging.getLogger(__name__)


class LiveSpreadPredictor:
    """Performs sub-millisecond AI inference on live streaming order books."""

    def __init__(self, checkpoint_path: str = "models/checkpoints/lstm_best.pt", seq_len: int = 20):
        self.checkpoint_path = Path(checkpoint_path)
        self.scaler_path = self.checkpoint_path.parent / "scaler_params.json"
        self.seq_len = seq_len
        self.model: Optional[MultiHeadSpreadLSTM] = None
        self.device = torch.device("cpu")
        self.history: Dict[Tuple[str, str, str], List[List[float]]] = {}
        self.is_loaded = False
        self.decision_threshold: float = 0.50
        self.feature_names: List[str] = []
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads trained PyTorch LSTM checkpoint with unified decision threshold, scaler, and feature list.
        
        Strictness guarantee (Defect 5):
        Heuristic fallbacks are disabled. Raises FileNotFoundError if checkpoint is missing.
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.checkpoint_path}. "
                f"Heuristic fallback is disabled; real trained checkpoint required."
            )

        try:
            self.model = MultiHeadSpreadLSTM(input_dim=14, hidden_dim=64, num_layers=2)
            payload = torch.load(self.checkpoint_path, map_location=self.device)

            if isinstance(payload, dict) and "model_state_dict" in payload:
                self.model.load_state_dict(payload["model_state_dict"])
                self.decision_threshold = float(payload.get("decision_threshold", 0.50))
                scaler_dict = payload.get("scaler", {})
                if "means" in scaler_dict and "stds" in scaler_dict:
                    self.means = np.array(scaler_dict["means"], dtype=np.float32)
                    self.stds = np.array(scaler_dict["stds"], dtype=np.float32)
                    self.stds[self.stds < 1e-6] = 1.0
                self.feature_names = payload.get("feature_names", [])
            else:
                self.model.load_state_dict(payload)
                self.decision_threshold = 0.50

            self.model.eval()
            self.is_loaded = True
            logger.info(
                "[AI Predictor] Loaded PyTorch LSTM checkpoint from %s (Threshold: %.2f)",
                self.checkpoint_path, self.decision_threshold
            )
        except Exception as e:
            logger.error("[AI Predictor] Failed loading checkpoint: %s", e)
            self.is_loaded = False
            raise RuntimeError(f"Failed loading checkpoint from {self.checkpoint_path}: {e}") from e

    def predict_opportunity(
        self,
        symbol: str,
        buy_snap: Any,
        sell_snap: Any,
        gross_spread_bps: float,
    ) -> Tuple[bool, float, float]:
        """Infers viability and duration for a candidate cross-exchange opportunity.
        
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
        )
        self.history[key].append(step_feats)
        if len(self.history[key]) > self.seq_len:
            self.history[key].pop(0)

        current_seq = list(self.history[key])
        while len(current_seq) < self.seq_len:
            current_seq.insert(0, step_feats)

        arr_in = np.array(current_seq, dtype=np.float32)
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
