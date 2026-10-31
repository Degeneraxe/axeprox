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
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self._load_model()

    def _load_model(self) -> None:
        if self.scaler_path.exists():
            try:
                import json
                with open(self.scaler_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.means = np.array(data["means"], dtype=np.float32)
                self.stds = np.array(data["stds"], dtype=np.float32)
                self.stds[self.stds < 1e-6] = 1.0
            except Exception as e:
                logger.warning("[AI Predictor] Could not load scaler: %s", e)

        if self.checkpoint_path.exists():
            try:
                self.model = MultiHeadSpreadLSTM(input_dim=14, hidden_dim=64, num_layers=2)
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_loaded = True
                logger.info("[AI Predictor] Loaded PyTorch LSTM checkpoint from %s", self.checkpoint_path)
            except Exception as e:
                logger.error("[AI Predictor] Failed loading checkpoint: %s", e)
                self.is_loaded = False
        else:
            logger.warning("[AI Predictor] Checkpoint not found at %s. Operating in heuristic mode.", self.checkpoint_path)

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
            - is_viable_predicted: bool
            - predicted_duration_ms: float
            - viability_confidence: float (0.0 to 1.0)
        """
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

        # If model is loaded, run neural inference
        if self.is_loaded and self.model is not None:
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
                is_viable = conf >= 0.40

            return is_viable, dur_ms, conf

        # Fallback heuristic if checkpoint not loaded
        is_viable = gross_spread_bps > 1.0
        dur_ms = 350.0 if is_viable else 45.0
        conf = 0.85 if is_viable else 0.30
        return is_viable, dur_ms, conf
