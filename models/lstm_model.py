"""Multi-head PyTorch LSTM architecture for spread-duration regression and viability classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadSpreadLSTM(nn.Module):
    """Deep sequence model predicting arbitrage viability and time-to-closure."""

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM sequence backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification Head: Viability probability in [0, 1]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Regression Head: Predicted log(1 + duration_ms)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of:
            - pred_viability: (batch_size, 1)
            - pred_log_duration: (batch_size, 1)
        """
        # lstm_out: (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Take representation at final sequence timestep
        h_last = lstm_out[:, -1, :]

        pred_viability = self.classifier(h_last)
        pred_log_dur = self.regressor(h_last)

        return pred_viability, pred_log_dur
