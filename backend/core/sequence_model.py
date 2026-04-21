"""
LSTM model for temporal compensatory‑movement detection and a
run‑time sequence buffer for streaming inference.
"""

import numpy as np
import torch
import torch.nn as nn


class LSTMCompensationDetector(nn.Module):
    """Bidirectional LSTM → FC classifier (binary: compensatory or not)."""

    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),   # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x : (batch, seq_len, input_size) → (batch, 1)"""
        _, (h_n, _) = self.lstm(x)
        # h_n : (num_layers*2, batch, hidden)  — take last layer fwd + bwd
        last_fwd = h_n[-2]
        last_bwd = h_n[-1]
        hidden = torch.cat([last_fwd, last_bwd], dim=1)
        return self.classifier(hidden)


class SequenceBuffer:
    """Sliding‑window buffer that collects per‑frame features and
    produces fixed‑length sequences ready for LSTM inference."""

    def __init__(self, seq_length=30, feature_names=None):
        self.seq_length = seq_length
        self.feature_names = feature_names or [
            "right_shoulder_angle",
            "left_shoulder_angle",
            "trunk_lean",
            "right_elbow_angle",
            "trunk_lateral_lean",
            "shoulder_height_diff",
        ]
        self.buffer: list[list[float]] = []

    def add_frame(self, features_dict: dict):
        row = [features_dict.get(n, 0.0) for n in self.feature_names]
        self.buffer.append(row)
        if len(self.buffer) > self.seq_length:
            self.buffer = self.buffer[-self.seq_length :]

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.seq_length

    def get_sequence(self) -> torch.Tensor | None:
        if not self.is_ready():
            return None
        arr = np.array(self.buffer[-self.seq_length :], dtype=np.float32)
        return torch.FloatTensor(arr).unsqueeze(0)  # (1, seq, feat)

    def reset(self):
        self.buffer.clear()
