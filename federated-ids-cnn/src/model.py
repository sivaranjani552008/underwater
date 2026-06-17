"""
1D-CNN model for binary intrusion detection (normal vs attack).
"""

import torch.nn as nn


class CNNIDSModel(nn.Module):
    """
    1D Convolutional Neural Network for intrusion detection.

    Architecture
    ------------
    Conv block 1 → Conv1d(1→32, k=3) + BN + ReLU + MaxPool(2)
    Conv block 2 → Conv1d(32→64, k=3) + BN + ReLU + MaxPool(2)
    Classifier   → Flatten → Linear(→128) → ReLU → Dropout(0.3)
                   → Linear(→64) → ReLU → Linear(→2)

    Parameters
    ----------
    input_dim : int
        Number of input features (length of the 1-D sequence).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()

        # ── Conv block 1 ────────────────────────────────────────────
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # ── Conv block 2 ────────────────────────────────────────────
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Flattened size after two MaxPool1d(2) operations
        pool1_out   = input_dim // 2
        pool2_out   = pool1_out // 2
        flatten_dim = 64 * pool2_out

        # ── Classifier head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),   # binary: 0 = normal, 1 = attack
        )

    def forward(self, x):
        # x: (batch, 1, features)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)
