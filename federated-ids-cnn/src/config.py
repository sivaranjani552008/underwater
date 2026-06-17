"""
Central configuration for the FedAvg CNN experiment.
Edit these values to change hyperparameters or dataset path.
"""

import torch

# ── Dataset ────────────────────────────────────────────────────────────
DATASET_PATH: str = "data/UNSW_NB15_training-set.csv"

# ── Federated Learning ─────────────────────────────────────────────────
NUM_CLIENTS: int   = 5
GLOBAL_ROUNDS: int = 50
LOCAL_EPOCHS: int  = 5

# ── Training ───────────────────────────────────────────────────────────
BATCH_SIZE: int      = 32
LEARNING_RATE: float = 0.001

# ── Device ─────────────────────────────────────────────────────────────
DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
