"""
Federated learning utilities: local training, FedAvg aggregation, evaluation.
"""

from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)

from model import CNNIDSModel


# ── Local training ──────────────────────────────────────────────────────

def local_train(
    global_model: CNNIDSModel,
    X_local: np.ndarray,
    y_local: pd.Series,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> OrderedDict:
    """
    Train a local copy of the global model on a client's data shard.

    Parameters
    ----------
    global_model  : current global model (weights are copied, not mutated)
    X_local       : shaped (n, 1, features)
    y_local       : binary labels
    local_epochs  : number of local SGD epochs
    batch_size    : mini-batch size
    learning_rate : Adam learning rate
    device        : CPU or CUDA device

    Returns
    -------
    state_dict of the locally trained model
    """
    model = deepcopy(global_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X_local, dtype=torch.float32)
    y_tensor = torch.tensor(y_local.values, dtype=torch.long)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model.state_dict()


# ── FedAvg aggregation ──────────────────────────────────────────────────

def fedavg(
    local_weights: List[OrderedDict],
) -> OrderedDict:
    """
    Average a list of client state-dicts (uniform weighting).

    Parameters
    ----------
    local_weights : list of state_dicts, one per client

    Returns
    -------
    Averaged global state_dict
    """
    global_weights = deepcopy(local_weights[0])

    for key in global_weights.keys():
        for i in range(1, len(local_weights)):
            global_weights[key] += local_weights[i][key]
        global_weights[key] = global_weights[key] / len(local_weights)

    return global_weights


# ── Evaluation ──────────────────────────────────────────────────────────

def evaluate(
    model: CNNIDSModel,
    X_test: np.ndarray,
    y_test: pd.Series,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the global model on the held-out test set.

    Returns
    -------
    (accuracy, precision, recall, f1)
    """
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = torch.argmax(model(X_tensor), dim=1).cpu().numpy()

    return (
        accuracy_score(y_test, preds),
        precision_score(y_test, preds, zero_division=0),
        recall_score(y_test, preds, zero_division=0),
        f1_score(y_test, preds, zero_division=0),
    )
