"""
Data loading, preprocessing, and client partitioning for UNSW-NB15.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess(
    dataset_path: str,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, int]:
    """
    Load the UNSW-NB15 CSV, encode categoricals, scale features,
    split into train/test, and reshape for Conv1d input.

    Returns
    -------
    X_train_cnn : ndarray, shape (n_train, 1, n_features)
    X_test_cnn  : ndarray, shape (n_test,  1, n_features)
    y_train     : Series
    y_test      : Series
    input_dim   : int – number of features (before reshape)
    """
    df = pd.read_csv(dataset_path)
    df.drop(columns=["id"], inplace=True, errors="ignore")

    categorical_cols = ["proto", "service", "state"]
    for col in categorical_cols:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str))

    X = df.drop(["label", "attack_cat"], axis=1, errors="ignore")
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Reshape to (samples, 1, features) for Conv1d
    X_train_cnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_cnn  = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])

    return X_train_cnn, X_test_cnn, y_train, y_test, X_train.shape[1]


def create_client_partitions(
    X: np.ndarray,
    y: pd.Series,
    num_clients: int,
) -> List[Tuple[np.ndarray, pd.Series]]:
    """
    Partition the training data into `num_clients` non-overlapping shards
    (IID, sequential split). The last client absorbs any remainder.

    Parameters
    ----------
    X          : shaped (n_samples, 1, n_features)
    y          : labels aligned with X
    num_clients: number of clients

    Returns
    -------
    List of (X_client, y_client) tuples
    """
    partition_size = len(X) // num_clients
    client_data: List[Tuple[np.ndarray, pd.Series]] = []

    for i in range(num_clients):
        start = i * partition_size
        end   = len(X) if i == num_clients - 1 else start + partition_size
        client_data.append((X[start:end], y.iloc[start:end]))

    return client_data
