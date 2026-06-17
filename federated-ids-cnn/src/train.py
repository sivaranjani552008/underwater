"""
Federated Learning (FedAvg) with CNN for UNSW-NB15 intrusion detection.
Entry point: runs federated training and saves results/plots.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import (
    DATASET_PATH, NUM_CLIENTS, GLOBAL_ROUNDS,
    LOCAL_EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE
)
from data import load_and_preprocess, create_client_partitions
from model import CNNIDSModel
from federated import local_train, fedavg, evaluate


def main():
    print(f"Using device: {DEVICE}\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    X_train_cnn, X_test_cnn, y_train, y_test, input_dim = \
        load_and_preprocess(DATASET_PATH)

    print(f"Training samples : {len(y_train)}")
    print(f"Testing  samples : {len(y_test)}")
    print(f"CNN input shape  : {X_train_cnn.shape}\n")

    client_data = create_client_partitions(
        X_train_cnn, y_train, NUM_CLIENTS
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    global_model = CNNIDSModel(input_dim).to(DEVICE)
    print("CNN Architecture:")
    print(global_model, "\n")

    # ------------------------------------------------------------------
    # Federated training
    # ------------------------------------------------------------------
    print("Starting Federated Training (FedAvg + CNN)...\n")
    history = []

    for round_num in range(GLOBAL_ROUNDS):

        local_weights = []
        for client_id in range(NUM_CLIENTS):
            X_local, y_local = client_data[client_id]
            weights = local_train(
                global_model, X_local, y_local,
                LOCAL_EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE
            )
            local_weights.append(weights)

        global_weights = fedavg(local_weights)
        global_model.load_state_dict(global_weights)

        acc, prec, rec, f1 = evaluate(
            global_model, X_test_cnn, y_test, DEVICE
        )
        history.append([acc, prec, rec, f1])

        print(
            f"Round {round_num + 1:03d} | "
            f"Accuracy={acc:.4f} | Precision={prec:.4f} | "
            f"Recall={rec:.4f} | F1={f1:.4f}"
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    results = pd.DataFrame(
        history, columns=["Accuracy", "Precision", "Recall", "F1"]
    )

    print("\nFinal Performance (last 5 rounds):")
    print(results.tail())
    print(f"\nBest Accuracy : {results['Accuracy'].max():.4f}")
    print(f"Best F1 Score : {results['F1'].max():.4f}")

    results.to_csv("results/metrics.csv", index=False)
    _plot_results(results)


def _plot_results(results: pd.DataFrame) -> None:
    import os
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(results["Accuracy"], color="steelblue")
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy")
    plt.title("FedAvg CNN – UNSW-NB15 Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(results["F1"], color="darkorange")
    plt.xlabel("Global Round")
    plt.ylabel("F1 Score")
    plt.title("FedAvg CNN – UNSW-NB15 F1 Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/f1_score.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(results["Precision"], label="Precision", color="green")
    plt.plot(results["Recall"],    label="Recall",    color="red")
    plt.xlabel("Global Round")
    plt.ylabel("Score")
    plt.title("FedAvg CNN – UNSW-NB15 Precision & Recall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/precision_recall.png", dpi=150)
    plt.close()

    print("\nPlots saved to results/")


if __name__ == "__main__":
    main()
