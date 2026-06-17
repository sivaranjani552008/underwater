# Federated Intrusion Detection – FedAvg + CNN on UNSW-NB15

Binary network intrusion detection using **Federated Averaging (FedAvg)** with a **1D-CNN** trained on the [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) dataset.

---

## Repository Layout

```
federated-ids-cnn/
├── data/                        # Put your UNSW-NB15 CSV here (git-ignored)
├── results/                     # Auto-created: plots + metrics CSV
├── src/
│   ├── config.py                # All hyperparameters in one place
│   ├── data.py                  # Loading, preprocessing, client partitioning
│   ├── model.py                 # CNNIDSModel definition
│   ├── federated.py             # local_train, fedavg, evaluate
│   └── train.py                 # Entry point – runs the full experiment
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/federated-ids-cnn.git
cd federated-ids-cnn
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the dataset

Download `UNSW_NB15_training-set.csv` from the [official page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place it at:

```
data/UNSW_NB15_training-set.csv
```

### 3. (Optional) Adjust hyperparameters

Edit `src/config.py`:

| Variable | Default | Description |
|---|---|---|
| `NUM_CLIENTS` | 5 | Number of federated clients |
| `GLOBAL_ROUNDS` | 50 | Communication rounds |
| `LOCAL_EPOCHS` | 5 | Local SGD epochs per round |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `LEARNING_RATE` | 0.001 | Adam learning rate |

### 4. Train

```bash
cd src
python train.py
```

Results are saved to `results/`:

- `metrics.csv` – per-round accuracy, precision, recall, F1
- `accuracy.png`
- `f1_score.png`
- `precision_recall.png`

---

## Model Architecture

```
Input  (batch, 1, features)
  │
  ├─ Conv1d(1→32, k=3, pad=1) → BN → ReLU → MaxPool(2)
  ├─ Conv1d(32→64, k=3, pad=1) → BN → ReLU → MaxPool(2)
  │
  └─ Flatten → Linear(→128) → ReLU → Dropout(0.3)
             → Linear(→64)  → ReLU
             → Linear(→2)            ← logits (normal / attack)
```

---

## Federated Protocol

1. Server broadcasts global weights to all clients.
2. Each client trains locally for `LOCAL_EPOCHS` epochs.
3. Server aggregates with **FedAvg** (uniform average).
4. Repeat for `GLOBAL_ROUNDS` rounds.

---

## Dataset

| Property | Value |
|---|---|
| Name | UNSW-NB15 |
| Task | Binary classification (normal / attack) |
| Features | 42 (after dropping `id`, `attack_cat`) |
| Categorical | `proto`, `service`, `state` (label-encoded) |
| Split | 80 % train · 20 % test (stratified) |

---

## License

MIT
