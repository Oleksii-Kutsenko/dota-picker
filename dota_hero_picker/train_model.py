import argparse
import logging
import os
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset

import settings

from .data_preparation import (
    create_input_vector,
    hero_data,
    hero_to_id,
    heroes,
    id_to_hero,
    num_counter_pairs,
    num_heroes,
    num_synergy_pairs,
    prepare_training_data,
)
from .neural_network import (
    HeroPredictorWithEmbedding,
    HeroPredictorWithOrder,
    embedding_dim,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyTorchEstimator(BaseEstimator):
    def __init__(
        self, input_size, epochs=10, lr=0.001, batch_size=32, patience=3
    ):
        self.input_size = input_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.model = None

    def fit(self, X, y):
        self.model = HeroPredictorWithOrder(self.input_size)
        self.model = train_model(
            self.model,
            X,
            y,
            self.epochs,
            self.lr,
            self.batch_size,
            self.patience,
        )
        return self

    def predict(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            outputs = torch.sigmoid(self.model(inputs)).cpu().numpy().flatten()
        return (outputs > 0.5).astype(int)

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)


def load_decisions_from_csv(csv_file_path):
    decisions = []
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    decisions_dataframe = pd.read_csv(
        csv_file_path,
        converters={
            "full_team_picks": str,
            "team_picks": str,
            "full_opponent_picks": str,
            "opponent_picks": str,
            "picked_hero": str,
        },
        dtype={"win": int},
    )

    for index, row in decisions_dataframe.iterrows():
        full_team_picks = row["full_team_picks"].strip().split(",")
        team_picks = row["team_picks"].strip().split(",")
        full_opponent_picks = row["full_opponent_picks"].strip().split(",")
        opponent_picks = row["opponent_picks"].strip().split(",")
        picked_hero = row["picked_hero"]
        win = row["win"]

        all_picks = team_picks + opponent_picks + [picked_hero]
        invalid_heroes = [
            hero for hero in all_picks if hero and hero not in hero_to_id
        ]
        if invalid_heroes:
            raise RuntimeError(f"Invalid Heroes: {invalid_heroes}")

        decisions.append(
            (
                full_team_picks,
                team_picks,
                full_opponent_picks,
                opponent_picks,
                picked_hero,
                win,
            )
        )

    if not decisions:
        raise ValueError("No valid decisions found in the CSV file")
    return decisions


class DotaDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]

        return (
            torch.tensor(sample[0], dtype=torch.float32),
            torch.tensor(sample[1], dtype=torch.float32),
            torch.tensor(sample[2], dtype=torch.float32),
            torch.tensor(sample[3], dtype=torch.long),  # synergy pairs
            torch.tensor(sample[4], dtype=torch.long),  # synergy pairs
            torch.tensor(sample[5], dtype=torch.long),  # counter pairs
            torch.tensor(sample[6], dtype=torch.float32),  # aggregate features
            label,
        )


def collate_fn(batch):
    (
        team_vecs,
        opp_vecs,
        pick_vecs,
        team_syns,
        opp_syns,
        team_cnts,
        agg_features,
        labels,
    ) = zip(*batch)

    team_syns_padded = pad_sequence(
        team_syns, batch_first=True, padding_value=0
    )
    opp_syns_padded = pad_sequence(opp_syns, batch_first=True, padding_value=0)
    team_cnts_padded = pad_sequence(
        team_cnts, batch_first=True, padding_value=0
    )

    inputs = [
        torch.stack(team_vecs),
        torch.stack(opp_vecs),
        torch.stack(pick_vecs),
        team_syns_padded,
        opp_syns_padded,
        team_cnts_padded,
        torch.stack(agg_features),
    ]

    labels_stacked = torch.stack(labels)

    return inputs, labels_stacked


def get_data_loader(x_data, y_data, batch_size, shuffle=True):
    dataset = DotaDataset(x_data, y_data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=True,
    )


def train_model(
    model,
    x_train,
    y_train,
    epochs,
    lr,
    batch_size,
    pos_weight,
    patience=3,
    x_val=None,
    y_val=None,
):
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=2,
        factor=0.1,
    )
    mid_point = 0.5

    train_loader = get_data_loader(x_train, y_train, batch_size)

    if x_val is not None and y_val is not None:
        val_loader = get_data_loader(x_val, y_val, batch_size)

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for inputs, labels in train_loader:
            inputs = [t.to(device) for t in inputs]
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = (
                (torch.sigmoid(outputs) > mid_point)
                .float()
                .detach()
                .cpu()
                .numpy()
            )
            train_preds.extend(preds)
            train_labels.extend(labels.detach().cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_prec = precision_score(train_labels, train_preds)
        train_rec = recall_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Acc: {train_acc:.4f}, "
            f"Prec: {train_prec:.4f}, "
            f"Rec: {train_rec:.4f}, "
            f"F1: {train_f1:.4f}",
        )

        if x_val is not None:
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
                model,
                val_loader,
                criterion,
            )
            scheduler.step(val_loss)
            logger.info(
                f"Val Loss: {val_loss:.4f}, "
                f"Acc: {val_acc:.4f}, "
                f"Prec: {val_prec:.4f}, "
                f"Rec: {val_rec:.4f}, "
                f"F1: {val_f1:.4f}",
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    break

    if x_val is not None and best_weights is not None:
        model.load_state_dict(best_weights)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
            model,
            val_loader,
            criterion,
        )
        logger.info(
            f"Final Validation - Loss: {val_loss:.4f}, "
            f"Acc: {val_acc:.4f}, "
            f"Prec: {val_prec:.4f}, "
            f"Rec: {val_rec:.4f}, "
            f"F1: {val_f1:.4f}",
        )
    return model


def evaluate_model(model, loader, criterion):
    model.eval()
    val_loss = 0
    mid_point = 0.5
    preds, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = [t.to(device) for t in inputs]
            labels = labels.to(device)

            outputs = model(*inputs)

            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item()

            pred = (torch.sigmoid(outputs) > mid_point).float().cpu().numpy()
            preds.extend(pred)
            labels_list.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(loader)
    acc = accuracy_score(labels_list, preds)
    prec = precision_score(labels_list, preds, zero_division=0)
    rec = recall_score(labels_list, preds, zero_division=0)
    f1 = f1_score(labels_list, preds, zero_division=0)
    return avg_val_loss, acc, prec, rec, f1


def compute_baseline_f1(y_train, y_val):
    # Determine majority class from training data
    counter = Counter(y_train)
    majority_class = counter.most_common(1)[0][0]

    # Predict majority class for all validation samples
    y_pred_baseline = [majority_class] * len(y_val)

    # Compute F1-score for the baseline (using pos_label=1 for win prediction)
    baseline_f1 = f1_score(y_val, y_pred_baseline, pos_label=1)

    # Also compute class distribution for context
    train_dist = {k: v / len(y_train) for k, v in counter.items()}
    val_dist = Counter(y_val)
    val_dist = {k: v / len(y_val) for k, v in val_dist.items()}

    return baseline_f1, majority_class, train_dist, val_dist


def optimize_hyperparameters(x_train, y_train, x_val, y_val, input_size):
    param_grid = {
        "epochs": [10, 20, 30],
        "lr": [0.001, 0.01, 0.1],
        "batch_size": [32, 64, 128],
        "patience": [4, 5, 6],
    }

    best_f1 = -1
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        logger.info(f"Testing params: {params}")
        estimator = PyTorchEstimator(input_size=input_size, **params)
        estimator.fit(x_train, y_train)
        val_pred = estimator.predict(x_val)
        val_f1 = f1_score(y_val, val_pred)
        logger.info(f"Validation F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params
            best_model = estimator.model

    logger.info(f"Best params: {best_params} with F1: {best_f1:.4f}")
    return best_model


def main(csv_file_path, matchups_statistics_path):
    optimize = False
    epochs = 30
    lr = 0.0002
    batch_size = 32
    patience = 3

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decisions = load_decisions_from_csv(csv_file_path)

        num_of_positives = len(
            [1 for decision in decisions if decision[5] == 1]
        )
        num_of_negatives = len(decisions) - num_of_positives
        pos_weight = torch.tensor([num_of_negatives / num_of_positives]).to(
            device
        )
        logger.info("Class distribution for original data")
        logger.info(f"Number of Wins {num_of_positives}")
        logger.info(f"Number of Loses {num_of_negatives}")
        logger.info(f"Pos Weight {pos_weight}")

        X_train, X_val, y_train, y_val = prepare_training_data(decisions)
        logger.info(
            f"Loaded {len(decisions)} decisions "
            f"(Train: {len(X_train)}, Val: {len(X_val)}).",
        )

        # Compute and print baseline F1-score
        baseline_f1, majority_class, train_dist, val_dist = (
            compute_baseline_f1(y_train, y_val)
        )
        logger.info(
            "Baseline F1-score "
            f"(always predict majority class {majority_class}): "
            f"{baseline_f1:.4f}",
        )
        logger.info("Training class distribution: %s", train_dist)
        logger.info("Validation class distribution: %s", val_dist)

        model = HeroPredictorWithEmbedding(num_heroes, embedding_dim)

        if optimize:
            model = optimize_hyperparameters(
                X_train,
                y_train,
                X_val,
                y_val,
                input_size,
            )
            torch.save(model.state_dict(), "optimized_model.pth")
            logger.info(
                "Hyperparameter optimization complete! "
                "Model saved as 'optimized_model.pth'.",
            )
        else:
            model = train_model(
                model,
                X_train,
                y_train,
                epochs,
                lr,
                batch_size,
                pos_weight,
                patience,
                X_val,
                y_val,
            )
            torch.save(
                model.state_dict(),
                settings.MODELS_FOLDER_PATH / Path("trained_model.pth"),
            )
            logger.info(
                "Training complete! Model saved as 'trained_model.pth'.",
            )
    except Exception:
        logger.exception("Critical error during training")
