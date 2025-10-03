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
from torch.utils.data import DataLoader, TensorDataset

import settings

from .load_dota_matches import get_hero_data
from .neural_network import (
    HeroPredictorWithOrder,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger()

hero_data = get_hero_data()
heroes = [hero_data_item["localized_name"] for hero_data_item in hero_data]
num_heroes = len(heroes)
hero_to_id = {hero: idx for idx, hero in enumerate(heroes)}
id_to_hero = {idx: hero for hero, idx in hero_to_id.items()}

synergy = pd.read_csv(settings.MATCHUPS_STATISTICS_PATH / Path("synergy.csv"))
counters = pd.read_csv(
    settings.MATCHUPS_STATISTICS_PATH / Path("counters.csv")
)


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


def create_input_vector(team_picks, opponent_picks, actual_pick):
    # Vector for your team's heroes (size = num_heroes)
    team_vec = np.zeros(num_heroes)
    for index, hero in enumerate(team_picks, start=1):
        if hero and hero in hero_to_id:
            team_vec[hero_to_id[hero]] = index

    # Vector for opponent's heroes (size = num_heroes)
    opponent_vec = np.zeros(num_heroes)
    for index, hero in enumerate(opponent_picks, start=1):
        if hero and hero in hero_to_id:
            opponent_vec[hero_to_id[hero]] = index

    # One-hot vector for the hero pick being evaluated
    pick_vec = np.zeros(num_heroes)
    if actual_pick in hero_to_id:
        pick_vec[hero_to_id[actual_pick]] = len(team_picks) + 1

    synergy_vector = np.zeros(num_heroes)
    counter_vector = np.zeros(num_heroes)
    all_picks = [pick for pick in (team_picks + opponent_picks) if pick]
    for first_hero, second_hero in combinations(sorted(all_picks), 2):
        h1, h2 = sorted([hero_to_id[first_hero], hero_to_id[second_hero]])

        s_score_series = synergy[
            (synergy["hero_id_1"] == h1) & (synergy["hero_id_2"] == h2)
        ]["score"]
        c_score_series = counters[
            (counters["hero_id_1"] == h1) & (counters["hero_id_2"] == h2)
        ]["score"]

        if not s_score_series.empty:
            s_score = s_score_series.iloc[0]
            synergy_vector[hero_to_id[first_hero]] += s_score
            synergy_vector[hero_to_id[second_hero]] += s_score

        if not c_score_series.empty:
            c_score = c_score_series.iloc[0]
            counter_vector[hero_to_id[first_hero]] += c_score
            counter_vector[hero_to_id[second_hero]] += c_score
    return np.concatenate(
        [team_vec, opponent_vec, pick_vec, synergy_vector, counter_vector]
    )


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


def augment_decision_samples(decision, create_input_vector):
    (
        full_team_picks,
        team_picks,
        full_opponent_picks,
        opponent_picks,
        picked_hero,
        win,
    ) = decision

    augmented_samples = []

    own_picks_list = [pick for pick in full_team_picks]
    enemy_picks_list = [pick for pick in full_opponent_picks]

    total_own = len(own_picks_list)
    total_enemy = len(enemy_picks_list)

    # Your team's perspective: Cumulative with phased opponent inclusion
    for i in range(total_own):
        # Cumulative own picks up to but not including current (prior knowledge)
        prior_own_picks = own_picks_list[:i]

        # Phased opponent picks: Simulate Dota phases (empty early, then add)
        if i == 0:
            prior_enemy_picks = []  # No opponents visible yet
        elif i <= 2:
            prior_enemy_picks = enemy_picks_list[
                : min(2, total_enemy)
            ]  # Early phase: up to 2
        else:
            prior_enemy_picks = enemy_picks_list[
                : min(4, total_enemy)
            ]  # Later: up to 4

        # Actual pick is the current one
        actual_pick = own_picks_list[i]

        vec = create_input_vector(
            prior_own_picks, prior_enemy_picks, actual_pick
        )
        augmented_samples.append((vec, win))

    # Opponent's perspective: Swap and invert win
    inverted_win = 1 - win
    own_picks_list_inv = enemy_picks_list
    enemy_picks_list_inv = own_picks_list
    total_own_inv = len(own_picks_list_inv)
    total_enemy_inv = len(enemy_picks_list_inv)

    for i in range(total_own_inv):
        prior_own_picks = own_picks_list_inv[:i]

        if i == 0:
            prior_enemy_picks = []
        elif i <= 2:
            prior_enemy_picks = enemy_picks_list_inv[: min(2, total_enemy_inv)]
        else:
            prior_enemy_picks = enemy_picks_list_inv[: min(4, total_enemy_inv)]

        actual_pick = own_picks_list_inv[i]

        vec = create_input_vector(
            prior_own_picks, prior_enemy_picks, actual_pick
        )
        augmented_samples.append((vec, inverted_win))

    return augmented_samples


def prepare_training_data(decisions, test_size=0.2):
    x = []
    y = []

    for (
        full_team_picks,
        team_picks,
        full_opponent_picks,
        opponent_picks,
        picked_hero,
        win,
    ) in decisions:
        input_vec = create_input_vector(
            team_picks, opponent_picks, picked_hero
        )
        x.append(input_vec)
        y.append(win)

        augmented = augment_decision_samples(
            (
                full_team_picks,
                team_picks,
                full_opponent_picks,
                opponent_picks,
                picked_hero,
                win,
            ),
            create_input_vector,
        )
        for sample_vec, label in augmented:
            x.append(sample_vec)
            y.append(label)

    logger.info(f"Augmented dataset size before split: {len(x)}")

    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_index, val_index in strat_split.split(x, y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

    return x_train, x_val, y_train, y_val


def get_data_loader(x, y, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
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
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
    preds, labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            pred = (
                (torch.sigmoid(outputs) > mid_point)
                .float()
                .detach()
                .cpu()
                .numpy()
            )
            preds.extend(pred)
            labels.extend(targets.detach().cpu().numpy())

    avg_val_loss = val_loss / len(loader)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
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
    lr = 0.001
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

        input_size = num_heroes * 5
        model = HeroPredictorWithOrder(input_size)

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
