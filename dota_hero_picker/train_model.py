import ast
import logging
import os
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import settings

from .data_preparation import (
    api_id_2_model_id,
    create_augmented_dataframe,
    num_heroes,
    prepare_dataframe,
)
from .neural_network import (
    RecommenderWithPositionalAttention,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_ranking_metrics(
    logits: np.ndarray, targets: np.ndarray, k: int = 5
) -> tuple[float, float, float, float, float, float]:
    """Compute ranking metrics: Top-k Acc, MRR, Prec@k, Rec@k, F1@k, Macro F1."""
    top_k_preds = np.argpartition(logits, -k, axis=1)[:, -k:]
    targets_exp = targets[:, np.newaxis]
    is_in_top_k = np.isin(targets_exp, top_k_preds).any(axis=1)
    top_k_acc = np.mean(is_in_top_k)

    mrr = 0.0
    num_samples = len(targets)
    for i in range(num_samples):
        ranked_indices = np.argsort(logits[i])[::-1]
        rank_pos = np.where(ranked_indices == targets[i])[0]
        if len(rank_pos) > 0:
            mrr += 1.0 / (rank_pos[0] + 1)
    mrr /= num_samples

    precision_k = np.sum(is_in_top_k) / (num_samples * k)
    recall_k = top_k_acc
    f1_k = (
        2 * (precision_k * recall_k) / (precision_k + recall_k)
        if (precision_k + recall_k) > 0
        else 0.0
    )

    top1_preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(targets, top1_preds, average="macro", zero_division=0)

    return top_k_acc, mrr, precision_k, recall_k, f1_k, macro_f1


def load_personal_matches(csv_file_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    matches_dataframe = pd.read_csv(
        csv_file_path,
        converters={
            "team_picks": str,
            "opponent_picks": str,
        },
        dtype={
            "win": int,
            "picked_hero": int,
        },
    )
    to_split_columns = [
        "team_picks",
        "opponent_picks",
    ]
    for column in to_split_columns:
        matches_dataframe[column] = matches_dataframe[column].apply(
            ast.literal_eval
        )
        matches_dataframe[column] = matches_dataframe[column].apply(
            lambda hero_list: [
                api_id_2_model_id[api_id] for api_id in hero_list
            ]
        )
    matches_dataframe["picked_hero"] = matches_dataframe["picked_hero"].apply(
        lambda api_id: api_id_2_model_id[api_id]
    )

    return matches_dataframe


class DotaDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, max_picks_per_team: int = 5):
        self.dataframe = dataframe
        self.max_len = max_picks_per_team

    def __len__(self):
        """Return dataset length."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple:
        """Return training example."""
        row = self.dataframe.iloc[idx]
        team_picks_padded = row["visible_team_picks"] + [0] * (
            self.max_len - len(row["visible_team_picks"])
        )
        opp_picks_padded = row["visible_opp_picks"] + [0] * (
            self.max_len - len(row["visible_opp_picks"])
        )
        return (
            torch.tensor(team_picks_padded, dtype=torch.long),
            torch.tensor(opp_picks_padded, dtype=torch.long),
            torch.tensor(row["actual_pick"], dtype=torch.long),
            torch.tensor(row["win"], dtype=torch.float),
            torch.tensor(row.get("is_my_decision", 0.0), dtype=torch.float),
        )


def get_data_loader(
    dataset: Dataset, batch_size: int, shuffle: bool = True
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@dataclass
class TrainingArguments:
    """Stores data for training."""

    train_dataset: DotaDataset
    val_dataset: DotaDataset
    epochs: int = 60
    lr: float = 0.00001
    batch_size: int = 32
    patience: int = 3


def evaluate_model(
    model: RecommenderWithPositionalAttention,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    k: int = 5,
) -> tuple[float, float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch_data in loader:
            team_picks, opp_picks, targets, _, _ = [
                t.to(device) for t in batch_data
            ]

            outputs = model(team_picks, opp_picks)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * len(targets)

            all_logits.append(outputs.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / len(loader.dataset)

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_targets = torch.cat(all_targets).numpy()

    top_k_acc, mrr, prec_k, rec_k, f1_k, macro_f1 = compute_ranking_metrics(
        all_logits, all_targets, k
    )

    return avg_loss, top_k_acc, mrr, prec_k, rec_k, f1_k, macro_f1


def train_model(
    model: RecommenderWithPositionalAttention,
    training_arguments: TrainingArguments,
) -> RecommenderWithPositionalAttention:
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=training_arguments.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )

    train_loader = get_data_loader(
        training_arguments.train_dataset, training_arguments.batch_size
    )
    val_loader = get_data_loader(
        training_arguments.val_dataset,
        training_arguments.batch_size,
        shuffle=False,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(training_arguments.epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_logits = []
        epoch_train_labels = []

        for batch_data in train_loader:
            team_picks, opp_picks, targets, is_win, is_my_decision = [
                t.to(device) for t in batch_data
            ]

            optimizer.zero_grad()
            outputs = model(team_picks, opp_picks)
            per_sample_loss = criterion(outputs, targets)

            weights = torch.ones_like(per_sample_loss)
            weights += is_win * 1.0
            weights += (is_win * is_my_decision) * 2.0
            weights -= ((1.0 - is_win) * is_my_decision) * 0.5
            weights = torch.clamp(weights, min=0.1)

            loss = (per_sample_loss * weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_logits.append(outputs.detach().cpu())
            epoch_train_labels.extend(targets.cpu().numpy())

        avg_train_loss = epoch_train_loss / len(train_loader)
        all_train_logits = torch.cat(epoch_train_logits, dim=0).numpy()
        all_train_targets = np.array(epoch_train_labels)
        (
            train_top5,
            train_mrr,
            train_prec5,
            train_rec5,
            train_f15,
            train_macro_f1,
        ) = compute_ranking_metrics(all_train_logits, all_train_targets, k=5)

        logger.info(
            f"Epoch {epoch + 1}/{training_arguments.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Top-5 Acc: {train_top5:.4f} | "
            f"MRR: {train_mrr:.4f} | Prec@5: {train_prec5:.4f} | "
            f"Rec@5: {train_rec5:.4f} | F1: {train_f15:.4f}"
        )

        (
            val_loss,
            val_top5,
            val_mrr,
            val_prec5,
            val_rec5,
            val_f15,
            val_macro_f1,
        ) = evaluate_model(model, val_loader, nn.CrossEntropyLoss())
        scheduler.step(val_loss)

        logger.info(
            f"Val Log Loss: {val_loss:.4f} | Top-5 Acc: {val_top5:.4f} | "
            f"MRR: {val_mrr:.4f} | Val Prec@5: {val_prec5:.4f} | "
            f"Val Rec@5: {val_rec5:.4f} | Val F1: {val_f15:.4f} | Val Macro F1: {val_macro_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = deepcopy(model.state_dict())
            logger.info(
                f"New best validation loss: {best_val_loss:.4f}. Saving model weights."
            )
        else:
            patience_counter += 1
            if patience_counter >= training_arguments.patience:
                logger.info(
                    f"Early stopping triggered after {patience_counter} epochs of no improvement."
                )
                break

    if best_weights:
        model.load_state_dict(best_weights)
        logger.info(
            f"Loaded best model weights (val_loss: {best_val_loss:.4f}) for final use."
        )

    return model


def compute_baseline_f1(
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple[
    float,
    int,
]:
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(pd.DataFrame(index=y_train.index), y_train)
    y_pred_baseline = baseline_model.predict(pd.DataFrame(index=y_val.index))
    f1 = f1_score(y_val, y_pred_baseline, average="weighted", zero_division=0)
    majority_class = int(baseline_model.classes_[0])
    return f1, majority_class


def main(csv_file_path: str) -> None:
    """Model training entrypoint."""
    matches_dataframe = load_personal_matches(csv_file_path)

    train_dataframe, tmp_dataframe = train_test_split(
        matches_dataframe, test_size=0.2, stratify=matches_dataframe["win"]
    )
    validation_dataframe, test_dataframe = train_test_split(
        tmp_dataframe, test_size=0.5, stratify=tmp_dataframe["win"]
    )
    augmented_train_dataframe = create_augmented_dataframe(train_dataframe)
    prepared_validation_dataframe = prepare_dataframe(validation_dataframe)
    prepared_test_dataframe = prepare_dataframe(test_dataframe)

    baseline_f1, majority_class = compute_baseline_f1(
        augmented_train_dataframe["actual_pick"],
        prepared_test_dataframe["actual_pick"],
    )
    logger.info(
        "Baseline F1-score "
        f"(always predict majority class {majority_class}): "
        f"{baseline_f1:.4f}",
    )

    train_dataset = DotaDataset(augmented_train_dataframe)
    val_dataset = DotaDataset(prepared_validation_dataframe)
    test_dataset = DotaDataset(prepared_test_dataframe)

    model = RecommenderWithPositionalAttention(num_heroes)

    training_arguments = TrainingArguments(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trained_model = train_model(
        model,
        training_arguments,
    )
    torch.save(
        trained_model.state_dict(),
        settings.MODELS_FOLDER_PATH / Path("trained_model.pth"),
    )
    logger.info(
        "Training complete! Model saved as 'trained_model.pth'.",
    )
