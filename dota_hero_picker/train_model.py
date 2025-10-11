import ast
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import settings

from .data_preparation import (
    api_id_2_model_id,
    create_augmented_dataframe,
    num_heroes,
    prepare_dataframe,
)
from .neural_network import (
    WinPredictorWithPositionalAttention,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


TrainingExample = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class DotaDataset(Dataset[TrainingExample]):
    def __init__(self, dataframe: pd.DataFrame, max_picks_per_team: int = 5):
        self.dataframe = dataframe
        self.max_len = max_picks_per_team

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> TrainingExample:
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
    dataset: Dataset[TrainingExample], batch_size: int, shuffle: bool = True
) -> DataLoader[DotaDataset]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@dataclass
class TrainingArguments:
    """Stores data for training."""

    train_dataset: DotaDataset
    val_dataset: DotaDataset
    epochs: int = 60
    lr: float = 0.0001
    batch_size: int = 32
    patience: int = 3


def evaluate_model(
    model: WinPredictorWithPositionalAttention,
    loader: DataLoader[DotaDataset],
    criterion: nn.BCEWithLogitsLoss,
    decision_weight: float,
) -> tuple[float, float, float, float, float]:
    model.eval()
    val_loss = 0
    mid_point = 0.5
    preds: list[float] = []
    labels_list: list[float] = []

    with torch.no_grad():
        for batch_data in loader:  # Fixed: use loader, not train_loader
            team_picks, opp_picks, actual_pick, is_win, is_my_decision = [
                t.to(device) for t in batch_data
            ]
            outputs = model(team_picks, opp_picks, actual_pick)

            # Compute weighted loss to match training (optional: set decision_weight=1.0 for unweighted)
            per_sample_loss = criterion(outputs, is_win)
            mask = is_my_decision == 1.0  # Boolean mask from float tensor
            weights = torch.where(mask, decision_weight, 1.0)
            loss = (per_sample_loss * weights).mean()
            val_loss += loss.item()

            # Predictions and labels for unweighted metrics
            pred = (
                (torch.sigmoid(outputs) > mid_point)
                .float()
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
            label_batch = (
                is_win.detach().cpu().numpy().flatten()
            )  # Fixed: use is_win, flatten for 1D
            preds.extend(pred)
            labels_list.extend(label_batch)

    avg_val_loss = val_loss / len(loader)
    acc: float = accuracy_score(labels_list, preds)
    prec: float = precision_score(labels_list, preds, zero_division=0)
    rec: float = recall_score(labels_list, preds, zero_division=0)
    f1: float = f1_score(labels_list, preds, zero_division=0)
    return avg_val_loss, acc, prec, rec, f1


def train_model(
    model: WinPredictorWithPositionalAttention,
    training_arguments: TrainingArguments,
) -> WinPredictorWithPositionalAttention:
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_arguments.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=2,
        factor=0.1,
    )
    mid_point = 0.5
    decision_weight = 3.0

    train_loader = get_data_loader(
        training_arguments.train_dataset, training_arguments.batch_size, True
    )
    val_loader = get_data_loader(
        training_arguments.train_dataset, training_arguments.batch_size, False
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(training_arguments.epochs):
        model.train()
        train_loss = 0
        train_preds: list[torch.Tensor] = []
        train_labels: list[torch.Tensor] = []

        for batch_data in train_loader:
            team_picks, opp_picks, actual_pick, is_win, is_my_decision = [
                t.to(device) for t in batch_data
            ]
            optimizer.zero_grad()
            outputs = model(team_picks, opp_picks, actual_pick)

            per_sample_loss = criterion(outputs, is_win)
            weights = torch.where(
                (is_my_decision == 1.0), decision_weight, 1.0
            )
            loss = (per_sample_loss * weights).mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = (
                (torch.sigmoid(outputs) > mid_point)
                .float()
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
            train_labels.extend(is_win.detach().cpu().numpy().flatten())
            train_preds.extend(preds)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_prec = precision_score(
            train_labels, train_preds, zero_division=0
        )
        train_rec = recall_score(train_labels, train_preds, zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)

        logger.info(
            f"Epoch {epoch + 1}/{training_arguments.epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Acc: {train_acc:.4f}, "
            f"Prec: {train_prec:.4f}, "
            f"Rec: {train_rec:.4f}, "
            f"F1: {train_f1:.4f}",
        )

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
            model,
            val_loader,
            criterion,
            decision_weight,
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
            if patience_counter >= training_arguments.patience:
                logger.info("Early stopping triggered.")
                break

    if best_weights is None:
        raise RuntimeError("Variable best_weights is None")
    model.load_state_dict(best_weights)
    return model


def compute_baseline_f1(
    y_train: "pd.Series[float]",
    y_val: "pd.Series[float]",
) -> tuple[float, float, dict[float, float], dict[float, float]]:
    # Determine majority class from training data
    counter = Counter(y_train)
    majority_class = counter.most_common(1)[0][0]

    # Predict majority class for all validation samples
    y_pred_baseline = [majority_class] * len(y_val)

    # Compute F1-score for the baseline (using pos_label=1 for win prediction)
    baseline_f1: float = f1_score(y_val, y_pred_baseline, pos_label=1)

    # Also compute class distribution for context
    train_dist = {k: v / len(y_train) for k, v in counter.items()}
    val_counter = Counter(y_val)
    val_dist = {k: v / len(y_val) for k, v in val_counter.items()}

    return baseline_f1, majority_class, train_dist, val_dist


def main(csv_file_path: str) -> None:
    """Model training entrypoint."""
    matches_dataframe = load_personal_matches(csv_file_path)

    train_dataframe, tmp_dataframe = train_test_split(
        matches_dataframe,
        test_size=0.2,
        stratify=matches_dataframe["win"],
    )
    validation_dataframe, test_dataframe = train_test_split(
        tmp_dataframe,
        test_size=0.5,
        stratify=tmp_dataframe["win"],
    )
    augmented_train_dataframe = create_augmented_dataframe(train_dataframe)
    prepared_validation_dataframe = prepare_dataframe(validation_dataframe)
    prepared_test_dataframe = prepare_dataframe(test_dataframe)

    baseline_f1, majority_class, train_dist, val_dist = compute_baseline_f1(
        augmented_train_dataframe["win"],
        prepared_test_dataframe["win"],
    )
    logger.info(
        "Baseline F1-score "
        f"(always predict majority class {majority_class}): "
        f"{baseline_f1:.4f}",
    )
    logger.info(f"Training class distribution: {train_dist}")
    logger.info(f"Validation class distribution: {val_dist}")

    train_dataset = DotaDataset(augmented_train_dataframe)
    val_dataset = DotaDataset(prepared_validation_dataframe)
    test_dataset = DotaDataset(prepared_test_dataframe)

    model = WinPredictorWithPositionalAttention(num_heroes)

    training_arguments = TrainingArguments(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trained_model = train_model(
        model,
        training_arguments,
    )

    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
        model,
        get_data_loader(test_dataset, training_arguments.batch_size, False),  # noqa: FBT003
        nn.BCEWithLogitsLoss(),
        3,
    )
    logger.info(
        f"Test Loss: {val_loss:.4f}, "
        f"Acc: {val_acc:.4f}, "
        f"Prec: {val_prec:.4f}, "
        f"Rec: {val_rec:.4f}, "
        f"F1: {val_f1:.4f}",
    )

    torch.save(
        trained_model.state_dict(),
        settings.MODELS_FOLDER_PATH / Path("trained_model.pth"),
    )
    logger.info(
        "Training complete! Model saved as 'trained_model.pth'.",
    )
