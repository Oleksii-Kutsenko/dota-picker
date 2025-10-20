import ast
import logging
import random
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model() -> WinPredictorWithPositionalAttention:
    return WinPredictorWithPositionalAttention(
        num_heroes,
        embedding_dim=256,
        num_heads=16,
        dropout_rate=0.55,
        hidden_sizes=(
            4096,
            32,
            8,
        ),
    )


def load_personal_matches(csv_file_path: str) -> pd.DataFrame:
    if not Path(csv_file_path).exists():
        msg = f"CSV file not found: {csv_file_path}"
        raise FileNotFoundError(msg)

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
            ast.literal_eval,
        )
        matches_dataframe[column] = matches_dataframe[column].apply(
            lambda hero_list: [
                api_id_2_model_id[api_id] for api_id in hero_list
            ],
        )
    matches_dataframe["picked_hero"] = matches_dataframe["picked_hero"].apply(
        lambda api_id: api_id_2_model_id[api_id],
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
    """Stores dota 2 matches."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_picks_per_team: int = 5,
    ) -> None:
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


class ShuffleEnum(Enum):
    """Shuffle Options Enum."""

    SHUFFLED = True
    UNSHUFFLED = False


def get_data_loader(
    dataset: Dataset[TrainingExample],
    batch_size: int,
    shuffle: ShuffleEnum = ShuffleEnum.SHUFFLED,
) -> DataLoader[DotaDataset]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle.value)


@dataclass
class MetricsResult:
    """Training metrics."""

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None = None
    confusion_matrix: np.ndarray | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc if self.auc is not None else 0.0,
        }

    def __str__(self) -> str:
        return (
            f"Loss: {self.loss:.4f}, Acc: {self.accuracy:.4f}, "
            f"Prec: {self.precision:.4f}, Rec: {self.recall:.4f}, "
            f"F1: {self.f1:.4f}"
            + (f", AUC: {self.auc:.4f}" if self.auc else "")
        )


def process_evaluation_batch(
    model,
    batch_data,
    decision_weight,
    criterion,
):
    team_picks, opp_picks, actual_pick, is_win, is_my_decision = [
        t.to(device) for t in batch_data
    ]
    outputs = model(team_picks, opp_picks, actual_pick)
    per_sample_loss = criterion(outputs, is_win)

    mask = is_my_decision == 1.0
    weights = torch.where(mask, decision_weight, 1.0)
    loss = (per_sample_loss * weights).mean()

    return loss.item(), outputs, is_win


def process_training_batch(
    model: WinPredictorWithPositionalAttention,
    batch_data,
    training_components,
    decision_weight,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Process a single batch: forward pass, loss computation,
    and optimization step.
    """
    team_picks, opp_picks, actual_pick, is_win, is_my_decision = [
        t.to(device) for t in batch_data
    ]
    training_components.optimizer.zero_grad()
    outputs = model(team_picks, opp_picks, actual_pick)

    per_sample_loss = training_components.criterion(outputs, is_win)
    weights = torch.where(
        (is_my_decision == 1.0),
        decision_weight,
        1.0,
    )
    loss = (per_sample_loss * weights).mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    training_components.optimizer.step()
    return loss.item(), outputs, is_win


def evaluate_model(
    model: WinPredictorWithPositionalAttention,
    loader: DataLoader[DotaDataset],
    criterion: nn.BCEWithLogitsLoss,
    decision_weight: float,
) -> tuple[MetricsResult, np.ndarray]:
    model.eval()

    all_losses = []
    all_preds = []
    all_probs = []
    all_labels = []
    mid_point = 0.5

    with torch.no_grad():
        for batch_data in loader:
            batch_loss, outputs, is_win = process_evaluation_batch(
                model,
                batch_data,
                decision_weight,
                criterion,
            )

            all_losses.append(batch_loss)

            probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
            preds = (probs > mid_point).astype(float)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(is_win.cpu().numpy().flatten())

    avg_loss = np.mean(all_losses)

    metrics = calculate_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        y_proba=np.array(all_probs),
        loss=avg_loss,
    )
    return metrics, np.array(all_probs)


def train_step(
    model: WinPredictorWithPositionalAttention,
    train_loader,
    training_components,
    decision_weight,
) -> tuple[float, list[torch.Tensor], list[torch.Tensor]]:
    model.train()

    all_losses = []
    all_preds = []
    all_probs = []
    all_labels = []
    mid_point = 0.5

    for batch_data in train_loader:
        batch_loss, outputs, is_win = process_training_batch(
            model,
            batch_data,
            training_components,
            decision_weight,
        )

        all_losses.append(batch_loss)

        probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        preds = (probs > mid_point).astype(float)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(is_win.cpu().numpy().flatten())

    avg_train_loss = np.mean(all_losses)

    metrics = calculate_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        y_proba=np.array(all_probs),
        loss=avg_train_loss,
    )
    return metrics, np.array(all_probs)


def calculate_metrics(
    y_true,
    y_pred,
    y_proba,
    loss,
) -> MetricsResult:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    return MetricsResult(
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        confusion_matrix=cm,
    )


class EarlyStopping:
    """Simple early stopping."""

    def __init__(self, patience: int = 5, delta: float = 0) -> None:
        self.patience = patience
        self.delta = delta
        self.best_val_loss: float | None = None
        self.best_f1_score: float | None = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state: dict | None = None

    def __call__(
        self,
        val_loss: float,
        val_f1_score: float,
        model: WinPredictorWithPositionalAttention,
    ) -> None:
        score = -val_loss

        if self.best_val_loss is None:
            self.best_val_loss = score
            self.best_f1_score = val_f1_score
            self.best_model_state = model.state_dict()
        elif score < self.best_val_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_loss = score
            self.best_f1_score = val_f1_score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model) -> None:
        model.load_state_dict(self.best_model_state)


@dataclass
class TrainingData:
    """Data for training."""

    train_dataset: DotaDataset
    val_dataset: DotaDataset


@dataclass
class SchedulerParameters:
    """Scheduler Parameters."""

    factor: float
    threshold: float
    scheduler_patience: int


@dataclass
class OptimizerParameters:
    """Optimizer Parameters."""

    lr: float
    weight_decay: float


@dataclass
class TrainingArguments:
    """Stores data for training."""

    data: TrainingData
    early_stopping_patience: int
    scheduler_parameters: SchedulerParameters
    optimizer_parameters: OptimizerParameters
    batch_size: int
    decision_weight: float
    epochs: int
    pos_weight: torch.Tensor | None = None


@dataclass
class TrainingComponents:
    """Components for training."""

    criterion: nn.BCEWithLogitsLoss
    optimizer: optim.Adam
    scheduler: optim.lr_scheduler.ReduceLROnPlateau
    early_stopping: EarlyStopping
    callbacks: list[Callable[[None], None]] | None = None

    def __post_init__(self) -> None:
        if self.callbacks is None:
            self.callbacks = []


def train_epoch(
    epoch: int,
    model: WinPredictorWithPositionalAttention,
    training_arguments: TrainingArguments,
    training_components: TrainingComponents,
) -> None:
    logger.info(f"Epoch {epoch + 1}/{training_arguments.epochs}")
    train_loader = get_data_loader(
        training_arguments.data.train_dataset,
        training_arguments.batch_size,
        ShuffleEnum.SHUFFLED,
    )
    val_loader = get_data_loader(
        training_arguments.data.val_dataset,
        training_arguments.batch_size,
        ShuffleEnum.UNSHUFFLED,
    )

    metrics, _ = train_step(
        model,
        train_loader,
        training_components,
        training_arguments.decision_weight,
    )
    logger.info(metrics)

    val_metrics, _ = evaluate_model(
        model,
        val_loader,
        training_components.criterion,
        1,
    )
    logger.info(val_metrics)
    training_components.scheduler.step(val_metrics.loss)

    training_components.early_stopping(val_metrics.loss, val_metrics.f1, model)


def train_model(
    model: WinPredictorWithPositionalAttention,
    training_arguments: TrainingArguments,
) -> tuple[WinPredictorWithPositionalAttention, float, float]:
    model.to(device)

    if training_arguments.pos_weight:
        criterion = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=training_arguments.pos_weight,
        )
    else:
        criterion = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_arguments.optimizer_parameters.lr,
        weight_decay=training_arguments.optimizer_parameters.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=training_arguments.scheduler_parameters.factor,
        threshold=training_arguments.scheduler_parameters.threshold,
        patience=training_arguments.scheduler_parameters.scheduler_patience,
    )

    training_components = TrainingComponents(
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=EarlyStopping(
            patience=training_arguments.early_stopping_patience,
        ),
    )
    for epoch in range(training_arguments.epochs):
        train_epoch(
            epoch,
            model,
            training_arguments,
            training_components,
        )
        if training_components.early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    if (
        training_components.early_stopping.best_val_loss is None
        or training_components.early_stopping.best_f1_score is None
        or training_components.early_stopping.best_model_state is None
    ):
        msg = "Unexpected state"
        raise RuntimeError(msg)
    model.load_state_dict(training_components.early_stopping.best_model_state)
    return (
        model,
        -training_components.early_stopping.best_val_loss,
        training_components.early_stopping.best_f1_score,
    )


def compute_baseline_f1(
    y_train: "pd.Series[float]",
    y_val: "pd.Series[float]",
) -> None:
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

    logger.info(
        "Baseline F1-score "
        f"(always predict majority class {majority_class}): "
        f"{baseline_f1:.4f}",
    )
    logger.info(f"Training class distribution: {train_dist}")
    logger.info(f"Validation class distribution: {val_dist}")


def compute_pos_weight(decision_dataframe: pd.DataFrame) -> torch.Tensor:
    num_of_positives = len(decision_dataframe[decision_dataframe["win"] == 1])
    num_of_negatives = len(decision_dataframe) - num_of_positives
    pos_weight = torch.tensor([num_of_negatives / num_of_positives]).to(device)
    logger.info("Class distribution for original data")
    logger.info(f"Number of Wins {num_of_positives}")
    logger.info(f"Number of Loses {num_of_negatives}")
    logger.info(f"Pos Weight {pos_weight}")
    return pos_weight


class ModelTrainer:
    """Class responsible for model training."""

    def __init__(self, csv_file_path: str) -> None:
        self.csv_file_path = csv_file_path

    def prepare_datasets(
        self,
    ) -> tuple[DotaDataset, DotaDataset, DotaDataset, torch.Tensor]:
        matches_dataframe = load_personal_matches(self.csv_file_path)
        pos_weight = compute_pos_weight(matches_dataframe)

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

        compute_baseline_f1(
            augmented_train_dataframe["win"],
            prepared_test_dataframe["win"],
        )

        train_dataset = DotaDataset(augmented_train_dataframe)
        val_dataset = DotaDataset(prepared_validation_dataframe)
        test_dataset = DotaDataset(prepared_test_dataframe)
        return train_dataset, val_dataset, test_dataset, pos_weight

    def main(self) -> None:
        """Model training entrypoint."""
        train_dataset, val_dataset, test_dataset, pos_weight = (
            self.prepare_datasets()
        )

        training_arguments = TrainingArguments(
            data=TrainingData(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            ),
            pos_weight=None,
            early_stopping_patience=13,
            epochs=46,
            optimizer_parameters=OptimizerParameters(
                lr=0.0004994948947163486,
                weight_decay=0.000188288505786014,
            ),
            scheduler_parameters=SchedulerParameters(
                factor=0.15,
                threshold=0.0022,
                scheduler_patience=13,
            ),
            decision_weight=5,
            batch_size=128,
        )

        model = create_model()
        trained_model, _, _ = train_model(
            model,
            training_arguments,
        )

        if training_arguments.pos_weight:
            criterion = nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=training_arguments.pos_weight,
            )
        else:
            criterion = nn.BCEWithLogitsLoss(reduction="none")

        metrics, _ = evaluate_model(
            trained_model,
            get_data_loader(
                test_dataset,
                training_arguments.batch_size,
                ShuffleEnum.UNSHUFFLED,
            ),
            criterion,
            1,
        )

        logger.info("Test Metrics")
        logger.info(metrics)

        logger.info("--- Confusion Matrix ---")
        header = f"{'':<12}" + "Pred: Loss    " + "Pred: Win     "
        logger.info(header)
        row1 = (
            f"Actual: Loss  {metrics.confusion_matrix[0, 0]:<12}"  # type: ignore[index]
            f"{metrics.confusion_matrix[0, 1]:<13}"  # type: ignore[index]
        )
        row2 = (
            f"Actual: Win   {metrics.confusion_matrix[1, 0]:<12}"  # type: ignore[index]
            f"{metrics.confusion_matrix[1, 1]:<13}"  # type: ignore[index]
        )
        logger.info(row1)
        logger.info(row2)
        logger.info("------------------------")

        torch.save(
            trained_model.state_dict(),
            settings.MODELS_FOLDER_PATH / Path("trained_model.pth"),
        )
        logger.info(
            "Training complete! Model saved as 'trained_model.pth'.",
        )
