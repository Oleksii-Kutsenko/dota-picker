import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from .neural_network import RNNWinPredictor

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MID_POINT = 0.5


def count_trainable_params(model: RNNWinPredictor) -> int:
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


TrainingExample = tuple[
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
    ) -> None:
        self.draft_sequences = torch.tensor(
            np.stack(dataframe["draft_sequence"].values),  # type: ignore[call-overload]
            dtype=torch.long,
            device=device,
        )
        self.hero_features = torch.tensor(
            np.stack(dataframe["hero_features"].values),  # type: ignore[call-overload]
            dtype=torch.float,
            device=device,
        )
        self.wins = torch.tensor(
            dataframe["win"].values,
            dtype=torch.float,
            device=device,
        )
        self.is_my_decisions = torch.tensor(
            dataframe["is_my_decision"].values,
            dtype=torch.long,
            device=device,
        )

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.draft_sequences)

    def __getitem__(self, idx: int) -> TrainingExample:
        """Return training example."""
        return (
            self.draft_sequences[idx],
            self.hero_features[idx],
            self.wins[idx],
            self.is_my_decisions[idx],
        )


class ShuffleEnum(Enum):
    """Shuffle Options Enum."""

    SHUFFLED = True
    UNSHUFFLED = False


def get_data_loader(
    dataset: Dataset[TrainingExample],
    batch_size: int,
    shuffle: ShuffleEnum = ShuffleEnum.SHUFFLED,
) -> DataLoader[TrainingExample]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle.value,
    )


@dataclass
class MetricsResult:
    """Training metrics."""

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    mcc: float
    confusion_matrix: np.ndarray | None = None

    def to_dict(self) -> dict[str, float | np.floating[Any]]:
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
            f"F1: {self.f1:.4f}, "
            f"AUC: {self.auc:.4f}, "
            f"MCC: {self.mcc:.4f}"
        )


def process_evaluation_batch(
    model: RNNWinPredictor,
    batch_data: TrainingExample,
    criterion: nn.BCEWithLogitsLoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    draft_sequence, hero_features, is_win, _ = batch_data

    outputs = model(draft_sequence, hero_features)
    per_sample_loss = criterion(outputs, is_win)
    loss = per_sample_loss.mean()

    return loss.detach(), outputs, is_win


class EarlyStopping:
    """Simple early stopping."""

    def __init__(self, patience: int = 5, delta: float = 0) -> None:
        self.patience = patience
        self.delta = delta
        self.best_val_loss: float | None = None
        self.best_metrics: MetricsResult | None = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state: dict[str, Any] | None = None

    def __call__(
        self,
        val_loss: float,
        metrics: MetricsResult,
        model: RNNWinPredictor,
    ) -> None:
        score = -val_loss

        if self.best_val_loss is None:
            self.best_val_loss = score
            self.best_metrics = metrics
            self.best_model_state = model.state_dict()
        elif score < self.best_val_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_loss = score
            self.best_metrics = metrics
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model: RNNWinPredictor) -> None:
        if self.best_model_state is None:
            msg = "Unexpected state"
            raise RuntimeError(msg)
        model.load_state_dict(self.best_model_state)


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


def process_training_batch(
    model: RNNWinPredictor,
    batch_data: TrainingExample,
    training_components: TrainingComponents,
    decision_weight: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process a single batch: forward pass, loss computation,
    and optimization step.
    """
    draft_sequence, hero_features, is_win, is_my_decision = batch_data

    training_components.optimizer.zero_grad(set_to_none=True)
    outputs = model(draft_sequence, hero_features)

    per_sample_loss = training_components.criterion(outputs, is_win)
    weights = torch.where(
        (is_my_decision == 1.0),
        decision_weight,
        1.0,
    )
    loss = (per_sample_loss * weights).mean()

    loss.backward()
    training_components.optimizer.step()

    return loss.detach(), outputs, is_win


def evaluate_model(
    model: RNNWinPredictor,
    loader: DataLoader[TrainingExample],
    criterion: nn.BCEWithLogitsLoss,
) -> tuple[MetricsResult, np.ndarray]:
    model.eval()

    all_losses: list[torch.Tensor] = []
    all_probs_tensors: list[torch.Tensor] = []
    all_labels_tensors: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_data in loader:
            batch_loss, outputs, is_win = process_evaluation_batch(
                model,
                batch_data,
                criterion,
            )

            all_losses.append(batch_loss)

            probs = torch.sigmoid(outputs)
            all_probs_tensors.append(probs)
            all_labels_tensors.append(is_win)

    avg_loss = torch.stack(all_losses).mean().item()

    all_probs = torch.cat(all_probs_tensors).cpu().numpy().flatten()

    metrics = calculate_metrics(
        y_true=torch.cat(all_labels_tensors).cpu().numpy().flatten(),
        y_pred=(all_probs > MID_POINT).astype(float),
        y_proba=all_probs,
        loss=avg_loss,
    )
    return metrics, all_probs


def train_step(
    model: RNNWinPredictor,
    train_loader: DataLoader[TrainingExample],
    training_components: TrainingComponents,
    decision_weight: int,
) -> tuple[MetricsResult, np.ndarray]:
    model.train()

    all_losses: list[torch.Tensor] = []
    all_probs_tensors: list[torch.Tensor] = []
    all_labels_tensors: list[torch.Tensor] = []

    for batch_data in train_loader:
        batch_loss, outputs, is_win = process_training_batch(
            model,
            batch_data,
            training_components,
            decision_weight,
        )

        all_losses.append(batch_loss)

        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            all_probs_tensors.append(probs)
            all_labels_tensors.append(is_win)

    avg_loss = torch.stack(all_losses).mean().item()

    all_probs = torch.cat(all_probs_tensors).cpu().numpy().flatten()

    metrics = calculate_metrics(
        y_true=torch.cat(all_labels_tensors).cpu().numpy().flatten(),
        y_pred=(all_probs > MID_POINT).astype(float),
        y_proba=all_probs,
        loss=avg_loss,
    )
    return metrics, all_probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    loss: float,
) -> MetricsResult:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0, average="macro")
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(
        y_true,
        y_pred,
    )
    mcc = matthews_corrcoef(y_true, y_pred)

    return MetricsResult(
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        confusion_matrix=cm,
        mcc=mcc,
    )


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

    early_stopping_patience: int
    scheduler_parameters: SchedulerParameters
    optimizer_parameters: OptimizerParameters
    batch_size: int
    decision_weight: int
    data: TrainingData
    pos_weight: torch.Tensor | None = None
    epochs: int = 75


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
    baseline_f1: float = f1_score(
        y_val,
        y_pred_baseline,
        average="macro",
    )

    # Also compute class distribution for context
    train_dist = {k: round(v / len(y_train), 4) for k, v in counter.items()}
    val_counter = Counter(y_val)
    val_dist = {k: round(v / len(y_val), 4) for k, v in val_counter.items()}

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
