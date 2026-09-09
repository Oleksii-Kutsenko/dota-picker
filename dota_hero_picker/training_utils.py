import copy
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    f1_score,
)
from torch import nn, optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
)

from dota_hero_picker.data_preparation import SLOT_COLUMNS

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MID_POINT = 0.5


def count_trainable_params(model: nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_metrics_collection() -> MetricCollection:
    return MetricCollection(
        {
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
            "auc": BinaryAUROC(),
            "mcc": BinaryMatthewsCorrCoef(),
            "ece": BinaryCalibrationError(n_bins=10, norm="l1"),
            "confusion_matrix": BinaryConfusionMatrix(),
        },
    ).to(device)


TrainingExample = tuple[
    torch.Tensor,  # draft_sequence
    torch.Tensor,  # patch_id
    torch.Tensor,  # win
    torch.Tensor,  # is_my_decision
]


class DotaDataset(Dataset[TrainingExample]):
    """Stores Dota 2 matches."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        self.draft_sequences = torch.tensor(
            dataframe[SLOT_COLUMNS].fillna(0).to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self.wins = torch.tensor(
            dataframe["win"].values,
            dtype=torch.float,
            device=device,
        )
        self.patch_ids = torch.tensor(
            dataframe["patch_id"].values,
            dtype=torch.long,
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

    def __getitem__(self, index: int) -> TrainingExample:
        return (
            self.draft_sequences[index],
            self.patch_ids[index],
            self.wins[index],
            self.is_my_decisions[index],
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
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    mcc: float
    ece: float = 0.0
    confusion_matrix: np.ndarray | None = None

    def __str__(self) -> str:
        return (
            f"Loss: {self.loss:.4f}, Acc: {self.accuracy:.4f}, "
            f"Prec: {self.precision:.4f}, Rec: {self.recall:.4f}, "
            f"F1: {self.f1:.4f}, AUC: {self.auc:.4f}, "
            f"MCC: {self.mcc:.4f}, ECE: {self.ece:.4f}"
        )


def process_evaluation_batch(
    model: nn.Module,
    batch_data: TrainingExample,
    criterion: nn.BCEWithLogitsLoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    draft_sequence, patch_id, is_win, *_ = batch_data

    outputs = model(draft_sequence, patch_id)
    per_sample_loss = criterion(outputs, is_win)
    loss = per_sample_loss.mean()

    return loss.detach(), outputs, is_win


class EarlyStoppingMode(Enum):
    """Early stopping monitoring mode."""

    MIN = "min"
    MAX = "max"


class EarlyStopping:
    """Simple early stopping."""

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        mode: EarlyStoppingMode = EarlyStoppingMode.MAX,
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score: float | None = None
        self.best_metrics: MetricsResult | None = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state: dict[str, Any] | None = None

    def _is_improvement(self, score: float) -> bool:
        assert self.best_score is not None
        if self.mode == EarlyStoppingMode.MAX:
            return score > self.best_score + self.delta
        return score < self.best_score - self.delta

    def __call__(
        self,
        score: float,
        metrics: MetricsResult,
        model: nn.Module,
    ) -> None:
        if self.best_score is None or self._is_improvement(score):
            self.best_score = score
            self.best_metrics = metrics
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model: nn.Module) -> None:
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
    scaler: GradScaler
    callbacks: list[Callable[[None], None]] | None = None

    def __post_init__(self) -> None:
        if self.callbacks is None:
            self.callbacks = []


def process_training_batch(
    model: nn.Module,
    batch_data: TrainingExample,
    training_components: TrainingComponents,
    decision_weight: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        draft_sequence,
        patch_id,
        is_win,
        is_my_decision,
    ) = batch_data

    training_components.optimizer.zero_grad(set_to_none=True)

    outputs = model(draft_sequence, patch_id)

    per_sample_loss = training_components.criterion(outputs, is_win)

    decision_weights = torch.where(
        (is_my_decision == 1),
        float(decision_weight),
        1.0,
    )
    loss = (per_sample_loss * decision_weights).sum() / decision_weights.sum()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    training_components.optimizer.step()

    return loss.detach(), outputs.detach(), is_win


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[TrainingExample],
    criterion: nn.BCEWithLogitsLoss,
    temperature: float,
) -> tuple[MetricsResult, np.ndarray]:
    model.eval()

    total_loss = torch.tensor(0.0, device=device)

    metrics_collection = get_metrics_collection()
    total_samples = 0
    all_probs_tensors: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_data in loader:
            (
                draft_sequence,
                patch_id,
                is_win,
                _,
            ) = batch_data

            outputs = model(draft_sequence, patch_id)

            scaled_outputs = outputs.float() / temperature
            per_sample_loss = criterion(scaled_outputs, is_win)

            total_loss += per_sample_loss.sum()
            total_samples += is_win.numel()

            probs = torch.sigmoid(scaled_outputs)
            all_probs_tensors.append(probs)
            metrics_collection.update(probs, is_win)

    avg_loss = (total_loss / total_samples).item()
    all_probs = torch.cat(all_probs_tensors).cpu().numpy().flatten()
    results = metrics_collection.compute()

    metrics = MetricsResult(
        loss=avg_loss,
        accuracy=results["accuracy"].item(),
        precision=results["precision"].item(),
        recall=results["recall"].item(),
        f1=results["f1"].item(),
        auc=results["auc"].item(),
        mcc=results["mcc"].item(),
        ece=results["ece"].item(),
        confusion_matrix=results["confusion_matrix"].cpu().numpy(),
    )
    return metrics, all_probs


def train_step(
    model: nn.Module,
    train_loader: DataLoader[TrainingExample],
    training_components: TrainingComponents,
    decision_weight: int,
) -> tuple[MetricsResult, np.ndarray]:
    model.train()

    all_losses: list[torch.Tensor] = []
    all_probs_tensors: list[torch.Tensor] = []
    metrics_collection = get_metrics_collection()

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
            metrics_collection.update(probs, is_win)

    all_probs = torch.cat(all_probs_tensors).cpu().numpy().flatten()

    results = metrics_collection.compute()

    metrics = MetricsResult(
        loss=torch.stack(all_losses).mean().item(),
        accuracy=results["accuracy"].item(),
        precision=results["precision"].item(),
        recall=results["recall"].item(),
        f1=results["f1"].item(),
        auc=results["auc"].item(),
        mcc=results["mcc"].item(),
        confusion_matrix=results["confusion_matrix"].cpu().numpy(),
    )

    return metrics, all_probs


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


def collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader[TrainingExample],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect raw logits and true labels from a data loader."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_data in loader:
            draft_sequence, patch_id, is_win, *_ = batch_data

            outputs = model(draft_sequence, patch_id)
            all_logits.append(outputs)
            all_labels.append(is_win)

    logits = torch.cat(all_logits).cpu().numpy().flatten()
    labels = torch.cat(all_labels).cpu().numpy().flatten()
    return logits, labels


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Fit Platt scaling temperature to minimize NLL on validation logits."""

    def nll(temperature: float) -> float:
        scaled = logits / temperature
        probs = 1.0 / (1.0 + np.exp(-scaled))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return float(
            -np.mean(
                labels * np.log(probs) + (1 - labels) * np.log(1 - probs),
            ),
        )

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    logger.info(f"Fitted temperature: {result.x:.4f}")
    return float(result.x)
