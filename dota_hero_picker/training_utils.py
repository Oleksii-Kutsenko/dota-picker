import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Self

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryMatthewsCorrCoef,
)

from dota_hero_picker.data_preparation import SLOT_COLUMNS

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EARLY_STOPPING_PATIENCE = 6


def count_trainable_params(model: nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_metrics_collection() -> MetricCollection:
    return MetricCollection(
        {
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
    torch.Tensor,  # picked hero
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
        self.picked_heroes = torch.tensor(
            dataframe["picked_hero"].values,
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
            self.picked_heroes[index],
        )


class ShuffleEnum(Enum):
    """Shuffle Options Enum."""

    SHUFFLED = True
    UNSHUFFLED = False


class DotaBatchLoader:
    def __init__(
        self,
        dataset: DotaDataset,
        batch_size: int,
        shuffle: ShuffleEnum = ShuffleEnum.SHUFFLED,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle.value
        self.num_samples = len(dataset)

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[TrainingExample]:
        device = self.dataset.draft_sequences.device
        indices = (
            torch.randperm(self.num_samples, device=device)
            if self.shuffle
            else torch.arange(self.num_samples, device=device)
        )
        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]
            yield (
                self.dataset.draft_sequences[batch_indices],
                self.dataset.patch_ids[batch_indices],
                self.dataset.wins[batch_indices],
                self.dataset.is_my_decisions[batch_indices],
                self.dataset.picked_heroes[batch_indices],
            )


@dataclass
class MetricsResult:
    loss: float
    auc: float
    mcc: float = 0.0
    ece: float = 0.0
    confusion_matrix: np.ndarray | None = None

    @classmethod
    def from_collection(
        cls,
        loss: float,
        results: dict[str, torch.Tensor],
    ) -> Self:
        return cls(
            loss=loss,
            auc=results["auc"].item(),
            mcc=results["mcc"].item(),
            ece=results["ece"].item() if "ece" in results else 0.0,
            confusion_matrix=results["confusion_matrix"].cpu().numpy(),
        )

    def __str__(self) -> str:
        return (
            f"Loss: {self.loss:.4f}, AUC: {self.auc:.4f}, "
            f"MCC: {self.mcc:.4f}, ECE: {self.ece:.4f}"
        )


def process_evaluation_batch(
    model: nn.Module,
    batch_data: TrainingExample,
    criterion: nn.BCEWithLogitsLoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    draft_sequence, patch_id, is_win, *_, picked_hero = batch_data

    outputs = model(draft_sequence, patch_id, picked_hero)
    per_sample_loss = criterion(outputs, is_win)
    loss = per_sample_loss.mean()

    return loss.detach(), outputs, is_win


class EarlyStoppingMode(Enum):
    """Early stopping monitoring mode."""

    MIN = "min"
    MAX = "max"


class EarlyStoppingStateError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "Early stopping has no recorded best state or metrics.",
        )


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        delta: float = 1e-4,
        mode: EarlyStoppingMode = EarlyStoppingMode.MIN,
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.early_stop = False
        self.counter = 0

        self._best_score: float | None = None
        self._best_metrics: MetricsResult | None = None
        self._best_model_state: dict[str, torch.Tensor] | None = None

    @property
    def best_metrics(self) -> MetricsResult:
        if self._best_metrics is None:
            raise EarlyStoppingStateError
        return self._best_metrics

    @property
    def best_model_state(self) -> dict[str, torch.Tensor]:
        if self._best_model_state is None:
            raise EarlyStoppingStateError
        return self._best_model_state

    @property
    def best_score(self) -> float:
        if self._best_score is None:
            raise EarlyStoppingStateError
        return self._best_score

    def _is_improvement(self, score: float) -> bool:
        assert self._best_score is not None
        if self.mode == EarlyStoppingMode.MAX:
            return score > self._best_score + self.delta
        return score < self._best_score - self.delta

    def __call__(
        self,
        score: float,
        metrics: MetricsResult,
        model: nn.Module,
    ) -> None:
        if self._best_score is None or self._is_improvement(score):
            self._best_score = score
            self._best_metrics = metrics
            self._best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model: nn.Module) -> None:
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
    model: nn.Module,
    batch_data: TrainingExample,
    training_components: TrainingComponents,
    decision_weight: int,
) -> torch.Tensor:
    (draft_sequence, patch_id, is_win, is_my_decision, picked_hero) = (
        batch_data
    )

    training_components.optimizer.zero_grad(set_to_none=True)

    outputs = model(draft_sequence, patch_id, picked_hero)

    per_sample_loss = training_components.criterion(outputs, is_win)

    decision_weights = torch.where(
        (is_my_decision == 1),
        float(decision_weight),
        1.0,
    )
    loss = (per_sample_loss * decision_weights).sum() / decision_weights.sum()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        foreach=True,
    )
    training_components.optimizer.step()

    return loss.detach()  # type: ignore[no-any-return]


def evaluate_model(
    model: nn.Module,
    loader: DotaBatchLoader,
    criterion: nn.BCEWithLogitsLoss,
    temperature: float,
) -> tuple[MetricsResult, np.ndarray]:
    model.eval()
    total_loss: torch.Tensor | None = None
    total_samples = 0
    metrics_collection = get_metrics_collection()
    all_probs_tensors: list[torch.Tensor] = []

    with torch.no_grad():
        for draft_seq, patch_id, is_win, _, picked_hero in loader:
            scaled = (
                model(draft_seq, patch_id, picked_hero).float() / temperature
            )
            batch_loss = criterion(scaled, is_win).sum()
            total_loss = (
                batch_loss if total_loss is None else (total_loss + batch_loss)
            )
            total_samples += is_win.numel()

            probs = torch.sigmoid(scaled)
            all_probs_tensors.append(probs)
            metrics_collection.update(probs, is_win)

    all_probs = torch.cat(all_probs_tensors).cpu().numpy().flatten()
    avg_loss = (
        (total_loss / total_samples).item() if total_loss is not None else 0.0
    )
    return MetricsResult.from_collection(
        avg_loss,
        metrics_collection.compute(),
    ), all_probs


def train_step(
    model: nn.Module,
    train_loader: DotaBatchLoader,
    training_components: TrainingComponents,
    decision_weight: int,
) -> float:
    model.train()
    total_loss: torch.Tensor | None = None
    num_batches = 0

    for batch_data in train_loader:
        batch_loss = process_training_batch(
            model,
            batch_data,
            training_components,
            decision_weight,
        )
        total_loss = (
            batch_loss if total_loss is None else (total_loss + batch_loss)
        )
        num_batches += 1

    if total_loss is None or num_batches == 0:
        return 0.0

    return (total_loss / num_batches).item()


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

    scheduler_parameters: SchedulerParameters
    optimizer_parameters: OptimizerParameters
    batch_size: int
    decision_weight: int
    data: TrainingData
    epochs: int = 75
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
