import logging
from pathlib import Path

import optuna
import torch
from torch import nn, optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader

import settings
from dota_hero_picker.hero_data_manager import HeroDataManager

from .data_manager import DataManager
from .neural_network import (
    SiameseDraftPredictor,
    SiameseParameters,
)
from .patch_resolver import get_patches_number
from .training_utils import (
    EarlyStopping,
    MetricsResult,
    OptimizerParameters,
    SchedulerParameters,
    ShuffleEnum,
    TrainingArguments,
    TrainingComponents,
    TrainingData,
    TrainingExample,
    count_trainable_params,
    evaluate_model,
    get_data_loader,
    train_step,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class responsible for model training."""

    hero_data_manager = HeroDataManager()

    def __init__(self, csv_file_path: Path, random_state: int = 42) -> None:
        self.random_state = random_state

        self.data_manager = DataManager(
            csv_file_path,
            self.hero_data_manager,
            random_state,
        )

        self.model: nn.Module | None = None
        self.training_arguments: TrainingArguments | None = None
        self.training_components: TrainingComponents | None = None

    def setup_default_training(self) -> None:
        self.model = self.create_default_model()
        self.training_arguments = self.create_default_training_arguments()

        logger.info(
            "Model trainable parameters: "
            f"{count_trainable_params(self.model)}",
        )

    def setup_custom_training(
        self,
        model: nn.Module,
        training_arguments: TrainingArguments,
    ) -> None:
        self.model = model
        self.training_arguments = training_arguments

        logger.info(
            "Model trainable parameters: "
            f"{count_trainable_params(self.model)}",
        )

    @classmethod
    def create_default_model(cls) -> SiameseDraftPredictor:
        return SiameseDraftPredictor(
            SiameseParameters(
                num_heroes=cls.hero_data_manager.get_heroes_number(),
                num_patches=get_patches_number(),
                d_model=16,
                num_heads=2,
                num_synergy_layers=3,
                dropout_rate=0.363054,
                patch_embedding_dim=32,
            ),
        )

    def create_default_training_arguments(
        self,
    ) -> TrainingArguments:
        return TrainingArguments(
            data=TrainingData(
                train_dataset=self.data_manager.train_dataset,
                val_dataset=self.data_manager.val_dataset,
            ),
            early_stopping_patience=15,
            optimizer_parameters=OptimizerParameters(
                lr=0.000648,
                weight_decay=0.0,
            ),
            scheduler_parameters=SchedulerParameters(
                factor=0.785886,
                scheduler_patience=12,
                threshold=0.019316,
            ),
            decision_weight=22,
            batch_size=256,
        )

    def train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader[TrainingExample],
        val_loader: DataLoader[TrainingExample],
    ) -> tuple[MetricsResult, MetricsResult]:
        assert self.training_arguments is not None
        assert self.training_components is not None
        assert self.model is not None
        logger.info(f"Epoch {epoch + 1}/{self.training_arguments.epochs}")

        train_metrics, _ = train_step(
            self.model,
            train_loader,
            self.training_components,
            self.training_arguments.decision_weight,
        )
        logger.info(train_metrics)

        val_metrics, _ = evaluate_model(
            self.model,
            val_loader,
            self.training_components.criterion,
            1,
        )
        logger.info(val_metrics)
        self.training_components.scheduler.step(val_metrics.mcc)

        self.training_components.early_stopping(
            val_metrics.mcc,
            val_metrics,
            self.model,
        )
        return train_metrics, val_metrics

    def train_model(
        self,
        trial: optuna.trial.Trial | None = None,
    ) -> None:
        assert self.model is not None
        assert self.training_arguments is not None
        self.model.to(device)

        criterion = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=self.training_arguments.pos_weight
            if self.training_arguments.pos_weight
            else None,
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_arguments.optimizer_parameters.lr,
            weight_decay=self.training_arguments.optimizer_parameters.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "max",
            factor=self.training_arguments.scheduler_parameters.factor,
            threshold=self.training_arguments.scheduler_parameters.threshold,
            patience=self.training_arguments.scheduler_parameters.scheduler_patience,
        )

        self.training_components = TrainingComponents(
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=EarlyStopping(
                patience=self.training_arguments.early_stopping_patience,
            ),
            scaler=GradScaler(enabled=torch.cuda.is_available()),
        )

        train_loader = get_data_loader(
            self.training_arguments.data.train_dataset,
            self.training_arguments.batch_size,
            ShuffleEnum.SHUFFLED,
        )
        val_loader = get_data_loader(
            self.training_arguments.data.val_dataset,
            self.training_arguments.batch_size,
            ShuffleEnum.UNSHUFFLED,
        )

        for epoch in range(self.training_arguments.epochs):
            _, val_metrics = self.train_epoch(
                epoch,
                train_loader,
                val_loader,
            )

            if trial is not None:
                intermediate_value = float(val_metrics.mcc)
                trial.report(intermediate_value, step=epoch)

                if trial.should_prune():
                    msg = (
                        f"Pruned at epoch {epoch + 1} "
                        f"with mcc={intermediate_value:.4f}"
                    )
                    raise optuna.TrialPruned(msg)

            if self.training_components.early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break

        if (
            self.training_components.early_stopping.best_score is None
            or self.training_components.early_stopping.best_metrics is None
            or self.training_components.early_stopping.best_model_state is None
        ):
            msg = "Unexpected state"
            raise RuntimeError(msg)

        self.model.load_state_dict(
            self.training_components.early_stopping.best_model_state,
        )

    def evaluate_on_test(self, temperature: float) -> MetricsResult:
        if self.model is None or self.training_arguments is None:
            msg = "Model must be trained before evaluation"
            raise RuntimeError(msg)

        criterion = nn.BCEWithLogitsLoss(
            reduction="none",
        )

        test_loader = get_data_loader(
            self.data_manager.test_dataset,
            self.training_arguments.batch_size,
            ShuffleEnum.UNSHUFFLED,
        )

        metrics, _ = evaluate_model(
            self.model,
            test_loader,
            criterion,
            temperature,
        )

        return metrics

    def main(self) -> None:
        """Model training entrypoint."""
        self.setup_default_training()
        assert self.training_arguments is not None

        self.train_model()
        assert self.training_components is not None

        test_metrics = self.evaluate_on_test(1)

        logger.info("Test Metrics")
        logger.info(test_metrics)

        logger.info("--- Confusion Matrix ---")
        header = f"{'':<12}" + "Pred: Loss    " + "Pred: Win     "
        logger.info(header)
        row1 = (
            f"Actual: Loss  {test_metrics.confusion_matrix[0, 0]:<12}"  # type: ignore[index]
            f"{test_metrics.confusion_matrix[0, 1]:<13}"  # type: ignore[index]
        )
        row2 = (
            f"Actual: Win   {test_metrics.confusion_matrix[1, 0]:<12}"  # type: ignore[index]
            f"{test_metrics.confusion_matrix[1, 1]:<13}"  # type: ignore[index]
        )
        logger.info(row1)
        logger.info(row2)
        logger.info("------------------------")

        torch.save(
            self.training_components.early_stopping.best_model_state,
            settings.MODELS_FOLDER_PATH / Path("trained_model.pth"),
        )
        logger.info(
            "Training complete! Model saved as 'trained_model.pth'.",
        )
