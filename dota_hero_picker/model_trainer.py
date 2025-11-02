import logging
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import settings
from dota_hero_picker.hero_data_manager import HeroDataManager

from .data_manager import DataManager
from .neural_network import NNParameters, RNNWinPredictor
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

    def __init__(self, csv_file_path: Path, random_state: int = 42) -> None:
        self.random_state = random_state
        self.hero_data_manager = HeroDataManager()

        self.data_manager = DataManager(
            csv_file_path,
            self.hero_data_manager,
            random_state,
        )

        self.model: RNNWinPredictor | None = None
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
        model: RNNWinPredictor,
        training_arguments: TrainingArguments,
    ) -> None:
        self.model = model
        self.training_arguments = training_arguments

        logger.info(
            "Model trainable parameters: "
            f"{count_trainable_params(self.model)}",
        )

    def create_default_model(self) -> RNNWinPredictor:
        return RNNWinPredictor(
            NNParameters(
                num_heroes=self.hero_data_manager.get_heroes_number(),
                embedding_dim=128,
                gru_hidden_dim=8,
                num_gru_layers=1,
                dropout_rate=0.433040147239669,
                bidirectional=False,
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
            pos_weight=self.data_manager.pos_weight,
            early_stopping_patience=30,
            optimizer_parameters=OptimizerParameters(
                lr=0.0372695612898765,
                weight_decay=0.0282619707874099,
            ),
            scheduler_parameters=SchedulerParameters(
                factor=0.325895003163696,
                scheduler_patience=20,
                threshold=1.83931032095674,
            ),
            decision_weight=23,
            batch_size=16,
        )

    def train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader[TrainingExample],
        val_loader: DataLoader[TrainingExample],
    ) -> None:
        assert self.training_arguments is not None
        assert self.training_components is not None
        assert self.model is not None
        logger.info(f"Epoch {epoch + 1}/{self.training_arguments.epochs}")

        metrics, _ = train_step(
            self.model,
            train_loader,
            self.training_components,
            self.training_arguments.decision_weight,
        )
        logger.info(metrics)

        val_metrics, _ = evaluate_model(
            self.model,
            val_loader,
            self.training_components.criterion,
        )
        logger.info(val_metrics)
        self.training_components.scheduler.step(val_metrics.loss)

        self.training_components.early_stopping(
            val_metrics.loss,
            val_metrics,
            self.model,
        )

    def train_model(
        self,
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
            "min",
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
            self.train_epoch(
                epoch,
                train_loader,
                val_loader,
            )
            if self.training_components.early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break

        if (
            self.training_components.early_stopping.best_val_loss is None
            or self.training_components.early_stopping.best_metrics is None
            or self.training_components.early_stopping.best_model_state is None
        ):
            msg = "Unexpected state"
            raise RuntimeError(msg)

        self.model.load_state_dict(
            self.training_components.early_stopping.best_model_state,
        )

    def evaluate_on_test(self) -> MetricsResult:
        if self.model is None or self.training_arguments is None:
            msg = "Model must be trained before evaluation"
            raise RuntimeError(msg)

        criterion = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=self.training_arguments.pos_weight
            if self.training_arguments.pos_weight is not None
            else None,
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
        )

        return metrics

    def main(self) -> None:
        """Model training entrypoint."""
        self.setup_default_training()
        assert self.training_arguments is not None

        self.train_model()
        assert self.training_components is not None

        test_metrics = self.evaluate_on_test()

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
