import logging

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch import nn, optim

from dota_hero_picker.data_manager import DataManager
from dota_hero_picker.training_utils import (
    DotaBatchLoader,
    EarlyStopping,
    EarlyStoppingMode,
    EarlyStoppingStateError,
    MetricsResult,
    ShuffleEnum,
    TrainingArguments,
    TrainingComponents,
    count_trainable_params,
    evaluate_model,
    train_step,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class responsible for model training."""

    def __init__(
        self,
        model: nn.Module,
        training_arguments: TrainingArguments,
        data_manager: DataManager,
    ) -> None:
        self.model = model
        self.training_arguments = training_arguments
        self.data_manager = data_manager

        logger.info(
            f"Model trainable params: {count_trainable_params(self.model)}",
        )
        self.model.to(device)

        criterion = nn.BCEWithLogitsLoss(reduction="none")
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_arguments.optimizer_parameters.lr,
            weight_decay=self.training_arguments.optimizer_parameters.weight_decay,
            fused=True,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.training_arguments.scheduler_parameters.factor,
            patience=self.training_arguments.scheduler_parameters.scheduler_patience,
            threshold=self.training_arguments.scheduler_parameters.threshold,
        )
        early_stopping = EarlyStopping(
            patience=self.training_arguments.early_stopping_patience,
            mode=EarlyStoppingMode.MIN,
        )
        self.training_components = TrainingComponents(
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
        )

    def train_epoch(
        self,
        epoch: int,
        train_loader: DotaBatchLoader,
        val_loader: DotaBatchLoader,
    ) -> tuple[float, MetricsResult]:
        train_loss = train_step(
            self.model,
            train_loader,
            self.training_components,
            self.training_arguments.decision_weight,
        )

        val_metrics, _ = evaluate_model(
            self.model,
            val_loader,
            self.training_components.criterion,
            1,
        )

        current_lr = self.training_components.optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch + 1:02d}/{self.training_arguments.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics.loss:.4f} | "
            f"Val AUC: {val_metrics.auc:.4f} | "
            f"Val MCC: {val_metrics.mcc:.4f} | "
            f"LR: {current_lr:.2e}",
        )

        self.training_components.scheduler.step(val_metrics.loss)
        self.training_components.early_stopping(
            val_metrics.loss,
            val_metrics,
            self.model,
        )
        return train_loss, val_metrics

    def train(self) -> EarlyStopping:
        train_loader = DotaBatchLoader(
            self.training_arguments.data.train_dataset,
            batch_size=self.training_arguments.batch_size,
            shuffle=ShuffleEnum.SHUFFLED,
        )
        val_loader = DotaBatchLoader(
            self.training_arguments.data.val_dataset,
            batch_size=512,
            shuffle=ShuffleEnum.UNSHUFFLED,
        )

        for epoch in range(self.training_arguments.epochs):
            self.train_epoch(epoch, train_loader, val_loader)

            if self.training_components.early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break

        early_stopping = self.training_components.early_stopping
        if (
            early_stopping.best_score is None
            or early_stopping.best_metrics is None
            or early_stopping.best_model_state is None
        ):
            raise EarlyStoppingStateError

        self.model.load_state_dict(early_stopping.best_model_state)
        return early_stopping

    def evaluate_on_test(self, temperature: float) -> MetricsResult:
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        test_loader = DotaBatchLoader(
            self.data_manager.test_dataset,
            batch_size=self.training_arguments.batch_size,
            shuffle=ShuffleEnum.UNSHUFFLED,
        )
        metrics, _ = evaluate_model(
            self.model,
            test_loader,
            criterion,
            temperature,
        )
        return metrics

    def calibrate_temperature(self) -> float:
        val_loader = DotaBatchLoader(
            self.training_arguments.data.val_dataset,
            batch_size=512,
            shuffle=ShuffleEnum.UNSHUFFLED,
        )

        self.model.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for batch_data in val_loader:
                draft_sequence, patch_id, is_win, *_, picked_hero = batch_data
                outputs = self.model(draft_sequence, patch_id, picked_hero)
                all_logits.append(outputs)
                all_labels.append(is_win)

        logits = torch.cat(all_logits).cpu().numpy().flatten()
        labels = torch.cat(all_labels).cpu().numpy().flatten()

        def nll(temp: float) -> float:
            scaled = logits / temp
            probs = 1.0 / (1.0 + np.exp(-scaled))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return float(
                -np.mean(
                    labels * np.log(probs) + (1 - labels) * np.log(1 - probs),
                ),
            )

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        temperature = float(result.x)
        logger.info(f"Fitted temperature: {temperature:.4f}")
        return temperature
