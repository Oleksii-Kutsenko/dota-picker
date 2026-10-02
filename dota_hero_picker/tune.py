import gc
import logging
import math
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna import Trial

import settings
from dota_hero_picker.data_manager import DataManager
from dota_hero_picker.hero_data_manager import HeroDataManager
from dota_hero_picker.training_utils import (
    MetricsResult,
    OptimizerParameters,
    SchedulerParameters,
    TrainingArguments,
    TrainingData,
    count_trainable_params,
)

from .model_trainer import ModelTrainer
from .neural_network import (
    ClassifierParameters,
    DataDimensions,
    HeroFeatureParameters,
    MatchWinPredictor,
    ModelParameters,
    SynergyParameters,
    TransformerParameters,
)
from .patch_resolver import get_patches_number

logger = logging.getLogger(__name__)


class CudaOOMTrialPruned(optuna.TrialPruned):
    def __init__(
        self,
    ) -> None:
        super().__init__("CUDA OOM")


PARAMETER_ORDER = [
    # Architecture
    "d_model",
    "num_layers",
    "num_heads",
    "ffn_ratio",
    "synergy_num_layers",
    "synergy_num_heads",
    "synergy_ffn_ratio",
    "hidden_dim",
    "dropout_rate",
    "hero_embed_dim",
    "stat_embed_dim",
    "patch_embed_dim",
    "num_fusion_layers",
    # Optimizer
    "batch_size",
    "lr",
    "weight_decay",
    "decision_weight",
    # Scheduler
    "scheduler_patience",
    "factor",
    "threshold",
]


class LossMedianPruner:
    def __init__(
        self,
        minimum_startup_trials: int,
        warmup_epoch_count: int,
    ) -> None:
        self.minimum_startup_trials = minimum_startup_trials
        self.warmup_epoch_count = warmup_epoch_count
        self.historical_epoch_losses: dict[int, list[float]] = {}
        self.completed_trial_count = 0
        self.current_trial_best_loss = float("inf")

    def step(
        self,
        epoch: int,
        validation_metrics: MetricsResult,
    ) -> None:
        current_validation_loss = validation_metrics.loss
        self.current_trial_best_loss = min(
            self.current_trial_best_loss,
            current_validation_loss,
        )

        recorded_losses = self.historical_epoch_losses.setdefault(
            epoch,
            [],
        )
        recorded_losses.append(self.current_trial_best_loss)

        if (
            self.completed_trial_count >= self.minimum_startup_trials
            and epoch >= self.warmup_epoch_count
            and len(recorded_losses) > 1
        ):
            reference_losses = recorded_losses[:-1]
            median_validation_loss = float(np.median(reference_losses))

            if self.current_trial_best_loss > median_validation_loss:
                raise optuna.TrialPruned(  # noqa: TRY003
                    f"Pruned at epoch {epoch + 1}: "
                    f"Best loss {self.current_trial_best_loss:.4f} > "
                    f"Median {median_validation_loss:.4f}",
                )

    def on_trial_complete(self) -> None:
        self.completed_trial_count += 1
        self.current_trial_best_loss = float("inf")


def sample_model_parameters(
    trial: Trial,
    data_manager: DataManager,
) -> ModelParameters:
    d_model = trial.suggest_categorical(
        "d_model",
        [
            8,
            16,
            32,
            64,
            128,
            256,
            512,
        ],
    )
    num_layers = trial.suggest_int("num_layers", 1, 7)
    num_heads = trial.suggest_categorical(
        "num_heads",
        [
            2,
            4,
            8,
            16,
            32,
        ],
    )
    ffn_ratio = trial.suggest_categorical(
        "ffn_ratio",
        [
            1,
            2,
            4,
            8,
            16,
            32,
        ],
    )
    synergy_num_layers = trial.suggest_int("synergy_num_layers", 2, 4)
    synergy_num_heads = trial.suggest_categorical(
        "synergy_num_heads",
        [2, 4, 8],
    )
    synergy_ffn_ratio = trial.suggest_categorical(
        "synergy_ffn_ratio",
        [2, 4, 8],
    )
    hidden_dim = trial.suggest_categorical(
        "hidden_dim",
        [
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
        ],
    )
    dropout_rate = trial.suggest_float(
        "dropout_rate",
        0.05,
        0.70,
    )
    hero_embed_dim = trial.suggest_categorical(
        "hero_embed_dim",
        [
            4,
            8,
            16,
            32,
        ],
    )
    stat_embed_dim = trial.suggest_categorical(
        "stat_embed_dim",
        [16, 32, 64],
    )
    patch_embed_dim = trial.suggest_categorical(
        "patch_embed_dim",
        [
            2,
            4,
            8,
            16,
        ],
    )
    num_fusion_layers = trial.suggest_int("num_fusion_layers", 2, 4)

    return ModelParameters(
        data_dimensions=DataDimensions(
            num_heroes=data_manager.hero_data_manager.get_heroes_number(),
            num_patches=get_patches_number(),
        ),
        hero_feature_parameters=HeroFeatureParameters(
            hero_embed_dim=hero_embed_dim,
            stat_embed_dim=stat_embed_dim,
            patch_embed_dim=patch_embed_dim,
            num_fusion_layers=num_fusion_layers,
        ),
        d_model=d_model,
        dropout_rate=dropout_rate,
        synergy_parameters=SynergyParameters(
            num_layers=synergy_num_layers,
            num_heads=synergy_num_heads,
            ffn_ratio=synergy_ffn_ratio,
        ),
        transformer_parameters=TransformerParameters(
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
        ),
        classifier_parameters=ClassifierParameters(
            hidden_dim=hidden_dim,
        ),
    )


def sample_training_arguments(
    trial: Trial,
    data_manager: DataManager,
) -> TrainingArguments:
    batch_size = trial.suggest_categorical(
        "batch_size",
        [
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
        ],
    )
    lr = trial.suggest_float("lr", 1e-7, 1e-2, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-11,
        1e-6,
        log=True,
    )
    decision_weight = trial.suggest_int(
        "decision_weight",
        10,
        23,
    )

    scheduler_patience = trial.suggest_int(
        "scheduler_patience",
        4,
        13,
    )
    factor = trial.suggest_float(
        "factor",
        0.30,
        0.80,
    )
    threshold = trial.suggest_float(
        "threshold",
        1e-11,
        1e-6,
        log=True,
    )

    return TrainingArguments(
        data=TrainingData(
            train_dataset=data_manager.train_dataset,
            val_dataset=data_manager.val_dataset,
        ),
        optimizer_parameters=OptimizerParameters(
            lr=lr,
            weight_decay=weight_decay,
        ),
        scheduler_parameters=SchedulerParameters(
            factor=factor,
            threshold=threshold,
            scheduler_patience=scheduler_patience,
        ),
        batch_size=batch_size,
        decision_weight=decision_weight,
    )


def create_objective(
    data_manager: DataManager,
    pruner: LossMedianPruner,
) -> Callable[[Trial], tuple[float, float]]:
    def objective(trial: Trial) -> tuple[float, float]:
        model_params = sample_model_parameters(trial, data_manager)
        training_arguments = sample_training_arguments(trial, data_manager)

        if (
            model_params.d_model % model_params.synergy_parameters.num_heads
            != 0
            or model_params.d_model
            % model_params.transformer_parameters.num_heads
            != 0
        ):
            raise optuna.TrialPruned

        model = MatchWinPredictor(
            model_params,
            data_manager.hero_data_manager.get_hero_features_matrix(),
        )
        log_params = round(
            (math.log10(max(1, count_trainable_params(model)))),
            2,
        )

        try:
            trainer = ModelTrainer(model, training_arguments, data_manager)
            early_stopping = trainer.train(
                load_best_state=False,
                epoch_callback=pruner.step,
            )

        except torch.OutOfMemoryError as oom_error:
            logger.warning(
                "CUDA OOM encountered. Prunsing trial and clearing cache.",
            )
            gc.collect()
            torch.cuda.empty_cache()
            raise CudaOOMTrialPruned from oom_error
        finally:
            pruner.on_trial_complete()
            del model

        val_loss = float(early_stopping.best_metrics.loss)
        return val_loss, log_params

    return objective


def main(csv_file_path: Path) -> None:
    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.enable_propagation()

    hero_data_manager = HeroDataManager()
    data_manager = DataManager(csv_file_path, hero_data_manager)
    pruner = LossMedianPruner(
        minimum_startup_trials=20,
        warmup_epoch_count=7,
    )
    objective = create_objective(data_manager, pruner)

    current_date = pd.Timestamp.now().strftime("%Y%m%d")
    study_name = (
        f"dota_win_predictor_loss_{current_date}_{uuid.uuid4().hex[:8]}"
    )
    logger.info(f"Study name: {study_name}")

    study = optuna.create_study(
        study_name=study_name,
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
        ),
        storage=settings.OPTUNA_STORAGE,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=len(PARAMETER_ORDER) * 25,
        show_progress_bar=True,
    )
