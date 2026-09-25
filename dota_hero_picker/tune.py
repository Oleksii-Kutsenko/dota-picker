import gc
import logging
import math
import uuid
from collections.abc import Callable
from pathlib import Path

import optuna
import pandas as pd
import torch
from optuna import Trial

import settings
from dota_hero_picker.data_manager import DataManager
from dota_hero_picker.hero_data_manager import HeroDataManager
from dota_hero_picker.training_utils import (
    OptimizerParameters,
    SchedulerParameters,
    TrainingArguments,
    TrainingData,
    count_trainable_params,
)

from .model_trainer import ModelTrainer
from .neural_network import (
    ActivationEnum,
    DataDimensions,
    MatchWinPredictor,
    ModelParameters,
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
    "hidden_dim",
    "activation",
    "dropout_rate",
    "patch_embedding_dim",
    "stat_projection_activation",
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


def sample_model_parameters(
    trial: Trial,
    data_manager: DataManager,
) -> ModelParameters:
    d_model = trial.suggest_categorical(
        "d_model",
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
        ],
    )
    num_layers = trial.suggest_int("num_layers", 1, 9)
    num_heads = trial.suggest_categorical(
        "num_heads",
        [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
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
            64,
            128,
            256,
        ],
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
            16384,
            16384 * 2,
        ],
    )
    activation = trial.suggest_categorical(
        "activation",
        [activation.value for activation in ActivationEnum],
    )
    dropout_rate = trial.suggest_float(
        "dropout_rate",
        0.10,
        0.85,
    )
    patch_embedding_dim = trial.suggest_categorical(
        "patch_embedding_dim",
        [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
        ],
    )
    stat_projection_activation = trial.suggest_categorical(
        "stat_projection_activation",
        [
            ActivationEnum.SILU,
            ActivationEnum.RELU,
        ],
    )

    return ModelParameters(
        data_dimensions=DataDimensions(
            num_heroes=data_manager.hero_data_manager.get_heroes_number(),
            num_patches=get_patches_number(),
        ),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        d_model=d_model,
        dropout_rate=dropout_rate,
        patch_embedding_dim=patch_embedding_dim,
        stat_projection_activation=ActivationEnum(
            stat_projection_activation,
        ),
        activation=ActivationEnum(activation),
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
            8192,
            16384,
            16384 * 2,
        ],
    )
    lr = trial.suggest_float("lr", 1e-7, 1e-2, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-10,
        1e-3,
        log=True,
    )
    decision_weight = trial.suggest_int("decision_weight", 10, 23)

    scheduler_patience = trial.suggest_int(
        "scheduler_patience",
        1,
        13,
    )
    factor = trial.suggest_float(
        "factor",
        0.30,
        0.85,
    )
    threshold = trial.suggest_float(
        "threshold",
        1e-9,
        1e-4,
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
) -> Callable[[Trial], tuple[float, float]]:
    def objective(trial: Trial) -> tuple[float, float]:
        model_params = sample_model_parameters(trial, data_manager)
        training_arguments = sample_training_arguments(trial, data_manager)

        if model_params.d_model % model_params.num_heads != 0:
            raise optuna.TrialPruned

        model = MatchWinPredictor(
            model_params,
            data_manager.hero_data_manager.get_hero_features_matrix(),
        )
        log_params = float(math.log10(max(1, count_trainable_params(model))))

        try:
            trainer = ModelTrainer(model, training_arguments, data_manager)
            early_stopping = trainer.train(load_best_state=False)
        except torch.OutOfMemoryError as oom_error:
            logger.warning(
                "CUDA OOM encountered. Pruning trial and clearing cache.",
            )
            gc.collect()
            torch.cuda.empty_cache()
            raise CudaOOMTrialPruned from oom_error
        finally:
            del model

        val_loss = float(early_stopping.best_metrics.loss)
        return val_loss, log_params

    return objective


def main(csv_file_path: Path) -> None:
    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.enable_propagation()

    hero_data_manager = HeroDataManager()
    data_manager = DataManager(csv_file_path, hero_data_manager)
    objective = create_objective(data_manager)

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
        n_trials=16 * 25,
        show_progress_bar=True,
    )
