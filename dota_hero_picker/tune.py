import gc
import logging
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

hero_data_manager = HeroDataManager()


class CudaOOMTrialPruned(optuna.TrialPruned):
    def __init__(
        self,
    ) -> None:
        super().__init__("CUDA OOM")


def sample_model_parameters(
    trial: Trial,
    data_manager: DataManager,
) -> ModelParameters:
    d_model = trial.suggest_categorical(
        "d_model",
        [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
        ],
    )
    num_layers = trial.suggest_int("num_layers", 1, 7)
    num_heads = trial.suggest_categorical(
        "num_heads",
        [
            1,
            2,
            4,
            8,
            16,
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
    activation = trial.suggest_categorical(
        "activation",
        [activation.value for activation in ActivationEnum],
    )
    dropout_rate = trial.suggest_float(
        "dropout_rate",
        0.05,
        0.65,
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
        ],
    )
    lr = trial.suggest_float("lr", 1e-8, 1e-2, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-9,
        1e-1,
        log=True,
    )
    decision_weight = trial.suggest_int("decision_weight", 12, 24)

    scheduler_patience = trial.suggest_int(
        "scheduler_patience",
        2,
        16,
    )
    factor = trial.suggest_float(
        "factor",
        0.5,
        1,
    )
    threshold = trial.suggest_float(
        "threshold",
        1e-8,
        1e-1,
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
) -> Callable[[Trial], tuple[float, int, float]]:
    def objective(trial: Trial) -> tuple[float, int, float]:
        model_params = sample_model_parameters(trial, data_manager)
        training_arguments = sample_training_arguments(trial, data_manager)

        if model_params.d_model % model_params.num_heads != 0:
            raise optuna.TrialPruned

        model = MatchWinPredictor(
            model_params,
            hero_data_manager.get_hero_features_matrix(),
        )
        trainable_params = count_trainable_params(model)

        try:
            trainer = ModelTrainer(model, training_arguments, data_manager)
            early_stopping = trainer.train()
        except torch.OutOfMemoryError as oom_error:
            logger.warning(
                "CUDA OOM encountered. Pruning trial and clearing cache.",
            )
            gc.collect()
            torch.cuda.empty_cache()
            raise CudaOOMTrialPruned from oom_error

        val_loss = float(early_stopping.best_metrics.loss)
        val_auc = float(early_stopping.best_metrics.auc)
        return val_loss, trainable_params, val_auc

    return objective


def main(csv_file_path: Path) -> None:
    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.enable_propagation()

    data_manager = DataManager(csv_file_path, hero_data_manager)
    objective = create_objective(data_manager)

    current_date = pd.Timestamp.now().strftime("%Y%m%d")
    study_name = (
        f"dota_win_predictor_loss_{current_date}_{uuid.uuid4().hex[:8]}"
    )
    logger.info(f"Study name: {study_name}")

    study = optuna.create_study(
        study_name=study_name,
        directions=["minimize", "minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
        ),
        storage=settings.OPTUNA_STORAGE,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=340,
        show_progress_bar=True,
    )
    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_trials.csv", index=False)
