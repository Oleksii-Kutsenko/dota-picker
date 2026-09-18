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


def create_objective(
    model_trainer: ModelTrainer,
) -> Callable[[Trial], tuple[float, int]]:
    def objective(trial: Trial) -> tuple[float, int]:
        d_model = trial.suggest_categorical(
            "d_model",
            [
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
        num_layers = trial.suggest_int("num_layers", 1, 6)
        num_heads = trial.suggest_categorical(
            "num_heads",
            [
                1,
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
        hidden_dim = trial.suggest_categorical(
            "hidden_dim", [32, 64, 128, 256, 512, 1024, 2048]
        )
        activation = trial.suggest_categorical(
            "activation",
            [activation.value for activation in ActivationEnum],
        )
        dropout_rate = trial.suggest_float(
            "dropout_rate",
            0.10,
            0.55,
        )

        patch_embedding_dim = trial.suggest_categorical(
            "patch_embedding_dim",
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
        stat_projection_activation = trial.suggest_categorical(
            "stat_projection_activation",
            [activation.value for activation in ActivationEnum],
        )

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
            ],
        )
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        weight_decay = trial.suggest_float(
            "weight_decay",
            1e-7,
            1e-2,
            log=True,
        )
        decision_weight = trial.suggest_int("decision_weight", 13, 24)

        scheduler_patience = trial.suggest_int(
            "scheduler_patience",
            2,
            15,
        )
        factor = trial.suggest_float(
            "factor",
            0.5,
            0.85,
        )
        threshold = trial.suggest_float(
            "threshold",
            1e-8,
            1e-2,
            log=True,
        )
        early_stopping_patience = trial.suggest_int(
            "early_stopping_patience",
            7,
            19,
        )

        training_arguments = TrainingArguments(
            data=TrainingData(
                train_dataset=model_trainer.data_manager.train_dataset,
                val_dataset=model_trainer.data_manager.val_dataset,
            ),
            early_stopping_patience=(early_stopping_patience),
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

        if d_model % num_heads != 0:
            raise optuna.TrialPruned

        model_params = ModelParameters(
            data_dimensions=DataDimensions(
                num_heroes=model_trainer.hero_data_manager.get_heroes_number(),
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
        model = MatchWinPredictor(
            model_params,
            hero_data_manager.get_hero_features_matrix(),
        )

        trainable_params = count_trainable_params(model)

        try:
            model_trainer.setup_custom_training(model, training_arguments)
            model_trainer.train_model(trial=trial)
        except torch.OutOfMemoryError:
            logger.warning(
                "CUDA OOM encountered. Pruning trial and clearing cache."
            )
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("CUDA OOM")

        assert model_trainer.training_components is not None
        assert (
            model_trainer.training_components.early_stopping.best_metrics
            is not None
        )

        val_loss = float(
            model_trainer.training_components.early_stopping.best_metrics.loss
        )
        return val_loss, trainable_params

    return objective


def main(csv_file_path: Path) -> None:
    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.enable_propagation()

    model_trainer = ModelTrainer(csv_file_path)
    objective = create_objective(model_trainer)

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
        # pruner=optuna.pruners.MedianPruner(
        #     n_startup_trials=10,
        #     n_warmup_steps=10,
        #     interval_steps=1,
        # ),
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
