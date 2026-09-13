import logging
import uuid
from collections.abc import Callable
from pathlib import Path

import optuna
import pandas as pd
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
    MatchupParameters,
    MatchWinPredictor,
    ModelParameters,
    SynergyParameters,
)
from .patch_resolver import get_patches_number

logger = logging.getLogger(__name__)

hero_data_manager = HeroDataManager()


def create_objective(
    model_trainer: ModelTrainer,
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:

        patch_embedding_dim = trial.suggest_categorical(
            "patch_embedding_dim",
            [8, 16, 32, 64, 128, 256, 512, 1024],
        )
        stat_projection_activation = trial.suggest_categorical(
            "stat_projection_activation",
            [activation.value for activation in ActivationEnum],
        )

        activation = trial.suggest_categorical(
            "activation",
            [activation.value for activation in ActivationEnum],
        )
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
        num_heads = trial.suggest_categorical(
            "num_heads",
            [
                1,
                2,
                4,
            ],
        )
        num_synergy_layers = trial.suggest_int("num_synergy_layers", 2, 6)
        dropout_rate = trial.suggest_float(
            "dropout_rate",
            0.2,
            0.45,
        )
        ffn_ratio = trial.suggest_categorical("ffn_ratio", [2, 4, 8, 16])
        scheduler_patience = trial.suggest_int(
            "scheduler_patience",
            5,
            14,
        )
        early_stopping_patience = trial.suggest_int(
            "early_stopping_patience",
            7,
            17,
        )
        num_matchup_layers = trial.suggest_int("num_matchup_layers", 1, 3)

        training_arguments = TrainingArguments(
            data=TrainingData(
                train_dataset=model_trainer.data_manager.train_dataset,
                val_dataset=model_trainer.data_manager.val_dataset,
            ),
            early_stopping_patience=(early_stopping_patience),
            optimizer_parameters=OptimizerParameters(
                lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
                weight_decay=trial.suggest_float(
                    "weight_decay",
                    1e-4,
                    1e-1,
                    log=True,
                ),
            ),
            scheduler_parameters=SchedulerParameters(
                factor=trial.suggest_float(
                    "factor",
                    0.6,
                    0.75,
                ),
                threshold=trial.suggest_float(
                    "threshold",
                    1e-5,
                    1e-3,
                    log=True,
                ),
                scheduler_patience=scheduler_patience,
            ),
            batch_size=trial.suggest_categorical(
                "batch_size",
                [
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                    1024,
                    2048,
                ],
            ),
            decision_weight=trial.suggest_int("decision_weight", 14, 23),
        )

        if d_model % num_heads != 0:
            raise optuna.TrialPruned

        model_params = ModelParameters(
            data_dimensions=DataDimensions(
                num_heroes=model_trainer.hero_data_manager.get_heroes_number(),
                num_patches=get_patches_number(),
            ),
            synergy_parameters=SynergyParameters(
                num_heads=num_heads,
                num_layers=num_synergy_layers,
                ffn_ratio=ffn_ratio,
            ),
            matchup_parameters=MatchupParameters(
                num_heads=num_heads,
                num_layers=num_matchup_layers,
            ),
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
        trial.set_user_attr("model_trainable_params", trainable_params)

        model_trainer.setup_custom_training(model, training_arguments)
        model_trainer.train_model(
            trial=trial,
        )

        assert model_trainer.training_components is not None
        assert (
            model_trainer.training_components.early_stopping.best_metrics
            is not None
        )
        return float(
            model_trainer.training_components.early_stopping.best_metrics.loss,
        )

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
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=12,
            interval_steps=1,
        ),
        storage=settings.OPTUNA_STORAGE,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=250,
        show_progress_bar=True,
    )
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html("optimization_history.html")

    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_html("param_importances.html")

    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_trials.csv", index=False)
