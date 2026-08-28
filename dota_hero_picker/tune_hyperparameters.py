import logging
from collections.abc import Callable
from pathlib import Path

import optuna
from optuna import Trial

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
    NNParameters,
    RNNWinPredictor,
    SiameseDraftPredictor,
    SiameseParameters,
)
from .patch_resolver import get_patches_number

logger = logging.getLogger(__name__)

hero_data_manager = HeroDataManager()


def create_objective(
    model_trainer: ModelTrainer,
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        use_pos_weight = trial.suggest_categorical(
            "use_pos_weight",
            [False, True],
        )

        d_model = trial.suggest_categorical(
            "d_model",
            [16, 32, 64, 128],
        )
        valid_heads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
        num_heads = trial.suggest_categorical("num_heads", valid_heads)

        num_synergy_layers = trial.suggest_int("num_synergy_layers", 1, 3)
        dropout_rate = trial.suggest_float(
            "dropout_rate",
            0.15,
            0.55,
        )
        patch_embedding_dim = trial.suggest_categorical(
            "patch_embedding_dim",
            [1, 2, 4, 8, 16],
        )

        scheduler_patience = trial.suggest_int(
            "scheduler_patience",
            4,
            19,
        )
        early_stopping_patience = trial.suggest_int(
            "early_stopping_patience",
            12,
            28,
        )

        model_params = SiameseParameters(
            num_heroes=model_trainer.hero_data_manager.get_heroes_number(),
            num_patches=get_patches_number(),
            d_model=d_model,
            num_heads=num_heads,
            num_synergy_layers=num_synergy_layers,
            dropout_rate=dropout_rate,
            patch_embedding_dim=patch_embedding_dim,
        )
        model = SiameseDraftPredictor(model_params)

        trainable_params = count_trainable_params(model)
        trial.set_user_attr("model_trainable_params", trainable_params)

        training_arguments = TrainingArguments(
            data=TrainingData(
                train_dataset=model_trainer.data_manager.train_dataset,
                val_dataset=model_trainer.data_manager.val_dataset,
            ),
            pos_weight=model_trainer.data_manager.pos_weight
            if use_pos_weight
            else None,
            early_stopping_patience=(early_stopping_patience),
            optimizer_parameters=OptimizerParameters(
                lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
                weight_decay=trial.suggest_float(
                    "weight_decay",
                    1e-7,
                    1e-1,
                    log=True,
                ),
            ),
            scheduler_parameters=SchedulerParameters(
                factor=trial.suggest_float(
                    "factor",
                    0.6,
                    0.9,
                ),
                threshold=trial.suggest_float(
                    "threshold",
                    1e-4,
                    1,
                    log=True,
                ),
                scheduler_patience=scheduler_patience,
            ),
            batch_size=trial.suggest_categorical(
                "batch_size",
                [
                    32,
                    64,
                    128,
                    256,
                    512,
                    1024,
                ],
            ),
            decision_weight=trial.suggest_int("decision_weight", 8, 22),
        )

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
            model_trainer.training_components.early_stopping.best_metrics.mcc,
        )

    return objective


def main(csv_file_path: Path) -> None:
    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.enable_propagation()

    model_trainer = ModelTrainer(csv_file_path)
    objective = create_objective(model_trainer)

    study = optuna.create_study(
        study_name="dota_win_predictor_28_08_2026",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=8,
            max_resource=75,
            reduction_factor=3,
        ),
        storage="sqlite:///optuna_study.db",
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
