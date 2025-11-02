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
)

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

        embedding_dim = trial.suggest_categorical(
            "embedding_dim",
            [8, 16, 32, 64, 128],
        )
        gru_hidden_dim = trial.suggest_categorical(
            "gru_hidden_dim",
            [
                8,
                16,
                32,
                64,
                128,
                256,
            ],
        )
        num_gru_layers = trial.suggest_int("num_gru_layers", 1, 5)
        bidirectional = trial.suggest_categorical(
            "bidirectional",
            [True, False],
        )

        dropout_rate = (
            trial.suggest_float(
                "dropout_rate",
                0.2,
                0.8,
            )
            if num_gru_layers > 1
            else trial.suggest_float(
                "dropout_rate",
                0.1,
                0.7,
            )
        )
        scheduler_patience = trial.suggest_int(
            "scheduler_patience",
            1,
            20,
        )
        early_stopping_patience = trial.suggest_int(
            "early_stopping_patience",
            scheduler_patience + 1,
            scheduler_patience + 14,
        )

        model_params = NNParameters(
            num_heroes=model_trainer.hero_data_manager.get_heroes_number(),
            embedding_dim=embedding_dim,
            gru_hidden_dim=gru_hidden_dim,
            num_gru_layers=num_gru_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
        )
        model = RNNWinPredictor(model_params)

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
                    1e-8,
                    0.1,
                    log=True,
                ),
            ),
            scheduler_parameters=SchedulerParameters(
                factor=trial.suggest_float(
                    "factor",
                    0.3,
                    0.9,
                ),
                threshold=trial.suggest_float(
                    "threshold",
                    0.00001,
                    1e1,
                    log=True,
                ),
                scheduler_patience=scheduler_patience,
            ),
            batch_size=trial.suggest_categorical(
                "batch_size",
                [8, 16, 32, 64, 128, 256, 512, 1024],
            ),
            decision_weight=trial.suggest_int("decision_weight", 5, 25),
        )

        model_trainer.setup_custom_training(model, training_arguments)
        model_trainer.train_model()

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
        study_name="dota_win_predictor",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=100,
        show_progress_bar=True,
    )
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html("optimization_history.html")

    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_html("param_importances.html")

    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_trials.csv", index=False)
