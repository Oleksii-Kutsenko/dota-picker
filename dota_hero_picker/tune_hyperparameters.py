import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna import Trial
from sklearn.model_selection import StratifiedKFold

from dota_hero_picker.hero_data_manager import HeroDataManager
from dota_hero_picker.train_model import (
    DotaDataset,
    MetricsResult,
    ModelTrainer,
    OptimizerParameters,
    SchedulerParameters,
    TrainingArguments,
    TrainingData,
    compute_pos_weight,
    count_trainable_params,
    train_model,
)

from .data_preparation import (
    create_augmented_dataframe,
    prepare_dataframe,
)
from .neural_network import (
    NNParameters,
    RNNWinPredictor,
)

logger = logging.getLogger(__name__)


def perform_fold(
    matches_dataframe: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    model: RNNWinPredictor,
    training_arguments: TrainingArguments,
) -> MetricsResult:
    train_df = matches_dataframe.iloc[train_idx]
    val_df = matches_dataframe.iloc[val_idx]

    augmented_train = create_augmented_dataframe(train_df)
    prepared_val = prepare_dataframe(val_df)
    train_dataset = DotaDataset(augmented_train)
    val_dataset = DotaDataset(prepared_val)

    training_arguments.data = TrainingData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    model, metrics = train_model(
        model,
        training_arguments,
    )
    return metrics


def perform_cross_validation(
    matches_dataframe: pd.DataFrame,
    training_arguments: TrainingArguments,
    nn_parameters: NNParameters,
) -> tuple[np.floating[Any], np.floating[Any], np.floating[Any], int]:
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    folds_val_f1 = []
    folds_val_loss = []
    folds_val_auc = []

    for train_idx, val_idx in skf.split(
        matches_dataframe,
        matches_dataframe["win"],
    ):
        model = RNNWinPredictor(nn_parameters)

        metrics = perform_fold(
            matches_dataframe,
            train_idx,
            val_idx,
            model,
            training_arguments,
        )

        folds_val_f1.append(metrics.f1)
        folds_val_loss.append(metrics.loss)
        folds_val_auc.append(metrics.auc)
    return (
        np.mean(folds_val_f1),
        np.mean(folds_val_loss),
        np.mean(folds_val_auc),
        count_trainable_params(model),
    )


def create_objective(
    model_trainer: ModelTrainer,
) -> Callable[[Trial], tuple[float, float, float]]:
    num_heroes = len(HeroDataManager().raw_hero_data)
    matches_dataframe = ModelTrainer(
        model_trainer.csv_file_path
    ).create_matches_dataframe()

    pos_weight = compute_pos_weight(matches_dataframe)

    def objective(trial: Trial) -> tuple[float, float, float]:
        use_pos_weight = trial.suggest_categorical(
            "use_pos_weight", [False, True]
        )

        embedding_dim = trial.suggest_categorical(
            "embedding_dim", [16, 32, 64]
        )
        gru_hidden_dim = trial.suggest_categorical(
            "gru_hidden_dim", [64, 128, 256]
        )
        num_gru_layers = trial.suggest_categorical("num_gru_layers", [1, 2])
        bidirectional = trial.suggest_categorical(
            "bidirectional", [True, False]
        )

        dropout_rate = (
            trial.suggest_float("dropout_rate", 0.4, 0.7, step=0.05)
            if num_gru_layers > 1
            else trial.suggest_float("dropout_rate", 0.2, 0.5, step=0.05)
        )

        training_arguments = TrainingArguments(
            data=TrainingData(
                train_dataset=DotaDataset(pd.DataFrame()),
                val_dataset=DotaDataset(pd.DataFrame()),
            ),
            pos_weight=pos_weight if use_pos_weight else None,
            early_stopping_patience=(
                trial.suggest_int("early_stopping_patience", 3, 9, step=2)
            ),
            optimizer_parameters=OptimizerParameters(
                lr=trial.suggest_float("lr", 5e-5, 5e-4, log=True),
                weight_decay=trial.suggest_float(
                    "weight_decay", 1e-4, 0.1, log=True
                ),
            ),
            epochs=trial.suggest_int("epochs", 15, 40),
            scheduler_parameters=SchedulerParameters(
                factor=trial.suggest_float("factor", 0.3, 0.55, step=0.05),
                threshold=trial.suggest_float(
                    "threshold", 0.01, 0.1, log=True
                ),
                scheduler_patience=trial.suggest_int(
                    "scheduler_patience", 15, 25
                ),
            ),
            batch_size=trial.suggest_categorical(
                "batch_size",
                [64, 128, 256],
            ),
            decision_weight=trial.suggest_int("decision_weight", 7, 11),
        )

        mean_f1, mean_val_loss, mean_val_auc, trainable_params = (
            perform_cross_validation(
                matches_dataframe,
                training_arguments,
                NNParameters(
                    num_heroes=num_heroes,
                    embedding_dim=embedding_dim,
                    gru_hidden_dim=gru_hidden_dim,
                    num_gru_layers=num_gru_layers,
                    dropout_rate=dropout_rate,
                    bidirectional=bidirectional,
                ),
            )
        )

        trial.set_user_attr("mean_val_loss", mean_val_loss)
        trial.set_user_attr(
            "model_trainable_params",
            trainable_params,
        )
        trial.set_user_attr("model_val_auc", mean_val_auc)

        return float(mean_f1), float(mean_val_loss), float(mean_val_auc)

    return objective


def main(csv_file_path: str) -> None:
    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.enable_propagation()

    model_trainer = ModelTrainer(csv_file_path)
    objective = create_objective(model_trainer)

    study = optuna.create_study(
        study_name="dota_win_predictor",
        directions=["maximize", "minimize", "maximize"],
        sampler=optuna.samplers.NSGAIIISampler(),
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=300,
        show_progress_bar=True,
    )

    def select_f1(trial: optuna.trial.FrozenTrial) -> float:
        return float(trial.values[0])  # noqa: PD011

    fig1 = optuna.visualization.plot_optimization_history(
        study,
        target=select_f1,
        target_name="F1",
    )
    fig1.write_html("optimization_history.html")

    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_html("param_importances.html")

    fig3 = optuna.visualization.plot_pareto_front(
        study,
        target_names=["F1", "Val Loss", "AUC"],
    )
    fig3.write_html("pareto_front.html")

    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_trials.csv", index=False)
