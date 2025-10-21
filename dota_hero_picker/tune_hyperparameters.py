import logging
from collections.abc import Callable

import numpy as np
import optuna
from optuna import Trial
from sklearn.model_selection import StratifiedKFold

from dota_hero_picker.train_model import (
    DotaDataset,
    ModelTrainer,
    OptimizerParameters,
    SchedulerParameters,
    TrainingArguments,
    TrainingData,
    compute_pos_weight,
    count_trainable_params,
    load_personal_matches,
    set_seed,
    train_model,
)

from .data_preparation import (
    create_augmented_dataframe,
    num_heroes,
    prepare_dataframe,
)
from .neural_network import (
    WinPredictorWithPositionalAttention,
)

logger = logging.getLogger(__name__)

set_seed(42)


def create_objective(
    model_trainer: ModelTrainer,
) -> Callable[[Trial], float]:
    matches_dataframe = load_personal_matches(model_trainer.csv_file_path)
    pos_weight = compute_pos_weight(matches_dataframe)

    def objective(trial: Trial) -> float:
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float(
            "weight_decay",
            1e-5,
            1e-1,
            log=True,
        )
        scheduler_patience = trial.suggest_int("scheduler_patience", 5, 20)
        early_stopping_patience = trial.suggest_int(
            "early_stopping_patience",
            5,
            19,
        )
        epochs = trial.suggest_int("epochs", 1, 50)
        factor = trial.suggest_float("factor", 0.1, 0.5, step=0.05)
        threshold = trial.suggest_float("threshold", 1e-4, 1e-2, step=1e-4)
        use_pos_weight = trial.suggest_categorical(
            "use_pos_weight",
            [True, False],
        )

        decision_weight = trial.suggest_int("decision_weight", 1, 10)
        batch_size = trial.suggest_categorical(
            "batch_size", [8, 16, 32, 64, 128]
        )

        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.6, step=0.05)
        embedding_dim = trial.suggest_categorical(
            "embedding_dim",
            [
                8,
                16,
                32,
            ],
        )

        n_layers = trial.suggest_int("n_layers", 1, 4)
        all_options = [1, 2, 4, 8, 16, 32, 64]
        layer_sizes = []
        for i in range(n_layers):
            layer_sizes.append(
                trial.suggest_categorical(
                    f"layer_{i}_size",
                    all_options,
                )
            )

        model = WinPredictorWithPositionalAttention(
            num_heroes,
            embedding_dim,
            tuple(layer_sizes),
            dropout_rate,
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True)
        folds_val_f1 = []
        folds_val_loss = []
        folds_val_auc = []

        for train_idx, val_idx in skf.split(
            matches_dataframe,
            matches_dataframe["win"],
        ):
            train_df = matches_dataframe.iloc[train_idx]
            val_df = matches_dataframe.iloc[val_idx]

            augmented_train = create_augmented_dataframe(train_df)
            prepared_val = prepare_dataframe(val_df)
            train_dataset = DotaDataset(augmented_train)
            val_dataset = DotaDataset(prepared_val)

            training_arguments = TrainingArguments(
                data=TrainingData(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                ),
                early_stopping_patience=early_stopping_patience,
                optimizer_parameters=OptimizerParameters(
                    lr=learning_rate,
                    weight_decay=weight_decay,
                ),
                epochs=epochs,
                scheduler_parameters=SchedulerParameters(
                    factor=factor,
                    threshold=threshold,
                    scheduler_patience=scheduler_patience,
                ),
                batch_size=batch_size,
                decision_weight=decision_weight,
            )

            if use_pos_weight:
                training_arguments.pos_weight = pos_weight
            else:
                training_arguments.pos_weight = None

            model, metrics = train_model(
                model,
                training_arguments,
            )
            folds_val_f1.append(metrics.f1)
            folds_val_loss.append(metrics.loss)
            folds_val_auc.append(metrics.auc)
        mean_f1 = np.mean(folds_val_f1)
        mean_val_loss = np.mean(folds_val_loss)
        mean_val_auc = np.mean(folds_val_auc)

        trial.set_user_attr("mean_val_loss", mean_val_loss)
        trial.set_user_attr(
            "model_trainable_params", count_trainable_params(model)
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
        sampler=optuna.samplers.GPSampler(),
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=90,
        show_progress_bar=True,
    )

    fig1 = optuna.visualization.plot_optimization_history(
        study, target=lambda trial: trial.values[0], target_name="F1"
    )
    fig1.write_html("optimization_history.html")

    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_html("param_importances.html")

    fig3 = optuna.visualization.plot_pareto_front(
        study, target_names=["F1", "Val Loss", "AUC"]
    )
    fig3.write_html("pareto_front.html")

    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_trials.csv", index=False)
