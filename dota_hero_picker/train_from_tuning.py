import logging
from pathlib import Path

import optuna
import torch

import settings
from dota_hero_picker.model_trainer import ModelTrainer
from dota_hero_picker.neural_network import (
    NNParameters,
    RNNWinPredictor,
)
from dota_hero_picker.patch_resolver import get_patches_number
from dota_hero_picker.training_utils import (
    OptimizerParameters,
    SchedulerParameters,
    TrainingArguments,
    TrainingData,
)

logger = logging.getLogger(__name__)


def train_best_model(csv_file_path: Path) -> None:
    """
    Load the best parameters from the Optuna study
    and trains the final model.
    """
    logger.info("Connecting to Optuna study database...")
    study = optuna.load_study(
        study_name="dota_win_predictor_26_08_2026",
        storage="sqlite:///optuna_study.db",
    )

    trials_dataframe = study.trials_dataframe()
    top_n_trials = 10
    top_trials = (
        trials_dataframe.loc[trials_dataframe["state"] == "COMPLETE"]
        .sort_values("value", ascending=False)
        .head(top_n_trials)
    )

    model_trainer = ModelTrainer(csv_file_path)
    num_heroes = model_trainer.hero_data_manager.get_heroes_number()

    best_overall_state: dict[str, Any] | None = None
    best_overall_test_metrics: MetricsResult | None = None
    best_trial_params: Any = None

    for idx, (_, row) in enumerate(top_trials.iterrows(), 1):
        logger.info(
            f"--- Training Candidate {idx}/{len(top_trials)} "
            f"(Val MCC: {row['value']:.4f}) ---"
        )
        model_params = NNParameters(
            num_heroes=num_heroes,
            num_patches=get_patches_number(),
            heroes_embedding_dim=row["params_heroes_embedding_dim"],
            patch_embedding_dim=row["params_patch_embedding_dim"],
            gru_hidden_dim=row["params_gru_hidden_dim"],
            num_gru_layers=row["params_num_gru_layers"],
            dropout_rate=row["params_dropout_rate"],
            bidirectional=row["params_bidirectional"],
            num_heads=row["params_num_heads"],
        )
        model = RNNWinPredictor(model_params)

        use_pos_weight = row.get("params_use_pos_weight", False)

        training_args = TrainingArguments(
            data=TrainingData(
                train_dataset=model_trainer.data_manager.train_dataset,
                val_dataset=model_trainer.data_manager.val_dataset,
            ),
            pos_weight=(
                model_trainer.data_manager.pos_weight
                if use_pos_weight
                else None
            ),
            early_stopping_patience=row["params_early_stopping_patience"],
            optimizer_parameters=OptimizerParameters(
                lr=row["params_lr"],
                weight_decay=row["params_weight_decay"],
            ),
            scheduler_parameters=SchedulerParameters(
                factor=row["params_factor"],
                threshold=row["params_threshold"],
                scheduler_patience=row["params_scheduler_patience"],
            ),
            batch_size=row["params_batch_size"],
            decision_weight=row["params_decision_weight"],
        )

        model_trainer.setup_custom_training(model, training_args)
        model_trainer.train_model()

        test_metrics = model_trainer.evaluate_on_test()
        logger.info(f"Candidate {idx} Test Metrics: {test_metrics}")

        assert model_trainer.training_components is not None
        current_state = (
            model_trainer.training_components.early_stopping.best_model_state
        )

        if (
            best_overall_test_metrics is None
            or test_metrics.mcc > best_overall_test_metrics.mcc
        ):
            best_overall_test_metrics = test_metrics
            best_overall_state = current_state
            best_trial_params = row

    if (
        best_overall_state is not None
        and best_overall_test_metrics is not None
    ):
        save_path = settings.MODELS_FOLDER_PATH / Path("stable_model.pth")
        torch.save(best_overall_state, save_path)
        logger.info("--------------------------------------------------")
        logger.info(f"Successfully saved best model to {save_path}")
        logger.info(f"Best Test Metrics: {best_overall_test_metrics}")
        logger.info("Best Trial Parameters:")
        logger.info(best_trial_params)
        logger.info("--------------------------------------------------")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    train_best_model(settings.PERSONAL_DOTA_MATCHES_PATH)
