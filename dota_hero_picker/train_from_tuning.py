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
        study_name="dota_win_predictor",
        storage="sqlite:///optuna_study.db",
    )

    trails_dataframe = study.trials_dataframe().sort_values(
        "value",
        ascending=False,
    )
    model_trainer = ModelTrainer(csv_file_path)
    num_heroes = model_trainer.hero_data_manager.get_heroes_number()
    for _, row in trails_dataframe.iterrows():
        model_params = NNParameters(
            num_heroes=num_heroes,
            num_patches=get_patches_number(),
            heroes_embedding_dim=row["params_heroes_embedding_dim"],
            patch_embedding_dim=row["params_patch_embedding_dim"],
            gru_hidden_dim=row["params_gru_hidden_dim"],
            num_gru_layers=row["params_num_gru_layers"],
            dropout_rate=row["params_dropout_rate"],
            bidirectional=row["params_bidirectional"],
        )
        best_model = RNNWinPredictor(model_params)

        use_pos_weight = row["params_use_pos_weight"]

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

        logger.info("Setting up custom training with best parameters...")
        model_trainer.setup_custom_training(best_model, training_args)

        logger.info("Starting final model training...")
        model_trainer.train_model()

        logger.info("Training complete. Evaluating on test set...")
        test_metrics = model_trainer.evaluate_on_test()

        logger.info(f"Final Test Metrics: {test_metrics}")
        if test_metrics.mcc >= row["value"]:
            logger.info("Found good model. Parameters: ")
            logger.info(row)
            break

    assert model_trainer.training_components is not None
    save_path = settings.MODELS_FOLDER_PATH / Path("stable_model.pth")
    torch.save(
        model_trainer.training_components.early_stopping.best_model_state,
        save_path,
    )
    logger.info(f"Successfully trained and saved best model to {save_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    train_best_model(settings.PERSONAL_DOTA_MATCHES_PATH)
