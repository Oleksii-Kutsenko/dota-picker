import dataclasses
import logging
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import torch

import settings
from dota_hero_picker.model_trainer import ModelTrainer
from dota_hero_picker.neural_network import (
    SiameseDraftPredictor,
    SiameseParameters,
)
from dota_hero_picker.patch_resolver import get_patches_number
from dota_hero_picker.training_utils import (
    OptimizerParameters,
    SchedulerParameters,
    ShuffleEnum,
    TrainingArguments,
    TrainingData,
    collect_logits_and_labels,
    fit_temperature,
    get_data_loader,
)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_latest_study_name() -> str:
    """Find the most recent Optuna study in the database."""
    summaries = optuna.study.get_all_study_summaries(
        storage=settings.OPTUNA_STORAGE,
    )
    if not summaries:
        msg = "No studies found in the database"
        raise RuntimeError(msg)

    latest = max(
        summaries,
        key=lambda summary: summary.datetime_start,  # type: ignore[arg-type, return-value]
    )
    logger.info(f"Using latest study: {latest.study_name}")
    return latest.study_name


def build_candidate_setup(
    row: pd.Series,
    model_trainer: ModelTrainer,
) -> tuple[SiameseDraftPredictor, SiameseParameters, TrainingArguments]:
    """Convert an Optuna trial row into Model and TrainingArguments."""
    model_params = SiameseParameters(
        num_heroes=model_trainer.hero_data_manager.get_heroes_number(),
        num_patches=get_patches_number(),
        d_model=int(row["params_d_model"]),
        num_heads=int(row["params_num_heads"]),
        num_synergy_layers=int(row["params_num_synergy_layers"]),
        dropout_rate=float(row["params_dropout_rate"]),
        patch_embedding_dim=int(row["params_patch_embedding_dim"]),
    )
    training_args = TrainingArguments(
        data=TrainingData(
            train_dataset=model_trainer.data_manager.train_dataset,
            val_dataset=model_trainer.data_manager.val_dataset,
        ),
        early_stopping_patience=int(row["params_early_stopping_patience"]),
        optimizer_parameters=OptimizerParameters(
            lr=float(row["params_lr"]),
            weight_decay=float(row["params_weight_decay"]),
        ),
        scheduler_parameters=SchedulerParameters(
            factor=float(row["params_factor"]),
            threshold=float(row["params_threshold"]),
            scheduler_patience=int(row["params_scheduler_patience"]),
        ),
        batch_size=int(row["params_batch_size"]),
        decision_weight=int(row["params_decision_weight"]),
    )
    return SiameseDraftPredictor(model_params), model_params, training_args


def save_stable_model(
    model_state: dict[str, Any],
    model_params: SiameseParameters,
    temperature: float,
) -> None:
    """Save the best model state, architecture parameters, and temperature."""
    save_path = settings.MODELS_FOLDER_PATH / Path("stable_model.pth")
    torch.save(
        {
            "model_state": model_state,
            "model_params": dataclasses.asdict(model_params),
            "temperature": temperature,
        },
        save_path,
    )
    logger.info(f"Successfully saved best model to {save_path}")


def train_best_model(csv_file_path: Path) -> None:
    study = optuna.load_study(
        study_name=get_latest_study_name(),
        storage=settings.OPTUNA_STORAGE,
    )

    trials_dataframe = study.trials_dataframe()
    best_trial = (
        trials_dataframe.loc[trials_dataframe["state"] == "COMPLETE"]
        .sort_values("value", ascending=False)
        .iloc[0]
    )

    logger.info(f"Best Optuna trial Val MCC: {best_trial['value']:.4f}")

    model_trainer = ModelTrainer(csv_file_path)
    model, model_params, training_args = build_candidate_setup(
        best_trial,
        model_trainer,
    )

    model_trainer.setup_custom_training(model, training_args)
    model_trainer.train_model()

    assert model_trainer.training_components is not None
    early_stopping = model_trainer.training_components.early_stopping
    assert early_stopping.best_metrics is not None
    assert early_stopping.best_model_state is not None

    logger.info(f"Retrained Val Metrics: {early_stopping.best_metrics}")

    # --- Platt scaling: fit temperature on validation set ---
    val_loader = get_data_loader(
        model_trainer.data_manager.val_dataset,
        training_args.batch_size,
        ShuffleEnum.UNSHUFFLED,
    )
    val_logits, val_labels = collect_logits_and_labels(model, val_loader)
    temperature = fit_temperature(val_logits, val_labels)

    raw_test_metrics = model_trainer.evaluate_on_test(temperature=1.0)
    calibrated_test_metrics = model_trainer.evaluate_on_test(
        temperature=temperature,
    )

    save_stable_model(
        early_stopping.best_model_state,
        model_params,
        temperature,
    )

    logger.info("--------------------------------------------------")
    logger.info(f"Calibration temperature: {temperature:.4f}")
    logger.info(f"Best Validation Metrics: {early_stopping.best_metrics}")
    logger.info(f"Test Metrics (Uncalibrated T=1.0): {raw_test_metrics}")
    logger.info(
        f"Test Metrics (Calibrated T={temperature:.4f}): "
        f"{calibrated_test_metrics}",
    )
    logger.info(
        f"Calibration Impact: Loss {raw_test_metrics.loss:.4f} "
        f"-> {calibrated_test_metrics.loss:.4f} | "
        f"ECE {raw_test_metrics.ece:.4f} -> {calibrated_test_metrics.ece:.4f}",
    )
    logger.info("--------------------------------------------------")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    train_best_model(settings.PERSONAL_DOTA_MATCHES_PATH)
