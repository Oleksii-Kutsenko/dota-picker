import copy
import logging
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

import settings
from dota_hero_picker.model_trainer import ModelTrainer
from dota_hero_picker.neural_network import (
    ActivationEnum,
    DataDimensions,
    MatchWinPredictor,
    ModelParameters,
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

SEEDS = [42, 123, 999]
TOP_K_CANDIDATES = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_latest_study_name() -> str:
    summaries = optuna.study.get_all_study_summaries(
        storage=settings.OPTUNA_STORAGE
    )
    if not summaries:
        raise RuntimeError("No Optuna studies found in the database.")
    latest = max(summaries, key=lambda s: s.datetime_start)
    return latest.study_name


def parse_trial(
    trial_row: pd.Series,
    trainer: ModelTrainer,
) -> tuple[MatchWinPredictor, ModelParameters, TrainingArguments]:
    model_params = ModelParameters(
        data_dimensions=DataDimensions(
            num_heroes=trainer.hero_data_manager.get_heroes_number(),
            num_patches=get_patches_number(),
        ),
        d_model=int(trial_row["params_d_model"]),
        num_layers=int(trial_row["params_num_layers"]),
        num_heads=int(trial_row["params_num_heads"]),
        ffn_ratio=int(trial_row["params_ffn_ratio"]),
        hidden_dim=int(trial_row["params_hidden_dim"]),
        activation=ActivationEnum(trial_row["params_activation"]),
        dropout_rate=float(trial_row["params_dropout_rate"]),
        patch_embedding_dim=int(trial_row["params_patch_embedding_dim"]),
        stat_projection_activation=ActivationEnum(
            trial_row["params_stat_projection_activation"],
        ),
    )
    training_args = TrainingArguments(
        data=TrainingData(
            train_dataset=trainer.data_manager.train_dataset,
            val_dataset=trainer.data_manager.val_dataset,
        ),
        batch_size=int(trial_row["params_batch_size"]),
        decision_weight=int(trial_row["params_decision_weight"]),
        optimizer_parameters=OptimizerParameters(
            lr=float(trial_row["params_lr"]),
            weight_decay=float(trial_row["params_weight_decay"]),
        ),
        scheduler_parameters=SchedulerParameters(
            scheduler_patience=int(trial_row["params_scheduler_patience"]),
            factor=float(trial_row["params_factor"]),
            threshold=float(trial_row["params_threshold"]),
        ),
        early_stopping_patience=int(
            trial_row["params_early_stopping_patience"]
        ),
    )

    model = MatchWinPredictor(
        model_params,
        trainer.hero_data_manager.get_hero_features_matrix(),
    )
    return model, model_params, training_args


def train_best_model(csv_file_path: Path) -> None:
    study_name = get_latest_study_name()
    study = optuna.load_study(
        study_name=study_name, storage=settings.OPTUNA_STORAGE
    )

    trials_df = study.trials_dataframe()
    completed = trials_df[trials_df["state"] == "COMPLETE"]

    pareto_ids = {t.number for t in study.best_trials}
    top_trials = (
        completed[completed["number"].isin(pareto_ids)]
        .sort_values("values_0")
        .head(TOP_K_CANDIDATES)
    )

    trainer = ModelTrainer(csv_file_path, random_state=42)
    logger.info(
        f"Retraining top {len(top_trials)} candidates across seeds {SEEDS}..."
    )

    best_val_auc = -float("inf")
    champion = None

    for rank, (_, trial_row) in enumerate(top_trials.iterrows(), start=1):
        trial_num = int(trial_row["number"])
        optuna_val = float(trial_row["values_0"])
        seed_aucs = []

        for seed in SEEDS:
            set_seed(seed)
            model, params, args = parse_trial(trial_row, trainer)
            trainer.setup_custom_training(model, args)
            trainer.train_model()

            early_stopping = trainer.training_components.early_stopping
            val_auc = early_stopping.best_metrics.auc
            seed_aucs.append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                champion = {
                    "trial_number": trial_num,
                    "seed": seed,
                    "val_auc": val_auc,
                    "model": model,
                    "model_state": copy.deepcopy(
                        early_stopping.best_model_state
                    ),
                    "model_params": params,
                    "training_args": args,
                }

        mean_auc, std_auc = float(np.mean(seed_aucs)), float(np.std(seed_aucs))
        logger.info(
            f"Candidate {rank}/{len(top_trials)} (Trial #{trial_num}, score: {optuna_val:.4f}): "
            f"Val AUC = {mean_auc:.4f} +/- {std_auc:.4f}",
        )

    if champion is None:
        raise RuntimeError("No candidate was successfully trained.")

    # Calibrate temperature on validation set for the champion model
    champion["model"].load_state_dict(champion["model_state"])
    val_loader = get_data_loader(
        trainer.data_manager.val_dataset,
        champion["training_args"].batch_size,
        ShuffleEnum.UNSHUFFLED,
    )
    val_logits, val_labels = collect_logits_and_labels(
        champion["model"], val_loader
    )
    temperature = fit_temperature(val_logits, val_labels)

    # Evaluate champion on the held-out test set
    trainer.model = champion["model"]
    trainer.training_arguments = champion["training_args"]
    test_metrics = trainer.evaluate_on_test(temperature)

    logger.info(
        f"\nChampion: Trial #{champion['trial_number']} (seed={champion['seed']})\n"
        f"Val AUC: {champion['val_auc']:.4f} | Test AUC: {test_metrics.auc:.4f} | "
        f"Test Loss: {test_metrics.loss:.4f} | ECE: {test_metrics.ece:.4f} | Temp: {temperature:.3f}",
    )

    # Save to stable_model.pth
    settings.MODELS_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    save_path = settings.MODELS_FOLDER_PATH / "stable_model.pth"
    torch.save(
        {
            "model_state": champion["model_state"],
            "model_params": champion["model_params"].to_dict(),
            "temperature": temperature,
        },
        save_path,
    )
    logger.info(f"Champion model saved to '{save_path}'.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    train_best_model(settings.PERSONAL_DOTA_MATCHES_PATH)


if __name__ == "__main__":
    main()
