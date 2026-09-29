import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import torch

import settings
from dota_hero_picker.baseline import compute_meta_baseline
from dota_hero_picker.data_manager import DataManager
from dota_hero_picker.hero_data_manager import HeroDataManager
from dota_hero_picker.model_trainer import ModelTrainer
from dota_hero_picker.neural_network import (
    MatchWinPredictor,
    ModelParameters,
)
from dota_hero_picker.training_utils import (
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class ChampionModel:
    trial_number: int
    val_auc: float
    model: MatchWinPredictor
    model_state: dict[str, Any]
    model_params: ModelParameters
    training_args: TrainingArguments


class NoActiveStudyFoundError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("No active study found in storage.")


class NoChampionFoundError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("No champion found.")


def load_latest_study() -> optuna.Study:
    """Load the most recently started Optuna study from storage."""
    study_summaries = optuna.study.get_all_study_summaries(
        storage=settings.OPTUNA_STORAGE,
    )

    latest_study_summary: optuna.study.StudySummary | None = None
    for study_summary in study_summaries:
        if study_summary.datetime_start is None:
            continue
        if (
            latest_study_summary is None
            or latest_study_summary.datetime_start is None
            or study_summary.datetime_start
            > latest_study_summary.datetime_start
        ):
            latest_study_summary = study_summary

    if latest_study_summary is None:
        raise NoActiveStudyFoundError

    return optuna.load_study(
        study_name=latest_study_summary.study_name,
        storage=settings.OPTUNA_STORAGE,
    )


def parse_trial(
    trial: optuna.trial.FrozenTrial,
    hero_data_manager: HeroDataManager,
    data_manager: DataManager,
) -> tuple[MatchWinPredictor, ModelParameters, TrainingArguments]:
    model_params = ModelParameters.from_trial_params(
        trial.params,
        hero_data_manager,
    )
    training_args = TrainingArguments.from_trial_params(
        trial.params,
        data_manager,
    )
    model = MatchWinPredictor(
        model_params,
        hero_data_manager.get_hero_features_matrix(),
    )
    return model, model_params, training_args


def retrain_candidate(
    trial: optuna.trial.FrozenTrial,
    hero_data_manager: HeroDataManager,
    data_manager: DataManager,
) -> ChampionModel:
    model, params, args = parse_trial(
        trial,
        hero_data_manager,
        data_manager,
    )
    trainer = ModelTrainer(model, args, data_manager)
    early_stopping = trainer.train()

    return ChampionModel(
        trial_number=trial.number,
        val_auc=early_stopping.best_metrics.auc,
        model=model,
        model_state=copy.deepcopy(early_stopping.best_model_state),
        model_params=params,
        training_args=args,
    )


def select_champion(
    trials: list[optuna.trial.FrozenTrial],
    hero_data_manager: HeroDataManager,
    data_manager: DataManager,
) -> ChampionModel:
    logger.info(f"Retraining {len(trials)} Pareto-best trials...")

    best_val_auc = -float("inf")
    champion: ChampionModel | None = None

    for rank, trial in enumerate(trials, start=1):
        optuna_loss, optuna_log_params = trial.values
        candidate = retrain_candidate(trial, hero_data_manager, data_manager)

        logger.info(
            f"Candidate {rank}/{len(trials)}: Trial #{candidate.trial_number}",
        )
        logger.info(f"Optuna Loss:       {optuna_loss:.4f}")
        logger.info(f"Optuna Log Params: {optuna_log_params:.4f}")
        logger.info(f"Retrain Val AUC:   {candidate.val_auc:.4f}")

        if candidate.val_auc > best_val_auc:
            best_val_auc = candidate.val_auc
            champion = candidate

    if champion is None:
        raise NoChampionFoundError

    return champion


def train_best_model(csv_file_path: Path) -> None:
    study = load_latest_study()
    hero_data_manager = HeroDataManager()
    data_manager = DataManager(csv_file_path, hero_data_manager)

    baseline_loss, _ = compute_meta_baseline(data_manager)
    qualifying_trials = [
        trial for trial in study.best_trials if trial.values[0] < baseline_loss
    ]
    trials_to_retrain = qualifying_trials or study.best_trials

    champion = select_champion(
        trials_to_retrain,
        hero_data_manager,
        data_manager,
    )
    champion.model.load_state_dict(champion.model_state)
    champion_trainer = ModelTrainer(
        champion.model,
        champion.training_args,
        data_manager,
    )
    temperature = champion_trainer.calibrate_temperature()
    test_metrics = champion_trainer.evaluate_on_test(temperature)

    logger.info(f"Champion: Trial #{champion.trial_number}")
    logger.info(f"Validation AUC: {champion.val_auc:.4f}")
    logger.info(f"Test AUC:       {test_metrics.auc:.4f}")
    logger.info(f"Test Loss:      {test_metrics.loss:.4f}")
    logger.info(f"Test MCC:       {test_metrics.mcc:.4f}")
    logger.info(f"Test ECE:       {test_metrics.ece:.4f}")
    logger.info(f"Temperature:    {temperature:.3f}")

    settings.MODELS_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    save_path = settings.MODELS_FOLDER_PATH / "stable_model.pth"
    torch.save(
        {
            "model_state": champion.model_state,
            "model_params": champion.model_params.to_dict(),
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
