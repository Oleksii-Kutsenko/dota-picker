import copy
import dataclasses
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
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
    MetricsResult,
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

TOP_CANDIDATES_COUNT = 5
SEEDS_COUNT = 3
MASTER_SEED = 42
RANDOM_SEEDS = (
    np.random.default_rng(MASTER_SEED)
    .integers(
        low=1,
        high=100000,
        size=SEEDS_COUNT,
    )
    .tolist()
)


def set_seed(seed: int) -> None:
    """Set random seed across all libraries for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_latest_study_name() -> str:
    """Find the most recent Optuna study in the database."""
    summaries = optuna.study.get_all_study_summaries(
        storage=settings.OPTUNA_STORAGE,
    )
    if not summaries:
        msg = "No studies found in the database"
        raise RuntimeError(msg)

    latest_summary = max(
        summaries,
        key=lambda summary: summary.datetime_start,  # type: ignore[arg-type, return-value]
    )
    logger.info(f"Using latest study: {latest_summary.study_name}")
    return latest_summary.study_name


def build_candidate_setup(
    trial_row: pd.Series,
    model_trainer: ModelTrainer,
) -> tuple[SiameseDraftPredictor, SiameseParameters, TrainingArguments]:
    """Convert an Optuna trial row into Model and TrainingArguments."""
    model_params = SiameseParameters(
        num_heroes=model_trainer.hero_data_manager.get_heroes_number(),
        num_patches=get_patches_number(),
        d_model=int(trial_row["params_d_model"]),
        num_heads=int(trial_row["params_num_heads"]),
        num_synergy_layers=int(trial_row["params_num_synergy_layers"]),
        dropout_rate=float(trial_row["params_dropout_rate"]),
        patch_embedding_dim=int(trial_row["params_patch_embedding_dim"]),
    )
    training_arguments = TrainingArguments(
        data=TrainingData(
            train_dataset=model_trainer.data_manager.train_dataset,
            val_dataset=model_trainer.data_manager.val_dataset,
        ),
        early_stopping_patience=int(
            trial_row["params_early_stopping_patience"],
        ),
        optimizer_parameters=OptimizerParameters(
            lr=float(trial_row["params_lr"]),
            weight_decay=float(trial_row["params_weight_decay"]),
        ),
        scheduler_parameters=SchedulerParameters(
            factor=float(trial_row["params_factor"]),
            threshold=float(trial_row["params_threshold"]),
            scheduler_patience=int(trial_row["params_scheduler_patience"]),
        ),
        batch_size=int(trial_row["params_batch_size"]),
        decision_weight=int(trial_row["params_decision_weight"]),
    )
    return (
        SiameseDraftPredictor(model_params),
        model_params,
        training_arguments,
    )


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


@dataclasses.dataclass
class SeedRunResult:
    """Outcome of a single training run on a specific random seed."""

    seed: int
    test_metrics: MetricsResult
    model_state: dict[str, Any]
    model_params: SiameseParameters
    temperature: float


@dataclasses.dataclass
class CandidateTrial:
    """Evaluated candidate trial across all random seeds."""

    trial_number: int
    optuna_score: float
    seed_runs: list[SeedRunResult] = dataclasses.field(default_factory=list)

    @property
    def mean_test_auc(self) -> float:
        return float(
            np.mean([run.test_metrics.auc for run in self.seed_runs]),
        )

    @property
    def std_test_auc(self) -> float:
        return float(np.std([run.test_metrics.auc for run in self.seed_runs]))

    @property
    def mean_test_loss(self) -> float:
        return float(
            np.mean([run.test_metrics.loss for run in self.seed_runs]),
        )

    @property
    def std_test_loss(self) -> float:
        return float(
            np.std([run.test_metrics.loss for run in self.seed_runs]),
        )

    @property
    def mean_test_ece(self) -> float:
        return float(
            np.mean([run.test_metrics.ece for run in self.seed_runs]),
        )

    @property
    def mean_test_mcc(self) -> float:
        return float(
            np.mean([run.test_metrics.mcc for run in self.seed_runs]),
        )

    @property
    def best_run(self) -> SeedRunResult:
        """The run that achieved the highest test AUC."""
        return max(
            self.seed_runs,
            key=lambda run: run.test_metrics.auc,
        )


def evaluate_trial_seed(
    trial_row: pd.Series,
    model_trainer: ModelTrainer,
    seed: int,
) -> SeedRunResult:
    """Train and evaluate a single seed for a given trial configuration."""
    set_seed(seed)

    model, model_params, training_args = build_candidate_setup(
        trial_row,
        model_trainer,
    )
    model_trainer.setup_custom_training(model, training_args)
    model_trainer.train_model()

    assert model_trainer.training_components is not None
    early_stopping = model_trainer.training_components.early_stopping
    assert early_stopping.best_model_state is not None

    # Calibrate temperature on the validation set
    validation_loader = get_data_loader(
        model_trainer.data_manager.val_dataset,
        training_args.batch_size,
        ShuffleEnum.UNSHUFFLED,
    )
    validation_logits, validation_labels = collect_logits_and_labels(
        model,
        validation_loader,
    )
    temperature = fit_temperature(validation_logits, validation_labels)

    # Evaluate on the held-out test set
    calibrated_test_metrics = model_trainer.evaluate_on_test(
        temperature=temperature,
    )

    return SeedRunResult(
        seed=seed,
        test_metrics=calibrated_test_metrics,
        model_state=copy.deepcopy(early_stopping.best_model_state),
        model_params=model_params,
        temperature=temperature,
    )


def save_evaluation_report(
    candidate_trials: list[CandidateTrial],
) -> None:
    """Persist all individual seed runs across candidates to CSV."""
    report_path: Path = Path("tuning_evaluation_report.csv")
    report_records = [
        {
            "trial_number": candidate.trial_number,
            "seed": run.seed,
            "optuna_score": candidate.optuna_score,
            "test_auc": run.test_metrics.auc,
            "test_loss": run.test_metrics.loss,
            "test_ece": run.test_metrics.ece,
            "test_mcc": run.test_metrics.mcc,
            "temperature": run.temperature,
        }
        for candidate in candidate_trials
        for run in candidate.seed_runs
    ]
    pd.DataFrame(report_records).to_csv(report_path, index=False)
    logger.info(f"Detailed run records saved to {report_path.resolve()}")


def get_top_trials(
    study: optuna.Study,
    top_candidates_count: int = TOP_CANDIDATES_COUNT,
) -> pd.DataFrame:
    """Retrieve the top completed trials sorted by the study direction."""
    trials_dataframe = study.trials_dataframe()
    completed_trials = trials_dataframe.loc[
        trials_dataframe["state"] == "COMPLETE"
    ]
    if completed_trials.empty:
        msg = f"No completed trials found in study {study.study_name}"
        raise RuntimeError(msg)

    is_maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
    sorted_trials = completed_trials.sort_values(
        "value",
        ascending=not is_maximize,
    )
    return sorted_trials.head(top_candidates_count)


def evaluate_candidate_trial(
    trial_row: pd.Series,
    model_trainer: ModelTrainer,
    seeds: list[int],
    rank: int,
    total_candidates: int,
) -> CandidateTrial:
    """Evaluate a single candidate trial across multiple random seeds."""
    trial_number = int(trial_row["number"])
    optuna_score = float(trial_row["value"])
    logger.info(
        f"Evaluating Candidate {rank}/{total_candidates} "
        f"(Trial #{trial_number}, Optuna Score: {optuna_score:.4f})",
    )

    candidate = CandidateTrial(
        trial_number=trial_number,
        optuna_score=optuna_score,
    )

    for seed in seeds:
        logger.info(f"Training Trial #{trial_number} with seed={seed}...")
        run_result = evaluate_trial_seed(trial_row, model_trainer, seed)
        candidate.seed_runs.append(run_result)

        logger.info(
            f"Seed {seed} Result: "
            f"AUC={run_result.test_metrics.auc:.4f}, "
            f"Loss={run_result.test_metrics.loss:.4f}, "
            f"ECE={run_result.test_metrics.ece:.4f}, "
            f"MCC={run_result.test_metrics.mcc:.4f}",
        )

    return candidate


def train_best_model(csv_file_path: Path) -> None:
    """Retrain top Optuna candidates."""
    study = optuna.load_study(
        study_name=get_latest_study_name(),
        storage=settings.OPTUNA_STORAGE,
    )
    top_trials = get_top_trials(
        study,
        top_candidates_count=TOP_CANDIDATES_COUNT,
    )
    logger.info(
        f"Selected top {len(top_trials)} candidate trials for retraining.",
    )

    model_trainer = ModelTrainer(csv_file_path, random_state=MASTER_SEED)
    total_candidates = len(top_trials)

    candidate_trials = [
        evaluate_candidate_trial(
            trial_row=trial_row,
            model_trainer=model_trainer,
            seeds=RANDOM_SEEDS,
            rank=rank,
            total_candidates=total_candidates,
        )
        for rank, (_, trial_row) in enumerate(top_trials.iterrows(), start=1)
    ]

    # Rank candidate trials by mean Test AUC (highest first)
    candidate_trials.sort(key=lambda item: item.mean_test_auc, reverse=True)

    # Print summary benchmark table
    logger.info("=" * 79)
    logger.info("TOP CANDIDATES RETRAINING BENCHMARK (SEEDS: 42, 123, 999)")
    logger.info("=" * 79)
    header = (
        f"{'Rank':<5} | {'Trial':<7} | {'Optuna':<8} | "
        f"{'Mean AUC (std)':<18} | {'Mean Loss (std)':<18} | "
        f"{'Mean ECE':<8} | {'Mean MCC':<8}"
    )
    logger.info(header)
    logger.info("-" * 79)
    for rank, candidate in enumerate(candidate_trials, start=1):
        auc_summary = (
            f"{candidate.mean_test_auc:.4f} ({candidate.std_test_auc:.4f})"
        )
        loss_summary = (
            f"{candidate.mean_test_loss:.4f} ({candidate.std_test_loss:.4f})"
        )
        row_display = (
            f"{rank:<5} | #{candidate.trial_number:<6} | "
            f"{candidate.optuna_score:<8.4f} | "
            f"{auc_summary:<18} | {loss_summary:<18} | "
            f"{candidate.mean_test_ece:<8.4f} | "
            f"{candidate.mean_test_mcc:<8.4f}"
        )
        logger.info(row_display)
    logger.info("=" * 79)

    # Save detailed evaluation CSV report
    save_evaluation_report(
        candidate_trials,
    )

    # Save champion model (Rank 1 by mean Test AUC)
    champion_trial = candidate_trials[0]
    best_champion_run = champion_trial.best_run
    logger.info(
        f"Champion: Trial #{champion_trial.trial_number} with "
        f"Mean Test AUC: {champion_trial.mean_test_auc:.4f} "
        f"(std: {champion_trial.std_test_auc:.4f})",
    )
    save_stable_model(
        best_champion_run.model_state,
        best_champion_run.model_params,
        best_champion_run.temperature,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    train_best_model(settings.PERSONAL_DOTA_MATCHES_PATH)
