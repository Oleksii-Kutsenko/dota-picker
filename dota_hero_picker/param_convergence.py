import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd

import settings
from dota_hero_picker.baseline import compute_meta_baseline
from dota_hero_picker.data_manager import DataManager
from dota_hero_picker.hero_data_manager import HeroDataManager
from dota_hero_picker.train import load_latest_study
from dota_hero_picker.tune import PARAMETER_ORDER

logger = logging.getLogger(__name__)


def get_pareto_fronts(
    completed_trials: pd.DataFrame,
    target_candidates_count: int,
) -> pd.DataFrame:
    remaining_trials = completed_trials.copy().reset_index(drop=True)
    pareto_fronts: list[pd.DataFrame] = []
    current_rank = 1

    while (
        not remaining_trials.empty
        and sum(len(front) for front in pareto_fronts)
        < target_candidates_count
    ):
        objective_matrix = np.column_stack(
            [
                remaining_trials["values_0"].to_numpy(),
                remaining_trials["values_1"].to_numpy(),
            ],
        )

        is_dominated = [
            np.any(
                np.all(objective_matrix <= current_trial_objectives, axis=1)
                & np.any(objective_matrix < current_trial_objectives, axis=1),
            )
            for current_trial_objectives in objective_matrix
        ]

        current_front = remaining_trials[~np.array(is_dominated)].copy()
        current_front["pareto_rank"] = current_rank
        current_front = current_front.sort_values(by="values_0")
        pareto_fronts.append(current_front)

        remaining_trials = remaining_trials[is_dominated].reset_index(
            drop=True,
        )
        current_rank += 1

    combined_fronts = pd.concat(pareto_fronts, ignore_index=True)
    return combined_fronts.head(target_candidates_count)


def log_parameter_convergence(
    top_candidates: pd.DataFrame,
    parameter_distributions: dict[str, Any],
) -> None:
    """Log distribution ranges and frequencies for candidate trials."""
    min_loss = top_candidates["values_0"].min()
    max_loss = top_candidates["values_0"].max()

    logger.info("=" * 80)
    logger.info(
        f"SELECTED {len(top_candidates)} CANDIDATES | "
        f"VAL LOSS: [{min_loss:.4f}, {max_loss:.4f}]",
    )
    logger.info("=" * 80)

    for param_name in PARAMETER_ORDER:
        distribution = parameter_distributions[param_name]
        param_values = top_candidates[f"params_{param_name}"]

        if isinstance(
            distribution,
            optuna.distributions.CategoricalDistribution,
        ):
            frequencies = param_values.value_counts().sort_index()
            summary = ", ".join(
                f"{val} ({count / len(top_candidates):.0%})"
                for val, count in frequencies.items()
            )
            logger.info(f"{param_name:<28} | {summary}")
        else:
            min_val = param_values.min()
            max_val = param_values.max()

            if (
                isinstance(
                    distribution,
                    optuna.distributions.FloatDistribution,
                )
                and distribution.log
            ):
                range_str = f"[{min_val:.2e}, {max_val:.2e}]"
            elif isinstance(
                distribution,
                optuna.distributions.FloatDistribution,
            ):
                range_str = f"[{min_val:.3f}, {max_val:.3f}]"
            else:
                range_str = f"[{int(min_val)}, {int(max_val)}]"

            logger.info(f"{param_name:<28} | {range_str}")


def main() -> None:
    study = load_latest_study()

    hero_data_manager = HeroDataManager()
    data_manager = DataManager(
        settings.PERSONAL_DOTA_MATCHES_PATH,
        hero_data_manager,
    )
    baseline_loss, _ = compute_meta_baseline(data_manager)

    trials_dataframe = study.trials_dataframe()
    completed_trials = trials_dataframe[
        trials_dataframe["state"] == "COMPLETE"
    ]
    qualifying_trials = completed_trials[
        completed_trials["values_0"] < baseline_loss
    ]

    top_fraction = 0.10
    target_candidates_count = max(
        1,
        int(np.ceil(len(completed_trials) * top_fraction)),
    )
    top_candidates = get_pareto_fronts(
        qualifying_trials,
        target_candidates_count,
    )

    first_completed_trial = next(
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    )
    log_parameter_convergence(
        top_candidates,
        first_completed_trial.distributions,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
