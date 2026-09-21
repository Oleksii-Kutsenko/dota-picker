import logging

import numpy as np
import optuna
import pandas as pd

from dota_hero_picker.train import load_latest_study

logger = logging.getLogger(__name__)

PARAMETER_ORDER = [
    "d_model",
    "num_layers",
    "num_heads",
    "ffn_ratio",
    "hidden_dim",
    "activation",
    "dropout_rate",
    "patch_embedding_dim",
    "stat_projection_activation",
    "batch_size",
    "lr",
    "weight_decay",
    "decision_weight",
    "scheduler_patience",
    "factor",
    "threshold",
]


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
                -remaining_trials["values_2"].to_numpy(),
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


def main() -> None:
    study = load_latest_study()

    trials_dataframe = study.trials_dataframe()
    completed_trials = trials_dataframe[
        trials_dataframe["state"] == "COMPLETE"
    ]

    # 2. Select top 10% of completed trials based on Pareto fronts
    top_fraction = 0.10
    target_candidates_count = max(
        1,
        int(np.ceil(len(completed_trials) * top_fraction)),
    )
    top_candidates = get_pareto_fronts(
        completed_trials,
        target_candidates_count,
    )

    logger.info("=" * 80)
    logger.info("TOP 10% PARAMETER RANGES & FREQUENCIES")
    logger.info("=" * 80)

    first_completed_trial = next(
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    )
    parameter_distributions = first_completed_trial.distributions

    for parameter_name in PARAMETER_ORDER:
        if parameter_name not in parameter_distributions:
            continue

        distribution = parameter_distributions[parameter_name]
        parameter_values = top_candidates[f"params_{parameter_name}"]

        if isinstance(
            distribution,
            optuna.distributions.CategoricalDistribution,
        ):
            frequencies = parameter_values.value_counts().sort_index()
            frequency_summary = ", ".join(
                f"{val} ({count / len(top_candidates):.0%})"
                for val, count in frequencies.items()
            )
            logger.info(f"{parameter_name:<28} | {frequency_summary}")
        else:
            minimum_value = parameter_values.min()
            maximum_value = parameter_values.max()

            if (
                isinstance(
                    distribution,
                    optuna.distributions.FloatDistribution,
                )
                and distribution.log
            ):
                logger.info(
                    f"{parameter_name:<28} | "
                    f"[{minimum_value:.2e}, {maximum_value:.2e}]",
                )
            elif isinstance(
                distribution,
                optuna.distributions.FloatDistribution,
            ):
                logger.info(
                    f"{parameter_name:<28} | "
                    f"[{minimum_value:.3f}, {maximum_value:.3f}]",
                )
            else:
                logger.info(
                    f"{parameter_name:<28} | "
                    f"[{int(minimum_value)}, {int(maximum_value)}]",
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
