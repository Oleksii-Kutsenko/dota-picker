import numpy as np
import optuna
import pandas as pd

import settings

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
    "early_stopping_patience",
]

def get_pareto_fronts(
    completed_trials: pd.DataFrame,
    target_candidates_count: int,
) -> pd.DataFrame:
    """Extract successive Pareto fronts across validation loss (values_0) and model parameters (values_1)."""
    remaining_trials = completed_trials.copy().reset_index(drop=True)
    pareto_fronts: list[pd.DataFrame] = []
    current_rank = 1

    while (
        not remaining_trials.empty
        and sum(len(front) for front in pareto_fronts)
        < target_candidates_count
    ):
        objective_matrix = remaining_trials[
            ["values_0", "values_1"]
        ].to_numpy()

        # A trial is dominated if another trial has <= in both objectives and strictly < in at least one
        is_dominated = [
            np.any(
                np.all(objective_matrix <= current_trial_objectives, axis=1)
                & np.any(objective_matrix < current_trial_objectives, axis=1)
            )
            for current_trial_objectives in objective_matrix
        ]

        # Extract non-dominated trials for this front, sorted along its trade-off curve
        current_front = remaining_trials[~np.array(is_dominated)].copy()
        current_front["pareto_rank"] = current_rank
        current_front = current_front.sort_values(by="values_0")
        pareto_fronts.append(current_front)

        remaining_trials = remaining_trials[is_dominated].reset_index(
            drop=True
        )
        current_rank += 1

    combined_fronts = pd.concat(pareto_fronts, ignore_index=True)
    return combined_fronts.head(target_candidates_count)


def main() -> None:
    # 1. Load latest study from storage
    study_summaries = optuna.study.get_all_study_summaries(
        storage=settings.OPTUNA_STORAGE
    )
    latest_study_summary = max(
        study_summaries, key=lambda summary: summary.datetime_start
    )
    study = optuna.load_study(
        study_name=latest_study_summary.study_name,
        storage=settings.OPTUNA_STORAGE,
    )

    trials_dataframe = study.trials_dataframe()
    completed_trials = trials_dataframe[
        trials_dataframe["state"] == "COMPLETE"
    ]

    # 2. Select top 10% of completed trials based on Pareto fronts
    top_fraction = 0.10
    target_candidates_count = max(
        1, int(np.ceil(len(completed_trials) * top_fraction))
    )
    top_candidates = get_pareto_fronts(
        completed_trials, target_candidates_count
    )

    # 3. Print top candidates table (ordered logically)
    display_columns = ["number", "pareto_rank", "values_0", "values_1"] + [
        f"params_{name}"
        for name in PARAMETER_ORDER
        if f"params_{name}" in top_candidates.columns
    ]
    display_table = top_candidates[display_columns].rename(
        columns=lambda name: name.removeprefix("params_")
    )
    print(display_table.to_string(index=False))

    # 4. Print parameter distributions (ordered logically)
    print("\n" + "=" * 80)
    print("TOP 10% PARAMETER RANGES & FREQUENCIES")
    print("=" * 80)

    first_completed_trial = next(
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    )
    parameter_distributions = first_completed_trial.distributions

    # Iterate in logical order rather than alphabetical
    for parameter_name in PARAMETER_ORDER:
        if parameter_name not in parameter_distributions:
            continue

        distribution = parameter_distributions[parameter_name]
        parameter_values = top_candidates[f"params_{parameter_name}"]

        if isinstance(
            distribution, optuna.distributions.CategoricalDistribution
        ):
            frequencies = parameter_values.value_counts()
            frequency_summary = ", ".join(
                f"{val} ({count / len(top_candidates):.0%})"
                for val, count in frequencies.items()
            )
            print(f"{parameter_name:<28} | {frequency_summary}")
        else:
            minimum_value = parameter_values.min()
            maximum_value = parameter_values.max()

            if (
                isinstance(
                    distribution, optuna.distributions.FloatDistribution
                )
                and distribution.log
            ):
                print(
                    f"{parameter_name:<28} | [{minimum_value:.2e}, {maximum_value:.2e}]"
                )
            elif isinstance(
                distribution, optuna.distributions.FloatDistribution
            ):
                print(
                    f"{parameter_name:<28} | [{minimum_value:.3f}, {maximum_value:.3f}]"
                )
            else:
                print(
                    f"{parameter_name:<28} | [{int(minimum_value)}, {int(maximum_value)}]"
                )


if __name__ == "__main__":
    main()
