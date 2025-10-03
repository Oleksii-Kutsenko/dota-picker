from pathlib import Path

import pandas as pd


def compute_metric(pairs_df, condition, average_winrate_func):
    filtered_pairs = pairs_df[
        condition & (pairs_df["hero_id_x"] < pairs_df["hero_id_y"])
    ]
    pair_wins = filtered_pairs.groupby(["hero_id_x", "hero_id_y"]).apply(
        lambda g: (
            (g["is_radiant_x"] & g["radiant_win"])
            | (~g["is_radiant_x"] & ~g["radiant_win"])
        ).sum(),
        include_groups=False,
    )
    pair_total = filtered_pairs.groupby(["hero_id_x", "hero_id_y"]).size()
    reliable = pair_total[pair_total > 50].index
    pair_win_rate = pair_wins.loc[reliable] / pair_total.loc[reliable]

    expected = pair_win_rate.index.map(average_winrate_func)
    return pair_win_rate - expected


def calculate_matchups(public_matches_df):
    hero_wins = public_matches_df.groupby("hero_id").apply(
        lambda group: (
            (group["is_radiant"] & group["radiant_win"])
            | (~group["is_radiant"] & ~group["radiant_win"])
        ).sum(),
        include_groups=False,
    )
    hero_total = public_matches_df.groupby("hero_id").size()
    hero_win_rate = (hero_wins / hero_total).to_dict()

    def average_winrate(row):
        return (hero_win_rate[row[0]] + hero_win_rate[row[1]]) / 2

    pairs = public_matches_df.merge(
        public_matches_df,
        on=[
            "match_id",
            "radiant_win",
        ],
    )

    synergy = compute_metric(
        pairs,
        (pairs["is_radiant_x"] == pairs["is_radiant_y"]),
        average_winrate,
    )
    counters = compute_metric(
        pairs,
        (pairs["is_radiant_x"] != pairs["is_radiant_y"]),
        average_winrate,
    )

    return {"synergy": synergy, "counters": counters}


def main(public_matches_path, matchup_statistics_path):
    public_matches_df = pd.read_csv(
        public_matches_path,
        usecols=["match_id", "radiant_win", "hero_id", "is_radiant"],
    )
    for statistics_name, statistics_series in calculate_matchups(
        public_matches_df
    ).items():
        statistics_df = statistics_series.reset_index()
        statistics_df.columns = ["hero_id_1", "hero_id_2", "score"]
        statistics_df = statistics_df.sort_values("score")
        statistics_df.to_csv(
            matchup_statistics_path / Path(statistics_name + ".csv"),
            index=False,
        )
