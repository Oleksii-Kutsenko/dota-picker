import logging

import pandas as pd

from .load_personal_matches import get_hero_data

logger = logging.getLogger(__name__)

hero_data = get_hero_data()
MAX_PICK = 5
hero_names = []
hero_name_2_model_id = {}
model_id_2_hero_name = {}
api_id_2_model_id = {}
num_heroes = len(hero_data)
for model_id, hero in enumerate(hero_data, 1):
    hero_name = hero["localized_name"]
    api_hero_id = hero["id"]
    hero_names.append(hero_name)
    hero_name_2_model_id[hero_name] = model_id
    model_id_2_hero_name[model_id] = hero_name
    api_id_2_model_id[api_hero_id] = model_id

visibility_map = [0, 0, 2, 2, 4]


def create_augmented_dataframe(train_dataframe: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in train_dataframe.iterrows():
        team_picks = row.team_picks
        opp_picks = row.opponent_picks
        my_pick = row.picked_hero
        win = row.win

        draft_sequence = (
            team_picks[:2]
            + opp_picks[:2]
            + team_picks[2:4]
            + opp_picks[2:4]
            + team_picks[4:]
        )
        if len(draft_sequence) != 9:
            breakpoint()

        for i, actual_pick in enumerate(team_picks):
            visible_team_picks = team_picks[:i]
            visible_opp_picks = opp_picks[
                : min(visibility_map[i], len(opp_picks) - 1)
            ]

            is_my_decision = my_pick == actual_pick
            results.append(
                {
                    "visible_team_picks": visible_team_picks,
                    "visible_opp_picks": visible_opp_picks,
                    "actual_pick": actual_pick,
                    "win": win,
                    "is_my_decision": is_my_decision,
                }
            )

        team_picks = row.opponent_picks
        opp_picks = row.team_picks
        win = 1 - row.win

        for i, actual_pick in enumerate(team_picks):
            visible_team_picks = team_picks[:i]
            visible_opp_picks = opp_picks[
                : min(visibility_map[i], len(opp_picks) - 1)
            ]
            results.append(
                {
                    "visible_team_picks": visible_team_picks,
                    "visible_opp_picks": visible_opp_picks,
                    "actual_pick": actual_pick,
                    "win": win,
                    "is_my_decision": False,
                }
            )

    return pd.DataFrame(results)


def prepare_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe[
        ["team_picks", "opponent_picks", "picked_hero", "win"]
    ]
    prepared_rows = []
    for _, row in dataframe.iterrows():
        my_pick_index = row.team_picks.index(row.picked_hero)
        visible_team_picks = row.team_picks[:my_pick_index]
        visible_opp_picks = row.opponent_picks[
            : min(visibility_map[my_pick_index], len(row.opponent_picks) - 1)
        ]
        prepared_rows.append(
            {
                "visible_team_picks": visible_team_picks,
                "visible_opp_picks": visible_opp_picks,
                "actual_pick": row.picked_hero,
                "win": row.win,
                "is_my_decision": True,
            }
        )

    return pd.DataFrame(prepared_rows)
