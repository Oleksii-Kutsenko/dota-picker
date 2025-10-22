import logging

import pandas as pd

from dota_hero_picker.hero_data_manager import HeroDataManager

from .load_personal_matches import get_hero_data
from .neural_network import SEQ_LEN

logger = logging.getLogger(__name__)

MAX_PICK = 5
hero_data = get_hero_data()
hero_names = []
hero_name_2_model_id = {}
model_id_2_hero_name = {}
model_id_2_hero_data = {}
api_id_2_model_id = {}
num_heroes = len(hero_data)
for model_id, hero in enumerate(hero_data, 1):
    hero_name = hero["localized_name"]
    api_hero_id = hero["id"]
    hero_names.append(hero_name)
    hero_name_2_model_id[hero_name] = model_id
    model_id_2_hero_name[model_id] = hero_name
    model_id_2_hero_data[model_id] = hero
    model_id_2_hero_data[model_id]["is_melee"] = (
        1 if model_id_2_hero_data[model_id]["attack_type"] == "Melee" else 0
    )
    api_id_2_model_id[api_hero_id] = model_id


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
        for index, pick in enumerate(draft_sequence, 1):
            is_my_decision = my_pick == pick
            padded_draft_sequence = draft_sequence[:index] + [0] * (
                SEQ_LEN - index
            )
            is_melee_sequence = [
                model_id_2_hero_data.get(hero_id, {"is_melee": -1})["is_melee"]
                for hero_id in padded_draft_sequence
            ]

            results.append(
                {
                    "draft_sequence": padded_draft_sequence,
                    "is_melee_sequence": is_melee_sequence,
                    "win": win,
                    "is_my_decision": is_my_decision,
                }
            )

        team_picks = row.opponent_picks
        opp_picks = row.team_picks
        win = 1 - row.win

        draft_sequence = (
            team_picks[:2]
            + opp_picks[:2]
            + team_picks[2:4]
            + opp_picks[2:4]
            + team_picks[4:]
        )
        for index, _ in enumerate(draft_sequence, 1):
            padded_draft_sequence = draft_sequence[:index] + [0] * (
                SEQ_LEN - index
            )
            is_melee_sequence = [
                model_id_2_hero_data.get(hero_id, {"is_melee": -1})["is_melee"]
                for hero_id in padded_draft_sequence
            ]
            results.append(
                {
                    "draft_sequence": padded_draft_sequence,
                    "is_melee_sequence": is_melee_sequence,
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
        team_picks = row.team_picks
        opp_picks = row.opponent_picks
        draft_sequence = (
            team_picks[:2]
            + opp_picks[:2]
            + team_picks[2:4]
            + opp_picks[2:4]
            + team_picks[4:]
        )
        my_pick_index = draft_sequence.index(row.picked_hero)
        padded_draft_sequence = draft_sequence[: my_pick_index + 1] + [0] * (
            SEQ_LEN - my_pick_index - 1
        )
        is_melee_sequence = [
            model_id_2_hero_data.get(hero_id, {"is_melee": -1})["is_melee"]
            for hero_id in padded_draft_sequence
        ]
        prepared_rows.append(
            {
                "draft_sequence": padded_draft_sequence,
                "is_melee_sequence": is_melee_sequence,
                "win": row.win,
                "is_my_decision": True,
            }
        )

    return pd.DataFrame(prepared_rows)
