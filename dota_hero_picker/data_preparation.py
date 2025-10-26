import logging

import pandas as pd

from dota_hero_picker.hero_data_manager import HeroDataManager

from .neural_network import SEQ_LEN

logger = logging.getLogger(__name__)

MAX_PICK = 5


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
            is_my_decision = int(my_pick == pick)
            padded_draft_sequence = draft_sequence[:index] + [0] * (
                SEQ_LEN - index
            )

            results.append(
                {
                    "draft_sequence": padded_draft_sequence,
                    "win": win,
                    "is_my_decision": is_my_decision,
                },
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
            results.append(
                {
                    "draft_sequence": padded_draft_sequence,
                    "win": win,
                    "is_my_decision": 0,
                },
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
        prepared_rows.append(
            {
                "draft_sequence": padded_draft_sequence,
                "win": row.win,
                "is_my_decision": 1,
            },
        )

    return pd.DataFrame(prepared_rows)


def enrich_dataframe(dataframe: pd.DataFrame, hero_data_manager: HeroDataManager) -> pd.DataFrame:
    results = []
    for _, row in dataframe.iterrows():
        hero_features = [
            hero_data_manager.get_hero_features(hero_id)
            for hero_id in row.draft_sequence
        ]
        results.append(
            {
                "draft_sequence": row.draft_sequence,
                "hero_features": hero_features,
                "win": row.win,
                "is_my_decision": row.is_my_decision,
            },
        )
    return pd.DataFrame(results)
