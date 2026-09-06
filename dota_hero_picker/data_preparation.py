import logging

import numpy as np
import pandas as pd

from dota_hero_picker.hero_data_manager import HeroDataManager

logger = logging.getLogger(__name__)

MAX_PICK = 5
SLOT_COLUMNS = [
    "team_pick_1",
    "team_pick_2",
    "opp_pick_1",
    "opp_pick_2",
    "team_pick_3",
    "team_pick_4",
    "opp_pick_3",
    "opp_pick_4",
    "team_pick_5",
]


def create_augmented_dataframe(train_dataframe: pd.DataFrame) -> pd.DataFrame:
    augmented_records = []

    for _, row in train_dataframe.iterrows():
        # Perspective 1: Team as Ally
        ally_perspective_picks = [
            row["team_pick_1"],
            row["team_pick_2"],
            row["opp_pick_1"],
            row["opp_pick_2"],
            row["team_pick_3"],
            row["team_pick_4"],
            row["opp_pick_3"],
            row["opp_pick_4"],
            row["team_pick_5"],
        ]
        my_picked_hero = row["picked_hero"]

        for draft_step in range(1, len(SLOT_COLUMNS) + 1):
            prefix_record = {
                slot_name: (
                    ally_perspective_picks[slot_index]
                    if slot_index < draft_step
                    else None
                )
                for slot_index, slot_name in enumerate(SLOT_COLUMNS)
            }
            current_hero_pick = ally_perspective_picks[draft_step - 1]
            prefix_record.update(
                {
                    "win": row["win"],
                    "is_my_decision": int(my_picked_hero == current_hero_pick),
                    "patch_id": row["patch_id"],
                },
            )
            augmented_records.append(prefix_record)

        opponent_perspective_picks = [
            row["opp_pick_1"],
            row["opp_pick_2"],
            row["team_pick_1"],
            row["team_pick_2"],
            row["opp_pick_3"],
            row["opp_pick_4"],
            row["team_pick_3"],
            row["team_pick_4"],
            row["opp_pick_5"],
        ]
        for draft_step in range(1, len(SLOT_COLUMNS) + 1):
            prefix_record = {
                slot_name: (
                    opponent_perspective_picks[slot_index]
                    if slot_index < draft_step
                    else None
                )
                for slot_index, slot_name in enumerate(SLOT_COLUMNS)
            }
            prefix_record.update(
                {
                    "win": 1 - row["win"],
                    "is_my_decision": 0,
                    "patch_id": row["patch_id"],
                },
            )
            augmented_records.append(prefix_record)

    return pd.DataFrame(augmented_records)


def prepare_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    prepared_records = []

    for _, row in dataframe.iterrows():
        chronological_picks = [
            row["team_pick_1"],
            row["team_pick_2"],
            row["opp_pick_1"],
            row["opp_pick_2"],
            row["team_pick_3"],
            row["team_pick_4"],
            row["opp_pick_3"],
            row["opp_pick_4"],
            row["team_pick_5"],
        ]
        my_pick_step = chronological_picks.index(row["picked_hero"]) + 1

        prefix_record = {
            slot_name: (
                chronological_picks[slot_index]
                if slot_index < my_pick_step
                else None
            )
            for slot_index, slot_name in enumerate(SLOT_COLUMNS)
        }
        prefix_record.update(
            {
                "win": row["win"],
                "is_my_decision": 1,
                "patch_id": row["patch_id"],
            },
        )
        prepared_records.append(prefix_record)

    return pd.DataFrame(prepared_records)


def enrich_dataframe(
    dataframe: pd.DataFrame,
    hero_data_manager: HeroDataManager,
) -> pd.DataFrame:
    def get_slot_features(row: pd.Series) -> np.ndarray:
        return np.array(
            [
                hero_data_manager.get_hero_features(
                    int(hero_id) if pd.notna(hero_id) else 0,
                )
                for hero_id in row[SLOT_COLUMNS]
            ],
        )

    dataframe["hero_features"] = dataframe.apply(
        get_slot_features,
        axis=1,
    )
    return dataframe
