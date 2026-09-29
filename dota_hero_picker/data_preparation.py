import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_PICK = 5
TOTAL_DRAFT_SLOTS = 9

ALLY_PERSPECTIVE_COLUMNS = [
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
SLOT_COLUMNS = ALLY_PERSPECTIVE_COLUMNS

OPPONENT_PERSPECTIVE_COLUMNS = [
    "opp_pick_1",
    "opp_pick_2",
    "team_pick_1",
    "team_pick_2",
    "opp_pick_3",
    "opp_pick_4",
    "team_pick_3",
    "team_pick_4",
    "opp_pick_5",
]

DRAFT_STAGES = (1, 1, 1, 1, 2, 2, 2, 2, 3)


def build_perspective_dataframe(  # pylint: disable=too-many-locals
    chronological_picks: np.ndarray,
    match_wins: np.ndarray,
    patch_ids: np.ndarray,
    is_radiant_values: np.ndarray,
    player_picked_heroes: np.ndarray,
) -> pd.DataFrame:
    """Generate vectorized draft prefixes and metadata DataFrame."""
    match_count = len(chronological_picks)

    # 1. Expand matches 9x: one row for each draft step (0 to 8)
    repeated_picks = np.repeat(
        chronological_picks,
        TOTAL_DRAFT_SLOTS,
        axis=0,
    )
    steps = np.tile(np.arange(TOTAL_DRAFT_SLOTS), match_count)

    # 2. Mask future slots (slots after current step are 0)
    slot_indices = np.arange(TOTAL_DRAFT_SLOTS)
    draft_slots = np.where(
        slot_indices <= steps[:, None],
        repeated_picks,
        0,
    )

    # 3. Locate player hero slot (-1 if not in this team's draft)
    hero_in_slot = chronological_picks == player_picked_heroes[:, None]
    has_picked = hero_in_slot.any(axis=1) & (player_picked_heroes != 0)
    player_slots = np.where(
        has_picked,
        np.argmax(hero_in_slot, axis=1),
        -1,
    )

    repeated_player_slots = np.repeat(player_slots, TOTAL_DRAFT_SLOTS)
    is_my_decision = (steps == repeated_player_slots).astype(np.int64)
    my_hero_slot = np.where(
        steps >= repeated_player_slots,
        repeated_player_slots,
        -1,
    )

    data: dict[str, np.ndarray] = {
        col: draft_slots[:, i]
        for i, col in enumerate(ALLY_PERSPECTIVE_COLUMNS)
    }
    data["win"] = np.repeat(match_wins, TOTAL_DRAFT_SLOTS)
    data["is_my_decision"] = is_my_decision
    data["patch_id"] = np.repeat(patch_ids, TOTAL_DRAFT_SLOTS)
    data["is_radiant"] = np.repeat(is_radiant_values, TOTAL_DRAFT_SLOTS)
    data["draft_stage"] = np.tile(
        np.array(DRAFT_STAGES, dtype=np.int64),
        match_count,
    )
    data["my_hero_slot"] = my_hero_slot

    return pd.DataFrame(data)


def create_augmented_dataframe(train_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Generate 18x augmented dataset (9 steps for ally & opponent views)."""
    ally_picks = train_dataframe[ALLY_PERSPECTIVE_COLUMNS].to_numpy(
        dtype=np.int64,
    )
    opp_picks = train_dataframe[OPPONENT_PERSPECTIVE_COLUMNS].to_numpy(
        dtype=np.int64,
    )
    wins = train_dataframe["win"].to_numpy(dtype=np.int64)
    patch_ids = train_dataframe["patch_id"].to_numpy(dtype=np.int64)
    is_radiant = train_dataframe["is_radiant"].to_numpy(dtype=np.int64)
    player_heroes = train_dataframe["picked_hero"].to_numpy(dtype=np.int64)

    ally_df = build_perspective_dataframe(
        chronological_picks=ally_picks,
        match_wins=wins,
        patch_ids=patch_ids,
        is_radiant_values=is_radiant,
        player_picked_heroes=player_heroes,
    )
    opp_df = build_perspective_dataframe(
        chronological_picks=opp_picks,
        match_wins=1 - wins,
        patch_ids=patch_ids,
        is_radiant_values=1 - is_radiant,
        player_picked_heroes=np.zeros_like(player_heroes),
    )

    return pd.concat([ally_df, opp_df], ignore_index=True)


def prepare_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepare un-augmented test dataset evaluated at player's pick moment."""
    picks = dataframe[ALLY_PERSPECTIVE_COLUMNS].to_numpy(dtype=np.int64)
    player_heroes = dataframe["picked_hero"].to_numpy(dtype=np.int64)

    # Locate player pick slot
    hero_in_slot = picks == player_heroes[:, None]
    player_slots = np.argmax(hero_in_slot, axis=1)

    # Mask draft slots up to player pick
    slot_indices = np.arange(TOTAL_DRAFT_SLOTS)
    draft_slots = np.where(slot_indices <= player_slots[:, None], picks, 0)
    draft_stages = np.array(DRAFT_STAGES, dtype=np.int64)[player_slots]

    data: dict[str, np.ndarray] = {
        col: draft_slots[:, i]
        for i, col in enumerate(ALLY_PERSPECTIVE_COLUMNS)
    }
    data["win"] = dataframe["win"].to_numpy(dtype=np.int64)
    data["is_my_decision"] = np.ones(len(dataframe), dtype=np.int64)
    data["patch_id"] = dataframe["patch_id"].to_numpy(dtype=np.int64)
    data["is_radiant"] = dataframe["is_radiant"].to_numpy(dtype=np.int64)
    data["draft_stage"] = draft_stages
    data["my_hero_slot"] = player_slots

    return pd.DataFrame(data)
