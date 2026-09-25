from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch

import settings
from dota_hero_picker.data_preparation import DRAFT_STAGES, MAX_PICK
from dota_hero_picker.hero_data_manager import HeroDataManager
from dota_hero_picker.neural_network import (
    MatchWinPredictor,
    ModelParameters,
)
from dota_hero_picker.patch_resolver import get_latest_patch_id

st.set_page_config(
    page_title="Dota 2 Draft Assistant",
    page_icon="⚔️",
    layout="wide",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Cached Resources & Metadata ---


@st.cache_resource
def get_hero_data_manager() -> HeroDataManager:
    return HeroDataManager()


@st.cache_resource
def get_model() -> tuple[torch.nn.Module, float]:
    hdm = get_hero_data_manager()
    model_path = settings.MODELS_FOLDER_PATH / Path("stable_model.pth")
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Train a model first.")
        st.stop()

    checkpoint = torch.load(model_path, map_location=device)
    temp = float(checkpoint.get("temperature", 1.0))

    model_params = ModelParameters.from_dict(checkpoint["model_params"])
    draft_model = MatchWinPredictor(
        model_params,
        hdm.get_hero_features_matrix(),
    )
    draft_model.load_state_dict(checkpoint["model_state"], strict=False)
    draft_model.to(device)
    draft_model.eval()
    return draft_model, temp


@st.cache_data
def get_hero_metadata_map() -> dict[str, tuple[str, str]]:
    hdm = get_hero_data_manager()
    df = hdm.processed_heroes
    flags = [
        ("has_bkb_pierce", "BKB Pierce"),
        ("has_break", "Break"),
        ("has_stun", "Stun"),
        ("has_silence", "Silence"),
        ("has_dispel", "Dispel"),
        ("has_pure_damage", "Pure Dmg"),
        ("has_root", "Root"),
        ("is_illusion", "Illusion"),
        ("has_invis", "Invis"),
        ("has_heal", "Heal"),
    ]

    meta_map = {}
    for _, rec in df.iterrows():
        name = rec.get("localized_name")
        if not name:
            continue

        if rec.get("primary_attr_str", 0) == 1:
            attr = "Strength"
        elif rec.get("primary_attr_agi", 0) == 1:
            attr = "Agility"
        elif rec.get("primary_attr_int", 0) == 1:
            attr = "Intelligence"
        else:
            attr = "Universal"

        mechanics = [label for key, label in flags if rec.get(key, 0) == 1]
        meta_map[name] = (attr, ", ".join(mechanics))

    return meta_map


hero_data_manager = get_hero_data_manager()
model, temperature = get_model()
hero_meta_map = get_hero_metadata_map()
latest_patch_id = get_latest_patch_id()

TEAM_PICK_SLOTS = (0, 1, 4, 5, 8)

# --- Inference Helpers ---


def build_draft_sequence(
    team_picks: list[str],
    opponent_picks: list[str],
    candidate: str | None = None,
) -> list[int]:
    local_team_picks = team_picks.copy()
    if candidate:
        local_team_picks.append(candidate)

    team_ids = [
        hero_data_manager.get_hero_id_by_localized_name(h)
        for h in local_team_picks
    ]
    opp_ids = [
        hero_data_manager.get_hero_id_by_localized_name(h)
        for h in opponent_picks
    ]

    team_ids += [0] * (MAX_PICK - len(team_ids))
    opp_ids += [0] * (4 - len(opp_ids))

    return (
        team_ids[:2]
        + opp_ids[:2]
        + team_ids[2:4]
        + opp_ids[2:4]
        + team_ids[4:]
    )


def calculate_baseline_winrate(
    team_picks: list[str],
    opponent_picks: list[str],
    is_radiant: bool,  # noqa: FBT001
) -> float:
    if not team_picks and not opponent_picks:
        return 0.50

    baseline_ids = build_draft_sequence(team_picks, opponent_picks)
    baseline_tensor = torch.tensor(
        [baseline_ids],
        dtype=torch.long,
        device=device,
    )
    total_picks = len(team_picks) + len(opponent_picks)
    draft_stage = DRAFT_STAGES[min(total_picks - 1, len(DRAFT_STAGES) - 1)]
    match_context = torch.tensor(
        [[latest_patch_id, draft_stage, 1 if is_radiant else 0]],
        dtype=torch.long,
        device=device,
    )
    my_hero_slot = torch.tensor([-1], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(
            baseline_tensor,
            match_context,
            my_hero_slot,
        )
        return float(torch.sigmoid(logits / temperature).item())


def get_available_candidate_heroes(
    team_picks: list[str],
    opponent_picks: list[str],
) -> list[str]:
    """Return all heroes that have not been drafted yet."""
    already_picked_heroes = set(team_picks) | set(opponent_picks)
    return [
        hero_name
        for hero_name in hero_data_manager.get_heroes_localized_names()
        if hero_name not in already_picked_heroes
    ]


def predict_candidate_winrates(
    candidates: list[str],
    team_picks: list[str],
    opponent_picks: list[str],
    is_radiant: bool,  # noqa: FBT001
) -> np.ndarray:
    """Run batched model inference for all candidate heroes."""
    draft_sequences = [
        build_draft_sequence(team_picks, opponent_picks, candidate_hero)
        for candidate_hero in candidates
    ]
    batch_size = len(candidates)
    draft_tensor = torch.tensor(
        draft_sequences,
        dtype=torch.long,
        device=device,
    )
    candidate_slot = TEAM_PICK_SLOTS[min(len(team_picks), 4)]
    draft_stage = DRAFT_STAGES[candidate_slot]
    match_context = torch.tensor(
        [[latest_patch_id, draft_stage, 1 if is_radiant else 0]],
        dtype=torch.long,
        device=device,
    ).repeat(batch_size, 1)
    my_hero_slots = torch.full(
        (batch_size,),
        fill_value=candidate_slot,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits = model(
            draft_tensor,
            match_context,
            my_hero_slots,
        )
        return torch.sigmoid(logits / temperature).cpu().numpy().flatten()


def get_recommendations(
    team_picks: list[str],
    opponent_picks: list[str],
    baseline_winrate: float,
    is_radiant: bool,  # noqa: FBT001
) -> list[dict[str, Any]]:
    candidates = get_available_candidate_heroes(team_picks, opponent_picks)
    if not candidates:
        return []

    predicted_winrates = predict_candidate_winrates(
        candidates,
        team_picks,
        opponent_picks,
        is_radiant,
    )

    recommendations: list[dict[str, Any]] = []
    for hero_name, winrate in zip(
        candidates,
        predicted_winrates,
        strict=False,
    ):
        primary_attribute, key_mechanics = hero_meta_map.get(
            hero_name,
            ("Universal", ""),
        )
        recommendations.append(
            {
                "Hero": hero_name,
                "Attribute": primary_attribute,
                "Win Rate": float(winrate),
                "Impact": float(winrate - baseline_winrate),
                "Mechanics": key_mechanics,
            },
        )

    recommendations.sort(key=lambda item: item["Win Rate"], reverse=True)
    return recommendations


# --- UI Components ---


def render_side_selector() -> bool:
    """Render map side choice and return True if playing Radiant."""
    side_choice = st.radio(
        "You are playing as",
        options=["🟢 Radiant", "🔴 Dire"],
        index=0,
        horizontal=True,
        help="Select which side of the map your team is playing on.",
        key="team_side_radio",
    )
    return side_choice == "🟢 Radiant"


def render_draft_pickers(
    available_heroes: list[str],
    is_radiant: bool,  # noqa: FBT001
) -> tuple[list[str], list[str]]:
    """Render draft multiselects and resolve (team_picks, opponent_picks)."""
    column_radiant, column_dire = st.columns(2)

    with column_radiant:
        radiant_max = 5 if is_radiant else 4
        radiant_role = "Your Team" if is_radiant else "Opponents"
        radiant_picks = st.multiselect(
            f"🟢 Radiant ({radiant_role}, max {radiant_max})",
            options=available_heroes,
            max_selections=radiant_max,
            placeholder="Select Radiant heroes...",
            key="radiant_picks_multiselect",
        )

    with column_dire:
        dire_max = 4 if is_radiant else 5
        dire_role = "Opponents" if is_radiant else "Your Team"
        dire_options = [
            hero for hero in available_heroes if hero not in radiant_picks
        ]
        dire_picks = st.multiselect(
            f"🔴 Dire ({dire_role}, max {dire_max})",
            options=dire_options,
            max_selections=dire_max,
            placeholder="Select Dire heroes...",
            key="dire_picks_multiselect",
        )

    team_picks = radiant_picks if is_radiant else dire_picks
    opponent_picks = dire_picks if is_radiant else radiant_picks
    return team_picks, opponent_picks


def render_win_probability_metric(
    baseline_winrate: float,
    has_draft_picks: bool,  # noqa: FBT001
) -> None:
    """Render the team's current win rate percentage and delta."""
    delta_percentage = (baseline_winrate - 0.50) * 100
    st.metric(
        label="Your Team Win Probability",
        value=f"{baseline_winrate * 100:.1f}%",
        delta=f"{delta_percentage:+.1f}%" if has_draft_picks else None,
    )


def render_recommendations_section(
    team_picks: list[str],
    opponent_picks: list[str],
    baseline_winrate: float,
    is_radiant: bool,  # noqa: FBT001
) -> None:
    """Render the hero search bar and ranked recommendations table."""
    if len(team_picks) >= MAX_PICK:
        st.info("Your team is full (5/5 heroes picked).")
        return

    recommendations = get_recommendations(
        team_picks,
        opponent_picks,
        baseline_winrate,
        is_radiant,
    )

    search_query = st.text_input(
        "Search Hero",
        placeholder="Filter by hero name...",
        label_visibility="collapsed",
    )
    if search_query.strip():
        search_term = search_query.strip().lower()
        recommendations = [
            item
            for item in recommendations
            if search_term in item["Hero"].lower()
        ]

    if not recommendations:
        st.warning("No heroes found.")
        return

    table_rows = [
        {
            "Rank": rank_index,
            "Hero": item["Hero"],
            "Win Rate": f"{item['Win Rate'] * 100:.1f}%",
            "Win Impact": f"{item['Impact'] * 100:+.2f}%",
            "Attribute": item["Attribute"],
            "Key Mechanics": item["Mechanics"] if item["Mechanics"] else "—",
        }
        for rank_index, item in enumerate(recommendations, start=1)
    ]
    st.dataframe(
        pd.DataFrame(table_rows),
        width="stretch",
        hide_index=True,
        height=650,
    )


# --- Main App ---


def main() -> None:
    all_heroes = sorted(hero_data_manager.get_heroes_localized_names())

    header_left, header_right = st.columns([5, 3])
    with header_left:
        is_radiant = render_side_selector()

    team_picks, opponent_picks = render_draft_pickers(all_heroes, is_radiant)

    baseline_winrate = calculate_baseline_winrate(
        team_picks,
        opponent_picks,
        is_radiant,
    )
    with header_right:
        has_draft_picks = bool(team_picks or opponent_picks)
        render_win_probability_metric(baseline_winrate, has_draft_picks)

    st.divider()

    render_recommendations_section(
        team_picks,
        opponent_picks,
        baseline_winrate,
        is_radiant,
    )


if __name__ == "__main__":
    main()
