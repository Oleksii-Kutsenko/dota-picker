from pathlib import Path

import numpy as np
import streamlit as st
import torch

import settings
from dota_hero_picker.data_preparation import (
    MAX_PICK,
)
from dota_hero_picker.hero_data_manager import HeroDataManager, hero_positions
from dota_hero_picker.model_trainer import ModelTrainer
from dota_hero_picker.neural_network import RNNWinPredictor
from dota_hero_picker.patch_resolver import get_latest_patch_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_model() -> RNNWinPredictor:
    model_path = settings.MODELS_FOLDER_PATH / Path("stable_model.pth")
    st.success(f"Model loaded from {model_path}.")

    model = ModelTrainer.create_default_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


@st.cache_resource
def get_hero_data_manager() -> HeroDataManager:
    return HeroDataManager()


hero_data_manager = get_hero_data_manager()
loaded_model = get_model()
latest_patch_id = get_latest_patch_id()


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


def suggest_best_picks(
    model: RNNWinPredictor,
    team_picks: list[str],
    opponent_picks: list[str],
    allowed_positions: list[int],
    top_n: int = 20,
) -> list[tuple[str, float]]:
    model.eval()

    candidate_heroes = []
    already_picked = set(team_picks) | set(opponent_picks)
    all_heroes = hero_data_manager.get_heroes_localized_names()

    for localized_name in all_heroes:
        if localized_name in already_picked:
            continue

        hero_pos = hero_positions.get(localized_name, [1, 2, 3, 4, 5])
        if not any(pos in allowed_positions for pos in hero_pos):
            continue

        candidate_heroes.append(localized_name)

    batch_draft_ids = [
        build_draft_sequence(team_picks, opponent_picks, candidate)
        for candidate in candidate_heroes
    ]

    batch_hero_features = [
        [
            hero_data_manager.get_hero_features(draft_id)
            for draft_id in draft_ids
        ]
        for draft_ids in batch_draft_ids
    ]

    draft_tensor = torch.tensor(
        batch_draft_ids,
        dtype=torch.long,
        device=device,
    )
    hero_features_tensor = torch.tensor(
        np.array(batch_hero_features),
        dtype=torch.float,
        device=device,
    )
    patch_ids_tensor = torch.full(
        (len(batch_draft_ids),),
        fill_value=latest_patch_id,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits = model(
            draft_tensor,
            hero_features_tensor,
            patch_ids_tensor,
        )
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

    results = list(zip(candidate_heroes, probabilities, strict=False))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def calculate_baseline_probability(
    model: RNNWinPredictor,
    team_picks: list[str],
    opponent_picks: list[str],
) -> float:
    baseline_ids = build_draft_sequence(
        team_picks,
        opponent_picks,
    )
    hero_features = [
        hero_data_manager.get_hero_features(draft_id)
        for draft_id in baseline_ids
    ]

    baseline_tensor = torch.tensor(
        [baseline_ids],
        dtype=torch.long,
        device=device,
    )
    hero_features_tensor = torch.tensor(
        np.array([hero_features]),
        dtype=torch.float,
        device=device,
    )
    patch_tensor = torch.tensor(
        [latest_patch_id],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        baseline_logits = model(
            baseline_tensor,
            hero_features_tensor,
            patch_tensor,
        )
        return torch.sigmoid(baseline_logits).item()


st.title("Dota Picker Web UI")
st.header("Draft Setup")

if "team_picks" not in st.session_state:
    st.session_state.team_picks = []
if "opponent_picks" not in st.session_state:
    st.session_state.opponent_picks = []


def on_team_change() -> None:
    st.session_state.team_picks = st.session_state.team_picks_widget
    st.session_state.opponent_picks = [
        h
        for h in st.session_state.opponent_picks
        if h not in st.session_state.team_picks
    ]


def on_opponent_change() -> None:
    st.session_state.opponent_picks = st.session_state.opponent_picks_widget
    st.session_state.team_picks = [
        h
        for h in st.session_state.team_picks
        if h not in st.session_state.opponent_picks
    ]


all_hero_names = hero_data_manager.get_heroes_localized_names()

team_options = [
    h for h in all_hero_names if h not in st.session_state.opponent_picks
]
opponent_options = [
    h for h in all_hero_names if h not in st.session_state.team_picks
]

team_picks_multiselect = st.multiselect(
    "Your Team Picks (up to 5)",
    options=team_options,
    max_selections=5,
    default=st.session_state.team_picks,
    key="team_picks_widget",
    on_change=on_team_change,
)

opponent_picks_multiselect = st.multiselect(
    "Opponent Team Picks (up to 5)",
    options=opponent_options,
    max_selections=5,
    default=st.session_state.opponent_picks,
    key="opponent_picks_widget",
    on_change=on_opponent_change,
)

st.sidebar.header("Filter Suggestions by Position")
position_options = [1, 2, 3, 4, 5]
position_to_name = {
    1: "Carry",
    2: "Mid",
    3: "Offlane",
    4: "Roaming Support",
    5: "Hard Support",
}
selected_positions: list[int] = st.sidebar.multiselect(
    "Allowed Positions (select none for all)",
    options=position_options,
    format_func=lambda p: f"{p} ({position_to_name[p]})",
    default=position_options,
)


if st.button("Get Suggestions"):
    allowed = selected_positions if selected_positions else position_options
    suggestions = suggest_best_picks(
        loaded_model,
        team_picks_multiselect,
        opponent_picks_multiselect,
        allowed,
    )
    st.subheader("Top Suggested Picks (as next team pick)")

    baseline_prob = calculate_baseline_probability(
        loaded_model,
        team_picks_multiselect,
        opponent_picks_multiselect,
    )

    st.metric(
        "Baseline Win Prob (no new pick)",
        f"{baseline_prob * 100:.2f}%",
    )
    st.divider()

    top_hero, top_prob = suggestions[0]
    st.metric(
        label=f"#1 Suggestion: {top_hero}",
        value=f"{top_prob * 100:.2f}%",
        delta=f"{(top_prob - baseline_prob) * 100:.2f}%",
        delta_color="normal",
    )

    for idx, (hero, probability) in enumerate(suggestions[1:], 2):
        st.write(f"#{idx} {hero} (Win Prob: {probability * 100:.2f}%)")
