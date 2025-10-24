from pathlib import Path

import streamlit as st
import torch

import settings
from dota_hero_picker.data_preparation import (
    MAX_PICK,
)
from dota_hero_picker.hero_data_manager import HeroDataManager
from dota_hero_picker.model_trainer import ModelTrainer
from dota_hero_picker.neural_network import RNNWinPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hero_data = HeroDataManager().raw_hero_data
hero_positions = {
    "Troll Warlord": [1, 2],
    "Ogre Magi": [4, 5],
    "Keeper Of The Light": [4, 5],
    "Dark Willow": [4, 5],
    "Weaver": [1, 2],
    "Muerta": [1, 2, 3],
    "Tiny": [2],
    "Timbersaw": [2, 3],
    "Bristleback": [3],
    "Razor": [1, 2],
    "Magnus": [2, 3],
    "Slardar": [3, 4],
    "Snapfire": [4, 5],
    "Templar Assassin": [2],
    "Viper": [2, 3],
    "Monkey King": [1, 2],
    "Tusk": [4, 5],
    "Enigma": [3, 4],
    "Lifestealer": [1],
    "Enchantress": [3, 4, 5],
    "Invoker": [2],
    "Lina": [2, 4, 5],
    "Naga Siren": [1],
    "Leshrac": [2, 4, 5],
    "Morphling": [1, 2],
    "Huskar": [1, 2],
    "Spectre": [1],
    "Winter Wyvern": [4, 5],
    "Phantom Lancer": [1],
    "Sven": [1, 3],
    "Zeus": [2],
    "Necrophos": [2, 3],
    "Warlock": [4, 5],
    "Mirana": [2, 4, 5],
    "Luna": [1, 2],
    "Venomancer": [3, 4, 5],
    "Abaddon": [4, 5],
    "Alchemist": [1, 3],
    "Ancient Apparition": [4, 5],
    "Anti-Mage": [1],
    "Arc Warden": [2],
    "Axe": [3],
    "Bane": [4, 5],
    "Batrider": [3],
    "Beastmaster": [3],
    "Bloodseeker": [2, 3],
    "Centaur Warrunner": [3],
    "Chaos Knight": [1, 3],
    "Clinkz": [1],
    "Crystal Maiden": [4, 5],
    "Dawnbreaker": [3, 4],
    "Dazzle": [4, 5],
    "Death Prophet": [2],
    "Disruptor": [4, 5],
    "Drow Ranger": [1, 2],
    "Earth Spirit": [4, 5],
    "Earthshaker": [2],
    "Keeper of the Light": [4, 5],
    "Bounty Hunter": [4, 5],
    "Elder Titan": [3, 4],
    "Ember Spirit": [2, 3],
    "Faceless Void": [1],
    "Grimstroke": [4, 5],
    "Gyrocopter": [1],
    "Hoodwink": [4, 5],
    "Jakiro": [4, 5],
    "Juggernaut": [1],
    "Kunkka": [3, 4],
    "Legion Commander": [2, 3],
    "Lich": [4, 5],
    "Lion": [4, 5],
    "Marci": [4, 5],
    "Mars": [3],
    "Medusa": [1],
    "Nature's Prophet": [1, 2, 3, 4],
    "Nyx Assassin": [3, 4],
    "Outworld Destroyer": [2],
    "Pangolier": [3],
    "Phantom Assassin": [1],
    "Puck": [2, 3],
    "Pudge": [2, 3, 4],
    "Pugna": [2, 4, 5],
    "Queen of Pain": [2],
    "Ringmaster": [4, 5],
    "Rubick": [4, 5],
    "Sand King": [2],
    "Shadow Fiend": [2],
    "Shadow Shaman": [4, 5],
    "Silencer": [2],
    "Skywrath Mage": [2],
    "Sniper": [1, 2],
    "Spirit Breaker": [3],
    "Storm Spirit": [2],
    "Techies": [4, 5],
    "Terrorblade": [1],
    "Tidehunter": [3],
    "Ursa": [1, 3],
    "Vengeful Spirit": [4, 5],
    "Windranger": [2, 3],
    "Witch Doctor": [4, 5],
    "Wraith King": [1],
    "Brewmaster": [3],
    "Broodmother": [2, 3],
    "Chen": [4, 5],
    "Clockwerk": [3, 4],
    "Dark Seer": [3],
    "Doom": [3],
    "Dragon Knight": [2, 3],
    "Io": [2, 4, 5],
    "Lone Druid": [1, 3],
    "Lycan": [1, 3],
    "Meepo": [2],
    "Night Stalker": [3, 4],
    "Omniknight": [3, 4, 5],
    "Oracle": [4, 5],
    "Phoenix": [3, 4, 5],
    "Primal Beast": [3],
    "Riki": [1, 4],
    "Shadow Demon": [4, 5],
    "Slark": [1],
    "Tinker": [2],
    "Treant Protector": [3, 4, 5],
    "Underlord": [3],
    "Undying": [3, 4, 5],
    "Visage": [2, 3],
    "Void Spirit": [2, 3],
}

heroes = [hero_data_item["localized_name"] for hero_data_item in hero_data]
num_heroes = len(heroes)
hero_data_manager = HeroDataManager()


def build_draft_sequence(
    team_picks: list[str],
    opponent_picks: list[str],
    candidate: str | None = None,
) -> list[int]:
    team_picks = [h for h in team_picks if h in hero_data_manager.local_name_2_model_id]
    if candidate:
        team_picks.append(candidate)
    opponent_picks = [
        h for h in opponent_picks if h in hero_data_manager.local_name_2_model_id
    ]

    team_ids = [hero_data_manager.local_name_2_model_id[h] for h in team_picks]
    opp_ids = [hero_data_manager.local_name_2_model_id[h] for h in opponent_picks]

    team_ids += [0] * (MAX_PICK - len(team_ids))
    opp_ids += [0] * (4 - len(opp_ids))

    return team_ids[:2] + opp_ids[:2] + team_ids[2:4] + opp_ids[2:4] + team_ids[4:]


def evaluate_candidate(
    model: RNNWinPredictor,
    candidate_hero: str,
    team_picks: list[str],
    opponent_picks: list[str],
) -> tuple[str, float]:
    draft_ids = build_draft_sequence(
        team_picks,
        opponent_picks,
        candidate_hero,
    )
    hero_features = [
        hero_data_manager.get_hero_features(draft_id) for draft_id in draft_ids
    ]

    draft_tensor = torch.tensor([draft_ids], dtype=torch.long, device=device)
    hero_features_tensor = torch.tensor(
        [hero_features],
        dtype=torch.long,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        logits = model(draft_tensor, hero_features_tensor)
        prob = torch.sigmoid(logits).item()

    return (candidate_hero, prob)


def suggest_best_picks(
    model: RNNWinPredictor,
    team_picks: list[str],
    opponent_picks: list[str],
    allowed_positions: list[int],
    top_n: int = 20,
) -> list[tuple[str, float]]:
    model.eval()
    results = []
    for candidate_hero in heroes:
        if candidate_hero in team_picks or candidate_hero in opponent_picks:
            continue

        hero_pos = hero_positions.get(candidate_hero, [1, 2, 3, 4, 5])
        if not any(pos in allowed_positions for pos in hero_pos):
            continue

        results.append(
            evaluate_candidate(
                model,
                candidate_hero,
                team_picks,
                opponent_picks,
            ),
        )

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


st.title("Dota Picker Web UI")

model_path = settings.MODELS_FOLDER_PATH / Path("stable_model.pth")

if model_path.exists():
    loaded_model = ModelTrainer.create_model(
        len(HeroDataManager().raw_hero_data),
    )
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.to(device)
    loaded_model.eval()
    st.success(f"Model loaded from {model_path}.")
else:
    st.error(
        f"Model file '{model_path}' not found. "
        "Please train the model first using train_model.py.",
    )
    st.stop()

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


team_options = [h for h in heroes if h not in st.session_state.opponent_picks]
opponent_options = [h for h in heroes if h not in st.session_state.team_picks]
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

current_seq = build_draft_sequence(
    team_picks_multiselect,
    opponent_picks_multiselect,
)


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
        hero_data_manager.get_hero_features(draft_id) for draft_id in baseline_ids
    ]

    baseline_tensor = torch.tensor(
        [baseline_ids],
        dtype=torch.long,
        device=device,
    )
    hero_features_tensor = torch.tensor(
        [hero_features],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        baseline_logits = model(
            baseline_tensor,
            hero_features_tensor,
        )
        return torch.sigmoid(baseline_logits).item()


if st.button("Get Suggestions"):
    try:
        allowed = selected_positions if selected_positions else position_options
        suggestions = suggest_best_picks(
            loaded_model,
            team_picks_multiselect,
            opponent_picks_multiselect,
            allowed,
        )
        st.subheader("Top Suggested Picks (as next team pick)")
        for idx, (hero, probability) in enumerate(suggestions, 1):
            st.write(f"#{idx} {hero} (Win Prob: {probability * 100:.2f}%)")
        baseline_prob = calculate_baseline_probability(
            loaded_model,
            team_picks_multiselect,
            opponent_picks_multiselect,
        )
        st.metric(
            "Baseline Win Prob (no new pick)",
            f"{baseline_prob * 100:.2f}%",
        )
    except ValueError as e:
        st.error(f"Error: {e}")
