from pathlib import Path

import torch

import settings
import streamlit as st
from dota_hero_picker.data_preparation import (
    hero_name_2_model_id,
)
from dota_hero_picker.load_personal_matches import get_hero_data
from dota_hero_picker.neural_network import WinPredictorWithPositionalAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hero_data = get_hero_data()
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
hero_to_id = {hero: idx for idx, hero in enumerate(heroes)}
id_to_hero = {idx: hero for hero, idx in hero_to_id.items()}
max_picks = 5


def pad_hero_ids(hero_names: list[str], max_picks: int = 5) -> list[int]:
    ids = [hero_to_id.get(name, 0) for name in hero_names]  # 0 if invalid
    padded = ids + [0] * (max_picks - len(ids))
    return padded[:max_picks]


def suggest_best_picks(
    model: WinPredictorWithPositionalAttention,
    team_picks: list[str],
    opponent_picks: list[str],
    allowed_positions: list[int],
    top_n: int = 20,
) -> list[tuple[str, float]]:
    model.eval()
    suggestions = []

    team_ids = pad_hero_ids(team_picks)
    opp_ids = pad_hero_ids(opponent_picks)
    team_tensor = torch.tensor([team_ids], dtype=torch.long, device=device)
    opp_tensor = torch.tensor([opp_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        baseline_tensor = torch.tensor([0], dtype=torch.long, device=device)
        baseline_output = model(team_tensor, opp_tensor, baseline_tensor)
        baseline_prob = torch.sigmoid(baseline_output).item()

        for candidate_hero in heroes:
            if (
                candidate_hero in team_picks
                or candidate_hero in opponent_picks
            ):
                continue
            hero_pos = hero_positions.get(candidate_hero, [1, 2, 3, 4, 5])
            if not set(hero_pos) & set(allowed_positions):
                continue

            candidate_id = hero_name_2_model_id[candidate_hero]

            actual_pick_tensor = torch.tensor(
                [candidate_id],
                dtype=torch.long,
                device=device,
            )
            output = model(team_tensor, opp_tensor, actual_pick_tensor)
            prob = torch.sigmoid(output).item()
            delta = prob - baseline_prob

            suggestions.append((candidate_hero, delta))

    suggestions.sort(key=lambda x: x[1], reverse=True)
    return suggestions[:top_n]


st.title("Dota Picker Web UI")

model_path = settings.MODELS_FOLDER_PATH / Path("stable_model.pth")

if model_path.exists():
    model = WinPredictorWithPositionalAttention(
        num_heroes,
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    st.success(f"Model loaded from {model_path}.")
else:
    st.error(
        f"Model file '{model_path}' not found. "
        "Please train the model first using train_model.py.",
    )
    st.stop()  # Halt the app if no model

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
team_picks = st.multiselect(
    "Your Team Picks (up to 5)",
    options=team_options,
    max_selections=5,
    default=st.session_state.team_picks,
    key="team_picks_widget",
    on_change=on_team_change,
)

opponent_picks = st.multiselect(
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
    try:
        # Use all positions if none selected
        allowed = (
            selected_positions if selected_positions else position_options
        )
        suggestions = suggest_best_picks(
            model,
            team_picks,
            opponent_picks,
            allowed,
        )
        st.subheader("Top Suggested Picks")
        for idx, (hero, prob) in enumerate(suggestions, 1):
            st.write(f"#{idx} {hero} (Win Prob: {prob * 100:.2f}%)")
    except ValueError as e:
        st.error(f"Error: {e}")
