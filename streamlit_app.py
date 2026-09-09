from pathlib import Path

import pandas as pd
import streamlit as st
import torch

import settings
from dota_hero_picker.data_preparation import MAX_PICK
from dota_hero_picker.hero_data_manager import HeroDataManager
from dota_hero_picker.neural_network import (
    SiameseDraftPredictor,
    SiameseParameters,
)
from dota_hero_picker.patch_resolver import get_latest_patch_id

st.set_page_config(
    page_title="Dota 2 Draft Assistant",
    page_icon="⚔️",
    layout="wide",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# Model & Data Loading
# =====================================================================
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
    temperature = float(checkpoint.get("temperature", 1.0))

    model_params = SiameseParameters(**checkpoint["model_params"])
    model = SiameseDraftPredictor(
        model_params,
        hdm.get_projected_hero_embeddings(model_params.d_model),
    )
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model, temperature


hero_data_manager = get_hero_data_manager()
model, temperature = get_model()
latest_patch_id = get_latest_patch_id()


# =====================================================================
# Helpers
# =====================================================================
def get_hero_meta(localized_name: str) -> tuple[str, str]:
    row = hero_data_manager.processed_heroes.loc[
        hero_data_manager.processed_heroes["localized_name"] == localized_name
    ]
    if row.empty:
        return "Universal", ""
    rec = row.iloc[0]

    # Attribute
    if rec.get("primary_attr_str", 0) == 1:
        attr = "Strength"
    elif rec.get("primary_attr_agi", 0) == 1:
        attr = "Agility"
    elif rec.get("primary_attr_int", 0) == 1:
        attr = "Intelligence"
    else:
        attr = "Universal"

    # Mechanics
    mechanics = []
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
    for key, label in flags:
        if rec.get(key, 0) == 1:
            mechanics.append(label)

    return attr, ", ".join(mechanics)


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
) -> float:
    if not team_picks and not opponent_picks:
        return 0.50

    baseline_ids = build_draft_sequence(team_picks, opponent_picks)
    baseline_tensor = torch.tensor([baseline_ids], dtype=torch.long, device=device)
    patch_tensor = torch.tensor([latest_patch_id], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(baseline_tensor, patch_tensor)
        return float(torch.sigmoid(logits / temperature).item())


def get_recommendations(
    team_picks: list[str],
    opponent_picks: list[str],
) -> list[dict]:
    already_picked = set(team_picks) | set(opponent_picks)
    all_heroes = hero_data_manager.get_heroes_localized_names()
    candidates = [h for h in all_heroes if h not in already_picked]

    if not candidates:
        return []

    batch_draft_ids = [
        build_draft_sequence(team_picks, opponent_picks, c)
        for c in candidates
    ]

    draft_tensor = torch.tensor(batch_draft_ids, dtype=torch.long, device=device)
    patch_tensor = torch.full(
        (len(batch_draft_ids),),
        fill_value=latest_patch_id,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits = model(draft_tensor, patch_tensor)
        probs = torch.sigmoid(logits / temperature).cpu().numpy().flatten()

    baseline = calculate_baseline_winrate(team_picks, opponent_picks)
    results = []

    for hero, prob in zip(candidates, probs, strict=False):
        attr, mechanics = get_hero_meta(hero)
        results.append({
            "Hero": hero,
            "Attribute": attr,
            "Win Rate": float(prob),
            "Impact": float(prob - baseline),
            "Mechanics": mechanics,
        })

    results.sort(key=lambda x: x["Win Rate"], reverse=True)
    return results


# =====================================================================
# UI - Top Compact Draft Setup
# =====================================================================
all_heroes = sorted(hero_data_manager.get_heroes_localized_names())

top_col_allies, top_col_enemies, top_col_stat = st.columns([3, 3, 2])

with top_col_allies:
    team_picks = st.multiselect(
        "Allies (max 5)",
        options=all_heroes,
        max_selections=5,
        placeholder="Select allies...",
        key="team_picks_multiselect",
    )

with top_col_enemies:
    opp_options = [h for h in all_heroes if h not in team_picks]
    opponent_picks = st.multiselect(
        "Enemies (max 5)",
        options=opp_options,
        max_selections=5,
        placeholder="Select enemies...",
        key="opponent_picks_multiselect",
    )

with top_col_stat:
    baseline = calculate_baseline_winrate(team_picks, opponent_picks)
    delta_val = (baseline - 0.50) * 100
    delta_str = f"{delta_val:+.1f}% vs 50%" if abs(delta_val) >= 0.3 else "Even (50/50)"
    st.metric(
        label="Draft Win Probability",
        value=f"{baseline * 100:.1f}%",
        delta=delta_str,
    )

st.divider()

# =====================================================================
# UI - Information-Dense Recommendations Table
# =====================================================================
if len(team_picks) >= 5:
    st.info("Your team is full (5/5 heroes picked).")
else:
    recs = get_recommendations(team_picks, opponent_picks)

    search_hero = st.text_input(
        "Search Hero",
        placeholder="Filter by hero name...",
        label_visibility="collapsed",
    )
    if search_hero.strip():
        recs = [r for r in recs if search_hero.strip().lower() in r["Hero"].lower()]

    if recs:
        table_data = [
            {
                "Rank": idx + 1,
                "Hero": r["Hero"],
                "Win Rate": f"{r['Win Rate'] * 100:.1f}%",
                "Win Impact": f"{r['Impact'] * 100:+.2f}%",
                "Attribute": r["Attribute"],
                "Key Mechanics": r["Mechanics"] if r["Mechanics"] else "—",
            }
            for idx, r in enumerate(recs)
        ]
        df = pd.DataFrame(table_data)

        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            height=650,
        )
    else:
        st.warning("No heroes found.")
