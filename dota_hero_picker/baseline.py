import json
import logging
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics import log_loss, roc_auc_score

from dota_hero_picker.data_manager import DataManager
from dota_hero_picker.hero_data_manager import HeroDataManager

logger = logging.getLogger(__name__)

API_HERO_STATS_ENDPOINT = "https://api.opendota.com/api/heroStats"
API_EXPLORER_ENDPOINT = "https://api.opendota.com/api/explorer"
HERO_STATS_CACHE_FILE = Path(
    "dota_hero_picker/constants/hero_stats.json",
)
RADIANT_STATS_CACHE_FILE = Path(
    "dota_hero_picker/constants/radiant_stats.json",
)

ALLY_SLOT_INDICES = (0, 1, 4, 5, 8)
ENEMY_SLOT_INDICES = (2, 3, 6, 7)


def load_public_hero_winrates(
    hero_data_manager: HeroDataManager,
) -> np.ndarray:
    """Fetch public hero win rates and map directly to internal IDs."""
    if HERO_STATS_CACHE_FILE.exists():
        with HERO_STATS_CACHE_FILE.open("r", encoding="utf-8") as file:
            stats_data = json.load(file)
    else:
        response = requests.get(API_HERO_STATS_ENDPOINT, timeout=10)
        response.raise_for_status()
        stats_data = response.json()
        HERO_STATS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HERO_STATS_CACHE_FILE.open("w", encoding="utf-8") as file:
            json.dump(stats_data, file, indent=2)

    total_heroes_count = hero_data_manager.get_heroes_number()
    hero_winrates = np.zeros(total_heroes_count + 1, dtype=np.float32)

    for hero_entry in stats_data:
        model_id = hero_data_manager.get_hero_id_by_api_id(hero_entry["id"])
        hero_winrates[model_id] = (
            hero_entry["pub_win"] / hero_entry["pub_pick"]
        )

    return hero_winrates


def load_public_radiant_advantage() -> float:
    """Fetch aggregate public radiant win rate from OpenDota Explorer."""
    if RADIANT_STATS_CACHE_FILE.exists():
        with RADIANT_STATS_CACHE_FILE.open("r", encoding="utf-8") as file:
            stats = json.load(file)
    else:
        sql = (
            "SELECT "
            "count(*) filter (where radiant_win is true) "
            "as radiant_wins, "
            "count(*) filter (where radiant_win is not null) "
            "as total_matches "
            "FROM matches"
        )
        response = requests.get(
            API_EXPLORER_ENDPOINT,
            params={"sql": sql},
            timeout=10,
        )
        response.raise_for_status()
        stats = response.json()
        RADIANT_STATS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with RADIANT_STATS_CACHE_FILE.open("w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2)

    row = stats["rows"][0]
    radiant_winrate = row["radiant_wins"] / row["total_matches"]
    return float(radiant_winrate - 0.50)


def predict_draft_probability(
    draft: np.ndarray,
    is_radiant: int,
    hero_winrates: np.ndarray,
    radiant_advantage: float,
) -> float:
    """Predict win probability for a draft using public meta hero win rates."""
    ally_heroes = [draft[i] for i in ALLY_SLOT_INDICES if draft[i] > 0]
    enemy_heroes = [draft[i] for i in ENEMY_SLOT_INDICES if draft[i] > 0]

    ally_mean = (
        float(np.mean(hero_winrates[ally_heroes])) if ally_heroes else 0.50
    )
    enemy_mean = (
        float(np.mean(hero_winrates[enemy_heroes])) if enemy_heroes else 0.50
    )

    map_bias = radiant_advantage if is_radiant == 1 else -radiant_advantage
    prob = 0.50 + (ally_mean - enemy_mean) + map_bias
    return float(np.clip(prob, 0.01, 0.99))


def compute_meta_baseline(
    data_manager: DataManager,
) -> tuple[float, float]:
    """Compute baseline loss and AUC using public meta rates and bias."""
    hero_winrates = load_public_hero_winrates(
        data_manager.hero_data_manager,
    )
    radiant_advantage = load_public_radiant_advantage()

    val_dataset = data_manager.val_dataset
    drafts = val_dataset.draft_sequences.cpu().numpy()
    is_radiant = val_dataset.match_contexts[:, 2].cpu().numpy()
    wins = val_dataset.wins.cpu().numpy()

    probabilities = np.array(
        [
            predict_draft_probability(
                draft,
                side,
                hero_winrates,
                radiant_advantage,
            )
            for draft, side in zip(drafts, is_radiant, strict=True)
        ],
    )

    baseline_loss = float(log_loss(wins, probabilities))
    baseline_auc = float(roc_auc_score(wins, probabilities))

    logger.info(
        f"Meta Baseline (Radiant Adv: {radiant_advantage:+.4f}) -> "
        f"Loss: {baseline_loss:.4f} | AUC: {baseline_auc:.4f}",
    )
    return baseline_loss, baseline_auc
