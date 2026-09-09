import json
from http import HTTPStatus
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from manage import DotaPickerError
from settings import (
    ABILITIES_FILE,
    API_HEROES_ENDPOINT,
    HERO_ABILITIES_FILE,
    HEROES_FILE,
)


def load_json_file(file_path: Path) -> dict[str, Any]:
    """Load and return JSON file content."""
    with file_path.open(encoding="utf-8") as file:
        data = json.load(file)
        assert isinstance(data, dict)
        return data


def fetch_hero_data(local_file: Path, endpoint: str) -> dict[str, Any]:
    """Load local hero data or fetch from API if not found."""
    if Path(local_file).exists():
        return load_json_file(local_file)

    response = requests.get(endpoint, timeout=5)
    if response.status_code != HTTPStatus.OK:
        message = f"Failed to fetch heroes from API: {response.status_code}"
        raise DotaPickerError(message)

    data = response.json()
    with Path(local_file).open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    assert isinstance(data, dict)
    return data


# =====================================================================
# Ability Mechanics Checkers (Readable, Standalone Functions)
# =====================================================================


def check_has_stun(ability: dict[str, Any]) -> bool:
    """Check if the ability applies a stun."""
    for attribute_entry in ability.get("attrib", []):
        if isinstance(attribute_entry, dict):
            key = attribute_entry.get("key", "").lower()
            header = attribute_entry.get("header", "").lower()
            if "stun" in key or "stun" in header:
                return True
    return False


def check_has_bkb_pierce(ability: dict[str, Any]) -> bool:
    """Check if the ability pierces debuff immunity (BKB)."""
    return str(ability.get("bkbpierce", "")).strip().lower() == "yes"


def check_has_silence(ability: dict[str, Any]) -> bool:
    """Check if the ability silences enemies."""
    display_name = str(ability.get("dname", "")).lower()
    description = str(ability.get("desc", "")).lower()
    if (
        "silence" in display_name
        or "silence" in description
        or "silences" in description
        or "preventing enemy" in description
    ):
        return True
    for attribute_entry in ability.get("attrib", []):
        if isinstance(attribute_entry, dict):
            key = attribute_entry.get("key", "").lower()
            header = attribute_entry.get("header", "").lower()
            if "silence" in key or "silence" in header:
                return True
    return False


def check_has_break(ability: dict[str, Any]) -> bool:
    """Check if the ability applies break (passive disabling)."""
    description = str(ability.get("desc", "")).lower()
    if "applying break" in description or "applies break" in description:
        return True
    for attribute_entry in ability.get("attrib", []):
        if isinstance(attribute_entry, dict):
            key = attribute_entry.get("key", "").lower()
            header = attribute_entry.get("header", "").lower()
            if (
                "break_duration" in key
                or "break duration" in header
                or key == "does_break"
            ):
                return True
    return False


def check_has_dispel(ability: dict[str, Any]) -> bool:
    """Check if the ability provides a dispel or dispels debuffs."""
    description = str(ability.get("desc", "")).lower()
    if "dispel type:" in description or "removes debuffs" in description:
        return True
    for attribute_entry in ability.get("attrib", []):
        if (
            isinstance(attribute_entry, dict)
            and "dispel" in attribute_entry.get("key", "").lower()
        ):
            return True
    return False


def check_has_root(ability: dict[str, Any]) -> bool:
    """Check if the ability roots or leashes enemies."""
    display_name = str(ability.get("dname", "")).lower()
    description = str(ability.get("desc", "")).lower()
    if any(
        keyword in display_name for keyword in ("root", "ensnare", "leash")
    ):
        return True
    if any(
        keyword in description
        for keyword in ("roots", "rooting", "leashes", "ensnares")
    ):
        return True
    for attribute_entry in ability.get("attrib", []):
        if isinstance(attribute_entry, dict):
            key = attribute_entry.get("key", "").lower()
            header = attribute_entry.get("header", "").lower()
            if any(
                keyword in key or keyword in header
                for keyword in ("root", "ensnare", "leash")
            ):
                return True
    return False


def check_is_illusion(ability: dict[str, Any]) -> bool:
    """Check if the ability creates player-controlled illusions."""
    description = str(ability.get("desc", "")).lower()
    if any(
        phrase in description
        for phrase in (
            "creating an illusion",
            "creates an illusion",
            "creates illusions",
            "mirror image",
        )
    ):
        return True
    for attribute_entry in ability.get("attrib", []):
        if isinstance(attribute_entry, dict):
            key = attribute_entry.get("key", "").lower()
            header = attribute_entry.get("header", "").lower()
            if (
                "illusion_duration" in key
                or "duration_illusion" in key
                or "illusion_damage" in key
                or "illusion_count" in key
                or ("illusion" in header and "duration" in header)
            ):
                return True
    return False


def check_has_pure_damage(ability: dict[str, Any]) -> bool:
    """Check if the ability deals Pure damage."""
    return str(ability.get("dmg_type", "")).strip().lower() == "pure"


def check_has_invis(ability: dict[str, Any]) -> bool:
    """Check if the ability grants invisibility."""
    description = str(ability.get("desc", "")).lower()
    if "reveals invisible" in description or "reveal invisible" in description:
        return False
    if any(
        phrase in description
        for phrase in (
            "invisible",
            "invisibly",
            "invisibility",
            "out of visibility",
        )
    ):
        return True
    for attribute_entry in ability.get("attrib", []):
        if (
            isinstance(attribute_entry, dict)
            and "fade_time" in attribute_entry.get("key", "").lower()
        ):
            return True
    return False


def check_has_heal(ability: dict[str, Any]) -> bool:
    """Check if the ability heals or restores health."""
    description = str(ability.get("desc", "")).lower()
    if any(
        phrase in description
        for phrase in ("heals", "healing", "restores health", "restores hp")
    ):
        return True
    for attribute_entry in ability.get("attrib", []):
        if isinstance(attribute_entry, dict):
            key = attribute_entry.get("key", "").lower()
            header = attribute_entry.get("header", "").lower()
            if "heal" in key or "heal" in header:
                return True
    return False


def extract_hero_ability_flags(
    hero_abilities_data: dict[str, Any],
    raw_abilities: dict[str, Any],
) -> dict[str, int]:
    """Collect all active abilities for a hero."""
    raw_abilities_list = list(hero_abilities_data.get("abilities", []))
    for facet in hero_abilities_data.get("facets", []):
        if facet.get("abilities") and facet.get("deprecated") != "true":
            raw_abilities_list.extend(facet.get("abilities", []))

    ability_names: list[str] = []
    for item in raw_abilities_list:
        if isinstance(item, list):
            ability_names.extend(
                str(sub_item) for sub_item in item if isinstance(sub_item, str)
            )
        elif isinstance(item, str):
            ability_names.append(item)

    ability_dicts = [
        raw_abilities[name]
        for name in ability_names
        if name in raw_abilities and isinstance(raw_abilities[name], dict)
    ]

    return {
        "has_stun": int(
            any(check_has_stun(ability) for ability in ability_dicts),
        ),
        "has_bkb_pierce": int(
            any(check_has_bkb_pierce(ability) for ability in ability_dicts),
        ),
        "has_silence": int(
            any(check_has_silence(ability) for ability in ability_dicts),
        ),
        "has_break": int(
            any(check_has_break(ability) for ability in ability_dicts),
        ),
        "has_dispel": int(
            any(check_has_dispel(ability) for ability in ability_dicts),
        ),
        "has_root": int(
            any(check_has_root(ability) for ability in ability_dicts),
        ),
        "is_illusion": int(
            any(check_is_illusion(ability) for ability in ability_dicts),
        ),
        "has_pure_damage": int(
            any(check_has_pure_damage(ability) for ability in ability_dicts),
        ),
        "has_invis": int(
            any(check_has_invis(ability) for ability in ability_dicts),
        ),
        "has_heal": int(
            any(check_has_heal(ability) for ability in ability_dicts),
        ),
    }


# =====================================================================
# Main Hero Data Manager
# =====================================================================


class HeroDataManager:
    """Class for hero data interactions and tabular feature preparation."""

    FEATURES = (
        "attack_type",
        "base_health_regen",
        "base_mana",
        "base_mana_regen",
        "base_armor",
        "base_mr",
        "base_attack_min",
        "base_attack_max",
        "base_str",
        "base_agi",
        "base_int",
        "str_gain",
        "agi_gain",
        "int_gain",
        "attack_range",
        "projectile_speed",
        "attack_rate",
        "base_attack_time",
        "attack_point",
        "move_speed",
        "turn_rate",
        "day_vision",
        "night_vision",
        "primary_attr_agi",
        "primary_attr_all",
        "primary_attr_int",
        "primary_attr_str",
        "Carry",
        "Disabler",
        "Durable",
        "Escape",
        "Initiator",
        "Nuker",
        "Pusher",
        "Support",
        "has_stun",
        "has_bkb_pierce",
        "has_silence",
        "has_break",
        "has_dispel",
        "has_root",
        "is_illusion",
        "has_pure_damage",
        "has_invis",
        "has_heal",
    )

    def __init__(self) -> None:
        raw_heroes = fetch_hero_data(HEROES_FILE, API_HEROES_ENDPOINT)
        raw_abilities = load_json_file(ABILITIES_FILE)
        raw_hero_abilities = load_json_file(HERO_ABILITIES_FILE)

        self.scaler = StandardScaler()
        self.processed_heroes = self._process_heroes(
            raw_heroes,
            raw_abilities,
            raw_hero_abilities,
        )
        # Expose self.processor for backwards-compatibility
        self.processor = self

    def _process_heroes(
        self,
        raw_heroes: dict[str, Any],
        raw_abilities: dict[str, Any],
        raw_hero_abilities: dict[str, Any],
    ) -> pd.DataFrame:
        heroes_number = len(raw_heroes)
        hero_records = list(raw_heroes.values())

        columns = [
            "id",
            "name",
            "localized_name",
            "primary_attr",
            "roles",
            "attack_type",
            "base_health_regen",
            "base_mana",
            "base_mana_regen",
            "base_armor",
            "base_mr",
            "base_attack_min",
            "base_attack_max",
            "base_str",
            "base_agi",
            "base_int",
            "str_gain",
            "agi_gain",
            "int_gain",
            "attack_range",
            "projectile_speed",
            "attack_rate",
            "base_attack_time",
            "attack_point",
            "move_speed",
            "turn_rate",
            "day_vision",
            "night_vision",
        ]
        dataframe = pd.DataFrame(hero_records, columns=columns)
        dataframe["hero_id"] = range(1, heroes_number + 1)

        # Impute missing stats
        dataframe["base_health_regen"] = dataframe["base_health_regen"].fillna(
            dataframe["base_health_regen"].median(),
        )
        dataframe["turn_rate"] = dataframe["turn_rate"].fillna(
            dataframe["turn_rate"].median(),
        )

        # Standardize numeric stats
        numeric_columns = [
            "base_health_regen",
            "base_mana",
            "base_mana_regen",
            "base_armor",
            "base_mr",
            "base_attack_min",
            "base_attack_max",
            "base_str",
            "base_agi",
            "base_int",
            "str_gain",
            "agi_gain",
            "int_gain",
            "attack_range",
            "projectile_speed",
            "attack_rate",
            "base_attack_time",
            "attack_point",
            "move_speed",
            "turn_rate",
            "day_vision",
            "night_vision",
        ]
        dataframe[numeric_columns] = self.scaler.fit_transform(
            dataframe[numeric_columns],
        )

        # One-hot encode primary attribute
        dataframe = pd.get_dummies(
            dataframe,
            columns=["primary_attr"],
            dtype=int,
        )

        # Multi-label binarize roles
        multi_label_binarizer = MultiLabelBinarizer()
        roles_encoded = multi_label_binarizer.fit_transform(dataframe["roles"])
        roles_dataframe = pd.DataFrame(
            roles_encoded,
            columns=multi_label_binarizer.classes_,
        )
        dataframe = pd.concat(
            [dataframe.drop("roles", axis=1), roles_dataframe],
            axis=1,
        )

        # Binary attack type (1 = Melee, 0 = Ranged)
        dataframe["attack_type"] = (
            dataframe["attack_type"] == "Melee"
        ).astype(int)

        # Compute ability indicators per hero
        ability_flag_records = [
            extract_hero_ability_flags(
                hero_abilities_data=raw_hero_abilities.get(hero_name, {}),
                raw_abilities=raw_abilities,
            )
            for hero_name in dataframe["name"]
        ]
        ability_flags_dataframe = pd.DataFrame(
            ability_flag_records,
            index=dataframe.index,
        )
        return pd.concat([dataframe, ability_flags_dataframe], axis=1)

    def get_heroes_number(self) -> int:
        """Return total number of processed heroes."""
        return len(self.processed_heroes)

    def get_hero_features(self, hero_id: int) -> list[float]:
        """Return tabular feature vector for a hero ID (0 for empty slot)."""
        if hero_id == 0:
            return [0.0] * len(self.FEATURES)
        hero_row = self.processed_heroes.loc[
            self.processed_heroes["hero_id"] == hero_id
        ]
        return list(hero_row[list(self.FEATURES)].iloc[0].to_numpy())

    def get_hero_id_by_localized_name(self, localized_name: str) -> int:
        """Get model ID by localized hero name."""
        return int(
            self.processed_heroes.loc[
                self.processed_heroes["localized_name"] == localized_name
            ].iloc[0]["hero_id"],
        )

    def get_heroes_localized_names(self) -> list[str]:
        """Return all localized hero names."""
        return self.processed_heroes["localized_name"].tolist()

    def get_hero_id_by_api_id(self, api_id: int) -> int:
        """Get model ID by Valve API ID."""
        hero_row = self.processed_heroes.loc[
            self.processed_heroes["id"] == api_id
        ].iloc[0]
        return int(hero_row["hero_id"])

    def get_projected_hero_embeddings(self, embedding_dim: int) -> np.ndarray:
        num_heroes = self.get_heroes_number()
        num_features = len(self.FEATURES)

        # Build feature matrix (index 0 is padding)
        feature_matrix = np.zeros(
            (num_heroes + 1, num_features),
            dtype=np.float32,
        )
        for hero_id in range(1, num_heroes + 1):
            feature_matrix[hero_id] = self.get_hero_features(hero_id)

        # Fit PCA on actual heroes (rows 1:), not padding
        num_components = min(embedding_dim, num_features)
        pca = PCA(n_components=num_components, random_state=42)
        projected_heroes = pca.fit_transform(feature_matrix[1:])
        embeddings = np.zeros(
            (num_heroes + 1, embedding_dim),
            dtype=np.float32,
        )
        embeddings[1:, :num_components] = projected_heroes

        # Scale non-padding vectors to standard initialization variance
        std = embeddings[1:].std()
        if std > 0:
            embeddings[1:] = (embeddings[1:] / std) * (
                1.0 / np.sqrt(embedding_dim)
            )

        return embeddings
