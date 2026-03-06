import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from manage import DotaPickerError
from settings import (
    ABILITIES_FILE,
    API_HEROES_ENDPOINT,
    HERO_ABILITIES_FILE,
    HEROES_FILE,
)

from .load_personal_matches import HTTP_OK

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
    "Wraith King": [1, 3],
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


class HeroAttributesDict(TypedDict):
    """Hero attributes dictionary."""

    is_melee: int
    roles: list[int]


class HeroAbilitiesDict(TypedDict):
    """Type definition for hero abilities."""

    has_stun: int


@dataclass
class HeroData:
    """Consolidated hero data."""

    model_id: int
    api_id: int
    name: str
    localized_name: str
    attributes: HeroAttributesDict
    abilities: HeroAbilitiesDict


class RawHeroData(TypedDict):
    """Raw hero data dict."""

    attack_type: str
    roles: list[str]
    id: int
    localized_name: str
    name: str


class DataLoader:
    """Handles loading data from files and API."""

    HTTP_OK = 200
    REQUEST_TIMEOUT = 5

    @classmethod
    def get_hero_data(
        cls,
        local_file: Path,
        endpoint: str,
    ) -> dict[str, RawHeroData]:
        if Path(local_file).exists():
            return cls.load_hero_json(local_file)
        response = requests.get(endpoint, timeout=5)
        if response.status_code != HTTP_OK:
            msg = f"Failed to fetch heroes: {response.status_code}"
            raise DotaPickerError(
                msg,
            )
        data = response.json()
        with Path(local_file).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        assert isinstance(data, dict)
        return data

    @staticmethod
    def load_hero_json(file_path: Path) -> dict[str, RawHeroData]:
        with file_path.open(encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict)
            return data

    @staticmethod
    def load_json(
        file_path: Path,
    ) -> dict[str, Any]:
        with file_path.open(encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict)
            return data


class HeroProcessor:
    """Processes raw hero data into structured format."""

    STUN_NAME = "stun"

    def __init__(
        self,
        raw_heroes: dict[str, RawHeroData],
        raw_abilities: dict[str, Any],
        raw_hero_abilities: dict[Any, Any],
    ) -> None:
        self.scaler = StandardScaler()
        self.process_heroes(raw_heroes)
        self.process_abilities(raw_abilities)
        self.process_hero_abilties(raw_hero_abilities)
        self.merge_and_assign_stun()

    def process_heroes(self, heroes: dict[str, RawHeroData]) -> None:
        heroes_number = len(heroes)
        hero_dicts = list(heroes.values())
        dataframe = pd.DataFrame(
            hero_dicts,
            columns=[
                "id",
                "name",
                "localized_name",
                "primary_attr",
                "roles",
                "attack_type",
                # "base_health",
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
            ],
        )
        dataframe["base_health_regen"] = dataframe["base_health_regen"].fillna(
            dataframe["base_health_regen"].median(),
        )

        dataframe["turn_rate"] = dataframe["turn_rate"].fillna(
            dataframe["turn_rate"].median(),
        )
        dataframe["hero_id"] = range(1, heroes_number + 1)
        numeric_cols = [
            # "base_health",
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
        dataframe[numeric_cols] = self.scaler.fit_transform(
            dataframe[numeric_cols],
        )

        dataframe = pd.get_dummies(
            dataframe,
            columns=["primary_attr"],
            dtype=int,
        )
        mlb = MultiLabelBinarizer()
        roles_encoded = mlb.fit_transform(dataframe["roles"])
        roles_df = pd.DataFrame(
            roles_encoded,
            columns=mlb.classes_,
        )
        dataframe = pd.concat(
            [dataframe.drop("roles", axis=1), roles_df],
            axis=1,
        )

        dataframe["attack_type"] = dataframe["attack_type"] == "Melee"
        dataframe["attack_type"] = dataframe["attack_type"].astype(int)
        self.processed_heroes = dataframe

    def process_abilities(self, raw_abilities: dict[str, Any]) -> None:
        abilities_dataframe = pd.DataFrame.from_dict(
            raw_abilities,
            orient="index",
        )

        def has_stun_indicator(attrib_list: list[dict[str, str]]) -> bool:
            if not isinstance(attrib_list, list):
                return False
            for d in attrib_list:
                if isinstance(d, dict):
                    key_lower = d.get("key", "").lower()
                    header_lower = d.get("header", "").lower()
                    if "stun" in key_lower or "stun" in header_lower:
                        return True
            return False

        vectorized_check = np.vectorize(has_stun_indicator)
        abilities_dataframe["has_stun"] = vectorized_check(
            abilities_dataframe["attrib"].to_numpy(),
        )

        abilities_dataframe = abilities_dataframe[["has_stun"]]
        abilities_dataframe = abilities_dataframe.reset_index(drop=False)
        abilities_dataframe = abilities_dataframe.rename(
            columns={"index": "abilities"},
        )
        self.processed_abilities = abilities_dataframe

    def process_hero_abilties(
        self,
        raw_hero_abilities: dict[str, Any],
    ) -> None:
        preprocessed_hero_abilities = []
        for hero_name, hero_abilities_data in raw_hero_abilities.items():
            preprocessed_hero_abilities.append(
                {
                    "name": hero_name,
                    "abilities": hero_abilities_data["abilities"]
                    + [
                        ability
                        for facets in hero_abilities_data["facets"]
                        if facets.get("abilities")
                        for ability in facets.get("abilities")
                    ],
                },
            )

        hero_abilities_dataframe = pd.DataFrame(preprocessed_hero_abilities)
        self.processed_hero_abilites = hero_abilities_dataframe

    def merge_and_assign_stun(self) -> None:
        merged_df = self.processed_heroes.merge(
            self.processed_hero_abilites[["name", "abilities"]],
            on="name",
            how="left",
        )

        exploded_hero_abil = merged_df[["hero_id", "name", "abilities"]]
        exploded_hero_abil = exploded_hero_abil.explode("abilities").dropna(
            subset=["abilities"],
        )

        exploded_with_stun = exploded_hero_abil.merge(
            self.processed_abilities[["abilities", "has_stun"]],
            on="abilities",
            how="left",
        )
        exploded_with_stun["has_stun"] = (
            exploded_with_stun["has_stun"].fillna(0).astype(int)
        )

        stun_per_hero = (
            exploded_with_stun.groupby("hero_id")["has_stun"]
            .any()
            .astype(int)
            .reset_index()
        )

        final_df = merged_df.merge(stun_per_hero, on="hero_id", how="left")
        final_df["has_stun"] = final_df["has_stun"].fillna(0).astype(int)

        final_df = final_df.drop(["abilities"], axis=1, errors="ignore")
        self.processed_heroes = final_df


class HeroRegistry:
    """Registry for quick hero lookups."""

    def __init__(self) -> None:
        self._heroes: dict[int, HeroData] = {}
        self._api_id_index: dict[int, int] = {}
        self._name_index: dict[str, int] = {}
        self._local_name_index: dict[str, int] = {}

    def register(self, hero: HeroData) -> None:
        self._heroes[hero.model_id] = hero
        self._api_id_index[hero.api_id] = hero.model_id
        self._name_index[hero.name] = hero.model_id
        self._local_name_index[hero.localized_name] = hero.model_id

    def get_heroes_number(self) -> int:
        return len(self._heroes)

    def get_hero_by_localized_name(self, localized_name: str) -> HeroData:
        model_id = self._local_name_index[localized_name]
        return self._heroes[model_id]

    def get_hero_by_hero_id(self, model_id: int) -> HeroData:
        return self._heroes[model_id]

    def get_heroes(self) -> dict[int, HeroData]:
        return self._heroes

    def get_hero_id_by_api_id(self, api_id: int) -> int:
        return self._api_id_index[api_id]


class HeroDataManager:
    """Class for the hero data interactions."""

    FEATURES = (
        "hero_id",
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
    )

    STUN_NAME = "STUN"

    def __init__(self) -> None:
        loader = DataLoader()
        raw_heroes: dict[str, RawHeroData] = loader.get_hero_data(
            HEROES_FILE,
            API_HEROES_ENDPOINT,
        )
        raw_abilities = loader.load_json(ABILITIES_FILE)
        raw_hero_abilities = loader.load_json(HERO_ABILITIES_FILE)

        assert isinstance(raw_abilities, dict)
        assert isinstance(raw_hero_abilities, dict)

        self.processor = HeroProcessor(
            raw_heroes,
            raw_abilities,
            raw_hero_abilities,
        )

    def get_heroes_number(self) -> int:
        return len(self.processor.processed_heroes)

    def get_hero_features(self, hero_id: int) -> list[int]:
        if hero_id == 0:
            return [0] * len(self.FEATURES)
        hero_row = self.processor.processed_heroes.loc[
            self.processor.processed_heroes["hero_id"] == hero_id
        ]
        return list(hero_row[list(self.FEATURES)].iloc[0].to_numpy())

    def get_hero_id_by_localized_name(self, localized_name: str) -> int:
        """Get model ID by localized hero name."""
        return int(
            self.processor.processed_heroes.loc[
                self.processor.processed_heroes["localized_name"]
                == localized_name
            ].iloc[0]["hero_id"],
        )

    def get_heroes_localized_names(self) -> list[str]:
        return self.processor.processed_heroes["localized_name"].tolist()

    def get_hero_id_by_api_id(self, api_id: int) -> int:
        hero_row = self.processor.processed_heroes.loc[
            self.processor.processed_heroes["id"] == api_id
        ].iloc[0]
        return int(hero_row["hero_id"])
