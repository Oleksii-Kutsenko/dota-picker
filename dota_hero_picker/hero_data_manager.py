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
    ) -> list[RawHeroData]:
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
        assert isinstance(data, list)
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
            assert isinstance(data, (list, dict))
            return data


class HeroProcessor:
    """Processes raw hero data into structured format."""

    STUN_NAME = "stun"

    def __init__(
        self,
        raw_heroes: list[RawHeroData],
        raw_abilities: dict[str, Any],
        raw_hero_abilities: dict[Any, Any],
    ) -> None:
        self.scaler = StandardScaler()
        self.process_heroes(raw_heroes)
        self.process_abilities(raw_abilities)
        self.process_hero_abilties(raw_hero_abilities)
        self.merge_and_assign_stun()

    def process_heroes(self, heroes: list[RawHeroData]) -> None:
        heroes_number = len(heroes)
        dataframe = pd.DataFrame(
            heroes.values(),
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
        dataframe["base_health_regen"].fillna(
            dataframe["base_health_regen"].median(), inplace=True
        )

        dataframe["turn_rate"].fillna(
            dataframe["turn_rate"].median(), inplace=True
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
            dataframe[numeric_cols]
        )

        dataframe = pd.get_dummies(
            dataframe, columns=["primary_attr"], dtype=int
        )
        mlb = MultiLabelBinarizer()
        roles_encoded = mlb.fit_transform(dataframe["roles"])
        roles_df = pd.DataFrame(
            roles_encoded,
            columns=mlb.classes_,
        )
        dataframe = pd.concat(
            [dataframe.drop("roles", axis=1), roles_df], axis=1
        )

        dataframe["attack_type"] = dataframe["attack_type"] == "Melee"
        dataframe["attack_type"] = dataframe["attack_type"].astype(int)
        self.processed_heroes = dataframe

    def process_abilities(self, raw_abilities: dict[str, Any]) -> None:
        abilities_dataframe = pd.DataFrame.from_dict(
            raw_abilities, orient="index"
        )

        def has_stun_indicator(attrib_list):
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
            abilities_dataframe["attrib"].values
        )

        abilities_dataframe = abilities_dataframe[["has_stun"]]
        abilities_dataframe = abilities_dataframe.reset_index(drop=False)
        abilities_dataframe.rename(
            columns={"index": "abilities"}, inplace=True
        )
        self.processed_abilities = abilities_dataframe

    def process_hero_abilties(
        self, raw_hero_abilities: dict[str, Any]
    ) -> None:
        # TODO: Parse facets
        self.processed_hero_abilites = pd.DataFrame.from_dict(
            raw_hero_abilities, orient="index"
        ).reset_index(drop=False)
        self.processed_hero_abilites.rename(
            columns={"index": "name"}, inplace=True
        )

    def merge_and_assign_stun(self):
        merged_df = pd.merge(
            self.processed_heroes,
            self.processed_hero_abilites[["name", "abilities"]],
            on="name",
            how="left",
        )

        exploded_hero_abil = merged_df[["hero_id", "name", "abilities"]]
        exploded_hero_abil = exploded_hero_abil.explode("abilities").dropna(
            subset=["abilities"]
        )

        exploded_with_stun = pd.merge(
            exploded_hero_abil,
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

        final_df = pd.merge(merged_df, stun_per_hero, on="hero_id", how="left")
        final_df["has_stun"] = final_df["has_stun"].fillna(0).astype(int)

        final_df.drop(["abilities"], axis=1, inplace=True, errors="ignore")
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

    FEATURES = [
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
    ]

    STUN_NAME = "STUN"

    def __init__(self) -> None:
        loader = DataLoader()
        raw_heroes: list[RawHeroData] = loader.get_hero_data(
            HEROES_FILE,
            API_HEROES_ENDPOINT,
        )
        raw_abilities = loader.load_json(ABILITIES_FILE)
        raw_hero_abilities = loader.load_json(HERO_ABILITIES_FILE)

        assert isinstance(raw_abilities, dict)
        assert isinstance(raw_hero_abilities, dict)

        self.processor = HeroProcessor(
            raw_heroes, raw_abilities, raw_hero_abilities
        )

    def get_heroes_number(self) -> int:
        return len(self.processor.processed_heroes)

    def get_hero_features(self, hero_id: int) -> list[int]:
        if hero_id == 0:
            return [0] * len(self.FEATURES)
        hero_row = self.processor.processed_heroes.loc[
            self.processor.processed_heroes["hero_id"] == hero_id
        ]
        return hero_row[self.FEATURES].iloc[0].values

    def get_hero_id_by_localized_name(self, localized_name: str) -> int:
        """Get model ID by localized hero name."""
        hero_id = self.processor.processed_heroes.loc[
            self.processor.processed_heroes["localized_name"] == localized_name
        ].iloc[0]["hero_id"]
        return hero_id

    def get_heroes_localized_names(self) -> dict[int, HeroData]:
        return self.processor.processed_heroes["localized_name"].tolist()

    def get_hero_id_by_api_id(self, api_id: int) -> int:
        return self.processor.processed_heroes.loc[
            self.processor.processed_heroes["id"] == api_id
        ].iloc[0]["hero_id"]
