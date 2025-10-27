import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import requests

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

    STUN_NAME = "STUN"

    def __init__(
        self,
        raw_heroes: list[RawHeroData],
        raw_abilities: list[dict[Any, Any]],
        raw_hero_abilities: dict[Any, Any],
    ) -> None:
        self.raw_abilities = raw_abilities
        self.raw_hero_abilities = raw_hero_abilities
        self.role_vector: list[str] = []
        self.primary_attr_vector = ["agi", "str", "int"]
        self.build_role_vector(raw_heroes)

    def build_role_vector(self, raw_heroes: list[RawHeroData]) -> list[str]:
        unique_roles = set()
        for hero in raw_heroes.values():
            if "roles" in hero:
                unique_roles.update(hero["roles"])
        self.role_vector = sorted(unique_roles)
        return self.role_vector

    def process_hero(self, raw_hero: RawHeroData, model_id: int) -> HeroData:
        attributes = self._build_attributes(raw_hero)
        abilities = self._build_abilities(raw_hero["name"])
        breakpoint()

        return HeroData(
            model_id=model_id,
            api_id=int(raw_hero["id"]),
            name=raw_hero["name"],
            localized_name=raw_hero["localized_name"],
            attributes=attributes,
            abilities=abilities,
        )

    def _build_attributes(self, raw_hero: RawHeroData) -> HeroAttributesDict:
        return {
            "is_melee": 1 if raw_hero["attack_type"] == "Melee" else 0,
            "roles": [
                1 if role in raw_hero["roles"] else 0
                for role in self.role_vector
            ],
            "primary_attr": [
                1 if primary_attr == raw_hero["primary_attr"] else 0
                for primary_attr in self.primary_attr_vector
            ],
            "base_health": raw_hero["base_health"],
            "base_health_regen": raw_hero["base_health_regen"],
            "base_mana": raw_hero["base_mana"],
            "base_mana_regen": raw_hero["base_mana_regen"],
            "base_armor": raw_hero["base_armor"],
            "base_mr": raw_hero["base_mr"],
            "base_attack_min": raw_hero["base_attack_min"],
            "base_attack_max": raw_hero["base_attack_max"],
            "base_str": raw_hero["base_str"],
            "base_agi": raw_hero["base_agi"],
            "base_int": raw_hero["base_int"],
            "str_gain": raw_hero["str_gain"],
            "agi_gain": raw_hero["agi_gain"],
            "int_gain": raw_hero["int_gain"],
            "attack_range": raw_hero["attack_range"],
            "projectile_speed": raw_hero["projectile_speed"],
            "attack_rate": raw_hero["attack_rate"],
            "base_attack_time": raw_hero["base_attack_time"],
            "attack_point": raw_hero["attack_point"],
            "move_speed": raw_hero["move_speed"],
            "turn_rate": raw_hero["turn_rate"],
            "day_vision": raw_hero["day_vision"],
            "night_vision": raw_hero["night_vision"],
        }

    def _build_abilities(self, hero_name: str) -> HeroAbilitiesDict:
        abilities_dict: HeroAbilitiesDict = {"has_stun": 0}

        hero_ability_data = self.raw_hero_abilities.get(hero_name, {})
        ability_names = hero_ability_data.get("abilities", [])

        for ability_name in ability_names:
            if ability_name in self.raw_abilities:
                ability_data = str(self.raw_abilities[ability_name])
                if self.STUN_NAME in ability_data:
                    abilities_dict["has_stun"] = 1
                    break

        return abilities_dict


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

    HERO_FEATURES_NUM = 13
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

        processor = HeroProcessor(
            raw_heroes, raw_abilities, raw_hero_abilities
        )

        self._registry = HeroRegistry()
        for model_id, raw_hero in enumerate(raw_heroes.values(), start=1):
            hero = processor.process_hero(raw_hero, model_id)
            self._registry.register(hero)

    def get_heroes_number(self) -> int:
        return self._registry.get_heroes_number()

    def get_hero_features(self, model_id: int) -> list[int]:
        if model_id == 0:
            return [0] * self.HERO_FEATURES_NUM
        hero_data = self._registry.get_hero_by_hero_id(model_id)
        attributes = hero_data.attributes
        abilities = hero_data.abilities

        return (
            [attributes["is_melee"]]
            + [abilities["has_stun"]]
            + attributes["roles"]
            + attributes["primary_attr"]
        )

    def get_hero_id_by_localized_name(self, localized_name: str) -> int:
        """Get model ID by localized hero name."""
        hero = self._registry.get_hero_by_localized_name(localized_name)
        return hero.model_id

    def get_heroes(self) -> dict[int, HeroData]:
        return self._registry.get_heroes()

    def get_hero_id_by_api_id(self, api_id: int) -> int:
        return self._registry.get_hero_id_by_api_id(api_id)
