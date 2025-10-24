import json
import os
from pathlib import Path

import requests

from manage import DotaPickerError
from settings import (
    ABILITIES_FILE,
    API_HEROES_ENDPOINT,
    HERO_ABILITIES_FILE,
    HEROES_FILE,
)


class HeroDataManager:
    """
    Class for the hero data interactions
    """

    HERO_FEATURES_NUM = 9

    def __init__(self) -> None:
        self.raw_hero_data = self.get_hero_data(
            HEROES_FILE,
            API_HEROES_ENDPOINT,
        )
        self.read_raw_abilities()
        self.read_raw_hero_abilities()
        self.api_id_2_model_id = {}
        self.model_id_2_hero_data = {}
        self.model_id_2_name = {}
        self.name_2_model_id = {}
        self.local_name_2_model_id = {}
        self.hero_roles_vector = self.create_hero_roles_vector()
        self.process_hero_data()
        self.process_hero_abilties()
        breakpoint()

    def read_raw_abilities(self) -> None:
        with open(ABILITIES_FILE, encoding="utf-8") as f:
            self.raw_abilities = json.load(f)

    def read_raw_hero_abilities(self) -> None:
        with open(HERO_ABILITIES_FILE, encoding="utf-8") as f:
            self.raw_hero_abilities = json.load(f)

    def create_hero_roles_vector(self) -> list[str]:
        unique_roles = set()
        for hero_data in self.raw_hero_data:
            unique_roles |= set(hero_data["roles"])
        return list(unique_roles)

    @staticmethod
    def get_hero_data(local_file: Path, endpoint: str) -> list[dict[str, str]]:
        if os.path.exists(local_file):
            with open(local_file, encoding="utf-8") as f:
                return json.load(f)
        else:
            response = requests.get(endpoint, timeout=5)
            if response.status_code != 200:
                raise DotaPickerError(
                    f"Failed to fetch heroes: {response.status_code}",
                )
            data = response.json()
            with open(local_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return data

    def process_hero_data(self):
        for model_id, raw_hero_data in enumerate(self.raw_hero_data, 1):
            self.api_id_2_model_id[raw_hero_data["id"]] = model_id
            hero_data = {
                "is_melee": 1
                if raw_hero_data["attack_type"] == "Melee"
                else 0,
                "roles": [
                    1 if role in raw_hero_data["roles"] else 0
                    for role in self.hero_roles_vector
                ],
            }
            self.model_id_2_hero_data[model_id] = hero_data
            self.local_name_2_model_id[raw_hero_data["localized_name"]] = (
                model_id
            )
            self.name_2_model_id[raw_hero_data["name"]] = model_id
            self.model_id_2_name[model_id] = raw_hero_data["name"]

    def get_hero_features(self, model_id: int) -> list[int]:
        if model_id == 0:
            return [0] * self.HERO_FEATURES_NUM
        hero_data = self.model_id_2_hero_data[model_id]
        return [hero_data["is_melee"]] + hero_data["roles"]

    def process_hero_abilties(self):
        STUN_NAME = "STUN"
        for name in self.raw_hero_abilities:
            model_id = self.name_2_model_id[name]
            abilities = self.raw_hero_abilities[name]["abilities"]
            self.model_id_2_hero_data[model_id]["has_stun"] = 0
            for ability in abilities:
                if STUN_NAME in str(self.raw_abilities[ability]):
                    self.model_id_2_hero_data[model_id]["has_stun"] = 1
