import json
import os
from pathlib import Path

import requests

from manage import DotaPickerError
from settings import API_HEROES_ENDPOINT, HEROES_FILE


class HeroDataManager:
    def __init__(self) -> None:
        self.raw_hero_data = self.get_hero_data(
            HEROES_FILE, API_HEROES_ENDPOINT
        )
        self.hero_data = self.process_hero_data()

    @staticmethod
    def get_hero_data(local_file: Path, endpoint: str) -> list[dict[str, str]]:
        if os.path.exists(local_file):
            with open(local_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            response = requests.get(endpoint, timeout=5)
            if response.status_code != 200:
                raise DotaPickerError(
                    f"Failed to fetch heroes: {response.status_code}"
                )
            data = response.json()
            with open(local_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return data

    def process_hero_data(self):
        for model_id, raw_hero_data in enumerate(self.raw_hero_data):
            breakpoint()
