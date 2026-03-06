import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from dota_hero_picker.hero_data_manager import HeroDataManager

from .data_preparation import (
    create_augmented_dataframe,
    enrich_dataframe,
    prepare_dataframe,
)
from .training_utils import (
    DotaDataset,
    compute_baseline_f1,
    compute_pos_weight,
)

logger = logging.getLogger(__name__)


class DataManager:
    """Class for datasets creation."""

    def __init__(
        self,
        csv_file_path: Path,
        hero_data_manager: HeroDataManager,
        random_state: int = 42,
    ) -> None:
        self.csv_file_path = csv_file_path
        self.hero_data_manager = hero_data_manager
        self.random_state = random_state

        self.matches_dataframe = self.create_matches_dataframe()
        self.pos_weight = compute_pos_weight(self.matches_dataframe)
        self.train_dataset, self.val_dataset, self.test_dataset = (
            self.prepare_datasets()
        )

    def create_matches_dataframe(
        self,
    ) -> pd.DataFrame:
        matches_dataframe = pd.read_csv(
            self.csv_file_path,
            converters={
                "team_picks": str,
                "opponent_picks": str,
            },
            dtype={
                "win": int,
                "picked_hero": int,
            },
        )
        pick_columns = [
            "team_picks",
            "opponent_picks",
        ]
        for column in pick_columns:
            matches_dataframe[column] = matches_dataframe[column].apply(
                json.loads,
            )
            matches_dataframe[column] = matches_dataframe[column].apply(
                lambda hero_list: [
                    self.hero_data_manager.get_hero_id_by_api_id(api_id)
                    for api_id in hero_list
                ],
            )
        matches_dataframe["picked_hero"] = matches_dataframe[
            "picked_hero"
        ].map(self.hero_data_manager.get_hero_id_by_api_id)

        return matches_dataframe

    def prepare_datasets(
        self,
    ) -> tuple[DotaDataset, DotaDataset, DotaDataset]:
        train_dataframe, tmp_dataframe = train_test_split(
            self.matches_dataframe,
            test_size=0.2,
            stratify=self.matches_dataframe["win"],
            random_state=self.random_state,
        )
        validation_dataframe, test_dataframe = train_test_split(
            tmp_dataframe,
            test_size=0.5,
            stratify=tmp_dataframe["win"],
            random_state=self.random_state,
        )

        augmented_train_dataframe = create_augmented_dataframe(train_dataframe)
        enriched_train_dataframe = enrich_dataframe(
            augmented_train_dataframe,
            self.hero_data_manager,
        )

        logger.info(
            f"Size of augmented dataset {len(augmented_train_dataframe)}",
        )

        prepared_validation_dataframe = enrich_dataframe(
            prepare_dataframe(validation_dataframe),
            self.hero_data_manager,
        )
        prepared_test_dataframe = enrich_dataframe(
            prepare_dataframe(test_dataframe),
            self.hero_data_manager,
        )

        compute_baseline_f1(
            augmented_train_dataframe["win"],
            prepared_test_dataframe["win"],
        )

        train_dataset = DotaDataset(enriched_train_dataframe)
        val_dataset = DotaDataset(prepared_validation_dataframe)
        test_dataset = DotaDataset(prepared_test_dataframe)

        return train_dataset, val_dataset, test_dataset
