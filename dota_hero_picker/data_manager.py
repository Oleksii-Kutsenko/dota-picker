import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from dota_hero_picker.hero_data_manager import HeroDataManager

from .data_preparation import (
    create_augmented_dataframe,
    prepare_dataframe,
)
from .training_utils import (
    DotaDataset,
    compute_baseline_f1,
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
        self.train_dataset, self.val_dataset, self.test_dataset = (
            self.prepare_datasets()
        )

    def create_matches_dataframe(
        self,
    ) -> pd.DataFrame:
        matches_dataframe = pd.read_csv(
            self.csv_file_path,
            converters={
                "team_picks": json.loads,
                "opponent_picks": json.loads,
            },
            dtype={
                "win": int,
                "picked_hero": int,
            },
        )

        team_pick_cols = [f"team_pick_{i}" for i in range(1, 6)]
        opp_pick_cols = [f"opp_pick_{i}" for i in range(1, 6)]

        matches_dataframe[team_pick_cols] = pd.DataFrame(
            matches_dataframe["team_picks"].tolist(),
            index=matches_dataframe.index,
        )
        matches_dataframe[opp_pick_cols] = pd.DataFrame(
            matches_dataframe["opponent_picks"].tolist(),
            index=matches_dataframe.index,
        )

        all_pick_cols = team_pick_cols + opp_pick_cols + ["picked_hero"]
        for col in all_pick_cols:
            matches_dataframe[col] = matches_dataframe[col].map(
                self.hero_data_manager.get_hero_id_by_api_id,
            )

        return matches_dataframe.drop(
            columns=["team_picks", "opponent_picks"],
        )

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

        logger.info(
            f"Size of augmented dataset {len(augmented_train_dataframe)}",
        )

        prepared_validation_dataframe = prepare_dataframe(validation_dataframe)
        prepared_test_dataframe = prepare_dataframe(test_dataframe)

        compute_baseline_f1(
            augmented_train_dataframe["win"],
            prepared_test_dataframe["win"],
        )

        train_dataset = DotaDataset(augmented_train_dataframe)
        val_dataset = DotaDataset(prepared_validation_dataframe)
        test_dataset = DotaDataset(prepared_test_dataframe)

        return train_dataset, val_dataset, test_dataset
