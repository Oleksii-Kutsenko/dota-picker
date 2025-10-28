import json
import logging
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

import settings
from dota_hero_picker.hero_data_manager import HeroDataManager

from .data_preparation import (
    create_augmented_dataframe,
    enrich_dataframe,
    prepare_dataframe,
)
from .neural_network import NNParameters, RNNWinPredictor
from .training_utils import (
    DotaDataset,
    OptimizerParameters,
    SchedulerParameters,
    ShuffleEnum,
    TrainingArguments,
    TrainingData,
    compute_baseline_f1,
    compute_pos_weight,
    count_trainable_params,
    evaluate_model,
    get_data_loader,
    train_model,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class responsible for model training."""

    def __init__(self, csv_file_path: str) -> None:
        self.csv_file_path = csv_file_path
        self.hero_data_manager = HeroDataManager()

    def create_matches_dataframe(self) -> pd.DataFrame:
        if not Path(self.csv_file_path).exists():
            msg = f"CSV file not found: {self.csv_file_path}"
            raise FileNotFoundError(msg)

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
    ) -> tuple[DotaDataset, DotaDataset, DotaDataset, torch.Tensor]:
        matches_dataframe = self.create_matches_dataframe()
        pos_weight = compute_pos_weight(matches_dataframe)

        train_dataframe, tmp_dataframe = train_test_split(
            matches_dataframe,
            test_size=0.2,
            stratify=matches_dataframe["win"],
        )
        validation_dataframe, test_dataframe = train_test_split(
            tmp_dataframe,
            test_size=0.5,
            stratify=tmp_dataframe["win"],
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
        return train_dataset, val_dataset, test_dataset, pos_weight

    def main(self) -> None:
        """Model training entrypoint."""
        train_dataset, val_dataset, test_dataset, pos_weight = (  # pylint: disable=W0612 # noqa: RUF059
            self.prepare_datasets()
        )

        training_arguments = self.create_training_arguments(
            train_dataset,
            val_dataset,
        )

        model = self.create_model(self.hero_data_manager.get_heroes_number())
        logger.info(
            f"Model trainable parameters: {count_trainable_params(model)}",
        )
        trained_model, _ = train_model(
            model,
            training_arguments,
        )

        if training_arguments.pos_weight:
            criterion = nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=training_arguments.pos_weight,
            )
        else:
            criterion = nn.BCEWithLogitsLoss(reduction="none")

        metrics, _ = evaluate_model(
            trained_model,
            get_data_loader(
                test_dataset,
                training_arguments.batch_size,
                ShuffleEnum.UNSHUFFLED,
            ),
            criterion,
        )

        logger.info("Test Metrics")
        logger.info(metrics)

        logger.info("--- Confusion Matrix ---")
        header = f"{'':<12}" + "Pred: Loss    " + "Pred: Win     "
        logger.info(header)
        row1 = (
            f"Actual: Loss  {metrics.confusion_matrix[0, 0]:<12}"  # type: ignore[index]
            f"{metrics.confusion_matrix[0, 1]:<13}"  # type: ignore[index]
        )
        row2 = (
            f"Actual: Win   {metrics.confusion_matrix[1, 0]:<12}"  # type: ignore[index]
            f"{metrics.confusion_matrix[1, 1]:<13}"  # type: ignore[index]
        )
        logger.info(row1)
        logger.info(row2)
        logger.info("------------------------")

        torch.save(
            trained_model.state_dict(),
            settings.MODELS_FOLDER_PATH / Path("trained_model.pth"),
        )
        logger.info(
            "Training complete! Model saved as 'trained_model.pth'.",
        )

    def create_training_arguments(
        self,
        train_dataset: DotaDataset,
        val_dataset: DotaDataset,
    ) -> TrainingArguments:
        return TrainingArguments(
            data=TrainingData(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            ),
            pos_weight=None,
            early_stopping_patience=25,
            optimizer_parameters=OptimizerParameters(
                lr=0.067307507600804,
                weight_decay=0.000001,
            ),
            scheduler_parameters=SchedulerParameters(
                factor=0.8,
                scheduler_patience=13,
                threshold=0.0001,
            ),
            decision_weight=15,
            batch_size=256,
        )

    @staticmethod
    def create_model(num_heroes: int) -> RNNWinPredictor:
        return RNNWinPredictor(
            NNParameters(
                num_heroes=num_heroes,
                embedding_dim=64,
                gru_hidden_dim=128,
                num_gru_layers=4,
                dropout_rate=0.746944748796439,
                bidirectional=False,
            ),
        )
