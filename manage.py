import importlib
import logging
import sys

import click

import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)
logger = logging.getLogger(__name__)


class DotaPickerError(Exception):
    """Main application exception."""


@click.group()
def cli() -> None:
    pass


@cli.command()
def fetch_personal_matches() -> None:
    """Fetch personal match history from OpenDota API."""
    try:
        module = importlib.import_module(
            "dota_hero_picker.fetch_personal_matches",
        )
        module.main(settings.PERSONAL_DOTA_MATCHES_PATH, settings.ACCOUNT_ID)
    except DotaPickerError as critical_error:
        logger.exception("Critical Error")
        click.echo(f"Error: {critical_error}")


@cli.command()
def tune() -> None:
    """Run Optuna multi-objective hyperparameter search."""
    try:
        module = importlib.import_module(
            "dota_hero_picker.tune",
        )
        module.main(settings.PERSONAL_DOTA_MATCHES_PATH)
    except DotaPickerError as critical_error:
        logger.exception("Critical Error")
        click.echo(f"Error: {critical_error}")


@cli.command()
def train() -> None:
    """Train and calibrate the Pareto champion model from tuning."""
    module = importlib.import_module(
        "dota_hero_picker.train",
    )
    module.main()


@cli.command()
def param_convergence() -> None:
    """Check hyperparameter convergence and ranges across top trials."""
    module = importlib.import_module(
        "dota_hero_picker.param_convergence",
    )
    module.main()


if __name__ == "__main__":
    cli()
