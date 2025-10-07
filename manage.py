import importlib
import logging
import sys

import click

import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class DotaPickerError(Exception):
    """Main applicaiton exception."""


@click.group()
def cli() -> None:
    pass


@cli.command()
def calculate_matchups() -> None:
    try:
        module = importlib.import_module("dota_hero_picker.calculate_matchups")
        module.main(
            settings.PUBLIC_DOTA_MATCHES_PATH,
            settings.MATCHUPS_STATISTICS_PATH,
        )
    except DotaPickerError as critical_error:
        logger.exception("Critical Error")
        click.echo(f"Error: {critical_error}")


@cli.command()
def load_public_matches() -> None:
    try:
        module = importlib.import_module(
            "dota_hero_picker.load_public_matches",
        )
        module.main(settings.ACCOUNT_ID, settings.PUBLIC_DOTA_MATCHES_PATH)
    except DotaPickerError as critical_error:
        logger.exception("Critical Error")
        click.echo(f"Error: {critical_error}")


@cli.command()
def train_model() -> None:
    try:
        module = importlib.import_module("dota_hero_picker.train_model")
        module.main(
            settings.PERSONAL_DOTA_MATCHES_PATH,
        )
    except DotaPickerError as critical_error:
        logger.exception("Critical Error")
        click.echo(f"Error: {critical_error}")


@cli.command()
def load_personal_matches() -> None:
    try:
        module = importlib.import_module(
            "dota_hero_picker.load_personal_matches",
        )
        module.main(settings.PERSONAL_DOTA_MATCHES_PATH, settings.ACCOUNT_ID)
    except DotaPickerError as critical_error:
        logger.exception("Critical Error")
        click.echo(f"Error: {critical_error}")


if __name__ == "__main__":
    cli()
