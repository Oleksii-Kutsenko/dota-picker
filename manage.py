import importlib
import logging
import sys
from sys import exc_info

import click

import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class DotaPickerException(Exception):
    pass


@click.group()
def cli():
    pass


@cli.command()
def calculate_matchups():
    try:
        module = importlib.import_module("dota_hero_picker.calculate_matchups")
        module.main(
            settings.PUBLIC_DOTA_MATCHES_PATH,
            settings.MATCHUPS_STATISTICS_PATH,
        )
    except DotaPickerException as critical_error:
        logger.error("Critical Error", exc_info=True)
        click.echo(f"Error: {critical_error}")


@cli.command()
def load_public_matches():
    try:
        module = importlib.import_module(
            "dota_hero_picker.load_public_matches"
        )
        module.main(settings.ACCOUNT_ID, settings.PUBLIC_DOTA_MATCHES_PATH)
    except DotaPickerException as critical_error:
        logger.error("Critical Error", exc_info=True)
        click.echo(f"Error: {critical_error}")


@cli.command()
def train_model():
    try:
        module = importlib.import_module("dota_hero_picker.train_model")
        module.main(
            settings.PERSONAL_DOTA_MATCHES_PATH,
            settings.MATCHUPS_STATISTICS_PATH,
        )
    except DotaPickerException as critical_error:
        logger.error("Critical Error", exc_info=True)
        click.echo(f"Error: {critical_error}")


@cli.command()
def load_personal_matches():
    try:
        module = importlib.import_module(
            "dota_hero_picker.load_personal_matches",
        )
        module.main(settings.PERSONAL_DOTA_MATCHES_PATH, settings.ACCOUNT_ID)
    except DotaPickerException as critical_error:
        logger.exception("Critical Error")
        click.echo(f"Error: {critical_error}")


if __name__ == "__main__":
    cli()
