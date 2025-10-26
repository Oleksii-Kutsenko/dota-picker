import csv
import datetime
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import requests

from manage import DotaPickerError

HEROES_FILE = "heroes.json"
API_MATCHES_ENDPOINT = "https://api.opendota.com/api/players/{}/matches"
API_MATCH_DETAILS_ENDPOINT = "https://api.opendota.com/api/matches/{}"
API_HEROES_ENDPOINT = "https://api.opendota.com/api/heroes"
CURRENT_PATCH_START = datetime.datetime(2025, 8, 15, tzinfo=datetime.UTC)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def read_existing_matches(csv_path: str) -> tuple[set[int], int | None]:
    if not Path(csv_path).exists():
        return set(), None
    match_ids = set()
    max_start_time = None
    with Path(csv_path).open(
        newline="",
        encoding="utf-8",
    ) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header
        for row in reader:
            match_ids.add(int(row[-2]))
            start_time = int(row[-1])
            if max_start_time is None or start_time > max_start_time:
                max_start_time = start_time
    return match_ids, max_start_time


def prepare_picks(
    picks_bans: list[dict[str, Any]] | None,
    actually_picked_heroes: dict[int, list[int]],
    your_team: int,
) -> tuple[list[int], list[int]]:
    opp_team = 1 - your_team

    if not picks_bans:
        team_picks = actually_picked_heroes[your_team]
        opp_picks = actually_picked_heroes[opp_team]
    else:
        all_picks = [
            p
            for p in picks_bans
            if p["is_pick"]
            and any(
                p["hero_id"] == h
                for team in actually_picked_heroes.values()
                for h in team
            )
        ]
        all_picks_sorted = sorted(all_picks, key=lambda pick: pick["order"])

        team_picks = [
            p["hero_id"] for p in all_picks_sorted if p["team"] == your_team
        ]
        opp_picks = [
            p["hero_id"] for p in all_picks_sorted if p["team"] == opp_team
        ]

        team_missing = [
            h for h in actually_picked_heroes[your_team] if h not in team_picks
        ]
        opp_missing = [
            h for h in actually_picked_heroes[opp_team] if h not in opp_picks
        ]

        team_picks = team_missing + team_picks
        opp_picks = opp_missing + opp_picks
    return team_picks, opp_picks


PICKS_NUMBER = 10


class MatchDict(TypedDict):
    """Match dict."""

    players: list[dict[str, int]] | None
    picks_bans: list[dict[str, int]] | None
    radiant_win: int


def parse_draft(match: MatchDict, account_id: int) -> dict[str, Any] | None:
    players = match.get("players")
    if players is None:
        return None
    picks_bans = match.get("picks_bans")

    your_player = None
    for player in players:
        if player.get("account_id") == account_id:
            your_player = player
            break
    else:
        return None

    your_team = 0 if your_player["isRadiant"] else 1

    radiant_win = match["radiant_win"]
    win = 1 if radiant_win == (your_team == 0) else 0

    your_hero_id = your_player["hero_id"]

    actually_picked_heroes = defaultdict(list)
    for player in players:
        actually_picked_heroes[player["team_number"]].append(player["hero_id"])

    if not players and not picks_bans:
        return None

    team_picks, opp_picks = prepare_picks(
        picks_bans,
        actually_picked_heroes,
        your_team,
    )

    if len(team_picks) + len(opp_picks) != PICKS_NUMBER:
        msg = "Picks parsing failed"
        raise RuntimeError(msg)

    return {
        "team_picks": team_picks,
        "opponent_picks": opp_picks,
        "picked_hero": your_hero_id,
        "win": win,
        "match_id": match.get("match_id"),
        "start_time": match.get("start_time", 0),
    }


def fetch_match_details(
    match_ids: list[int],
    account_id: int,
    delay_seconds: float = 1.2,
) -> list[dict[str, Any]]:
    detailed_decisions = []
    for match_id in match_ids:
        response = requests.get(
            API_MATCH_DETAILS_ENDPOINT.format(match_id),
            timeout=5,
        )
        if response.status_code == HTTP_OK:
            match = response.json()
            data = parse_draft(match, account_id)
            if data:
                detailed_decisions.append(data)
            else:
                logger.info(f"Match #{match['match_id']} failed to parse.")
        else:
            logger.info(
                f"Failed to fetch details for match "
                f"{match_id}: {response.status_code}",
            )
        logger.info(f"Match #{match_id} has been processed")

        time.sleep(delay_seconds)
    return detailed_decisions


def save_to_csv(csv_path: str, new_decisions: list[dict[str, Any]]) -> None:
    if Path(csv_path).exists():
        df_existing = pd.read_csv(csv_path)
    else:
        df_existing = pd.DataFrame(
            columns=[
                "team_picks",
                "opponent_picks",
                "picked_hero",
                "win",
                "match_id",
                "start_time",
            ],
        )

    data_rows = [
        {
            "team_picks": str(dec["team_picks"]),
            "opponent_picks": str(dec["opponent_picks"]),
            "picked_hero": dec["picked_hero"],
            "win": dec["win"],
            "match_id": dec["match_id"],
            "start_time": dec["start_time"],
        }
        for dec in new_decisions
    ]
    df_new = pd.DataFrame(data_rows)

    # Append new data and drop duplicates
    # (adjust subset if you want uniqueness on specific columns)
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    dropped_df_all = df_all.drop_duplicates()

    # Sort by start_time ascending
    sorted_df_all = dropped_df_all.sort_values(by="start_time")

    sorted_df_all.to_csv(csv_path, index=False)


HTTP_OK = 200


def fetch_and_save_new_decisions(
    personal_dota_matches_path: str,
    account_id: int,
) -> int:
    existing_match_ids, _ = read_existing_matches(personal_dota_matches_path)

    params = {}
    params["date"] = (
        datetime.datetime.now(tz=datetime.UTC) - CURRENT_PATCH_START
    ).days + 1

    response = requests.get(
        API_MATCHES_ENDPOINT.format(account_id),
        params=params,
        timeout=5,
    )
    if response.status_code != HTTP_OK:
        msg = f"API error: {response.status_code}"
        raise DotaPickerError(msg)
    data = response.json()

    new_match_ids = [
        m["match_id"]
        for m in data
        if m["match_id"] not in existing_match_ids
        and datetime.datetime.fromtimestamp(m["start_time"], tz=datetime.UTC)
        >= CURRENT_PATCH_START
    ]

    if not new_match_ids:
        logger.info("No new current-patch matches.")
        return 0

    detailed = fetch_match_details(new_match_ids, account_id)
    Path(personal_dota_matches_path).parent.mkdir(exist_ok=True)
    save_to_csv(personal_dota_matches_path, detailed)
    logger.info(f"Saved {len(detailed)} new decisions.")
    return len(detailed)


def main(personal_dota_matches_path: str, account_id: int) -> None:
    num_saved = fetch_and_save_new_decisions(
        personal_dota_matches_path,
        account_id,
    )
    logger.info(f"Total new matches saved: {num_saved}")
