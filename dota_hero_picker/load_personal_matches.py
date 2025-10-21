import csv
import json
import logging
import os
from collections import defaultdict
import sys
import time
from datetime import datetime

import pandas as pd
import requests

from manage import DotaPickerError

HEROES_FILE = "heroes.json"
API_MATCHES_ENDPOINT = "https://api.opendota.com/api/players/{}/matches"
API_MATCH_DETAILS_ENDPOINT = "https://api.opendota.com/api/matches/{}"
API_HEROES_ENDPOINT = "https://api.opendota.com/api/heroes"
CURRENT_PATCH_START = datetime(2025, 8, 15)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def get_hero_data(local_file: str = HEROES_FILE) -> list[dict[str, str]]:
    if os.path.exists(local_file):
        with open(local_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        response = requests.get(API_HEROES_ENDPOINT, timeout=5)
        if response.status_code != 200:
            raise DotaPickerError(
                f"Failed to fetch heroes: {response.status_code}"
            )
        data = response.json()
        with open(local_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data


def load_hero_map():
    heroes = get_hero_data()
    return {hero["id"]: hero["localized_name"] for hero in heroes}


def read_existing_matches(csv_path):
    if not os.path.exists(csv_path):
        return set(), None
    match_ids = set()
    max_start_time = None
    with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header
        for row in reader:
            match_ids.add(int(row[-2]))
            start_time = int(row[-1])
            if max_start_time is None or start_time > max_start_time:
                max_start_time = start_time
    return match_ids, max_start_time


def parse_draft(match, account_id):
    players = match.get("players")
    picks_bans = match.get("picks_bans")

    your_player = None
    for player in players:
        if player.get("account_id") == account_id:
            your_player = player
            break
    else:
        return None

    your_team = 0 if your_player.get("isRadiant") else 1
    win = 1 if match.get("radiant_win") == (your_team == 0) else 0

    if not picks_bans:
        if not players:
            return None
        team_picks = []
        opp_picks = []
        for player in players:
            if player["team_number"] == your_team:
                team_picks.append(player["hero_id"])
            else:
                opp_picks.append(player["hero_id"])

        your_hero_id = your_player.get("hero_id")

        return {
            "team_picks": team_picks,
            "opponent_picks": opp_picks,
            "picked_hero": your_hero_id,
            "win": win,
            "match_id": match["match_id"],
            "start_time": match.get("start_time", 0),
        }

    team_picks = []
    opp_picks = []

    actually_picked_heroes = defaultdict(list)
    for player in players:
        actually_picked_heroes[player["team_number"]].append(player["hero_id"])
    all_picks_sorted = sorted(
        [
            p
            for p in picks_bans
            if p.get("is_pick")
            and p["hero_id"] in actually_picked_heroes.values()
        ],
        key=lambda p: p["order"],
    )
    for pick in all_picks_sorted:
        hero_id = pick["hero_id"]
        if pick["team"] == your_team:
            team_picks.append(hero_id)
        else:
            opp_picks.append(hero_id)

    if len(team_picks) != 5 or len(opp_picks) != 5:
        breakpoint()

    your_hero_id = your_player.get("hero_id")

    return {
        "team_picks": team_picks,
        "opponent_picks": opp_picks,
        "picked_hero": your_hero_id,
        "win": win,
        "match_id": match["match_id"],
        "start_time": match.get("start_time", 0),
    }


def fetch_match_details(match_ids, account_id, delay_seconds=1.1):
    detailed_decisions = []
    for match_id in match_ids:
        response = requests.get(
            API_MATCH_DETAILS_ENDPOINT.format(match_id), timeout=5
        )
        if response.status_code == 200:
            match = response.json()
            data = parse_draft(match, account_id)
            if data:
                detailed_decisions.append(data)
        else:
            print(
                f"Failed to fetch details for match {match_id}: {response.status_code}"
            )

        time.sleep(delay_seconds)
    return detailed_decisions


def save_to_csv(csv_path, new_decisions):
    if os.path.exists(csv_path):
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
            ]
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

    # Append new data and drop duplicates (adjust subset if you want uniqueness on specific columns)
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all.drop_duplicates(inplace=True)

    # Sort by start_time ascending
    df_all.sort_values(by="start_time", inplace=True)

    # Write combined data back to CSV without index
    df_all.to_csv(csv_path, index=False)


def fetch_and_save_new_decisions(personal_dota_matches_path, account_id):
    existing_match_ids, _ = read_existing_matches(personal_dota_matches_path)

    params = {}
    params["date"] = (datetime.today() - CURRENT_PATCH_START).days + 1

    response = requests.get(
        API_MATCHES_ENDPOINT.format(account_id), params=params, timeout=5
    )
    if response.status_code != 200:
        raise DotaPickerError(f"API error: {response.status_code}")
    data = response.json()

    new_match_ids = [
        m["match_id"]
        for m in data
        if m["match_id"] not in existing_match_ids
        and datetime.fromtimestamp(m["start_time"]) >= CURRENT_PATCH_START
    ]

    if not new_match_ids:
        print("No new current-patch matches.")
        return 0

    detailed = fetch_match_details(new_match_ids, account_id)
    save_to_csv(personal_dota_matches_path, detailed)
    print(f"Saved {len(detailed)} new decisions.")
    return len(detailed)


def main(personal_dota_matches_path, account_id):
    num_saved = fetch_and_save_new_decisions(
        personal_dota_matches_path,
        account_id,
    )
    logger.info(f"Total new matches saved: {num_saved}")
