import csv
import json
import os
import time
from datetime import datetime

import pandas as pd
import requests

HEROES_FILE = "heroes.json"
API_MATCHES_ENDPOINT = "https://api.opendota.com/api/players/{}/matches"
API_MATCH_DETAILS_ENDPOINT = "https://api.opendota.com/api/matches/{}"
API_HEROES_ENDPOINT = "https://api.opendota.com/api/heroes"
CURRENT_PATCH_START = datetime(2025, 8, 15)


def get_hero_data(local_file=HEROES_FILE):
    if os.path.exists(local_file):
        with open(local_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        response = requests.get(API_HEROES_ENDPOINT)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch heroes: {response.status_code}")
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


def parse_draft(match, account_id, hero_map):
    picks_bans = match.get("picks_bans", [])
    if not picks_bans:
        return None

    players = match.get("players", [])
    your_player = next(
        (p for p in players if p.get("account_id") == account_id), None
    )
    if not your_player:
        return None

    your_hero_id = your_player.get("hero_id")
    your_team = 0 if your_player.get("isRadiant") else 1

    actually_picked_heroes = {player["hero_id"] for player in players}
    all_picks = [p for p in picks_bans if p["is_pick"]]

    team_picks_sorted = sorted(
        [p for p in all_picks if p["team"] == your_team],
        key=lambda x: x["order"],
    )
    opponent_picks_sorted = sorted(
        [p for p in all_picks if p["team"] != your_team],
        key=lambda x: x["order"],
    )

    my_pick_index = next(
        (
            i
            for i, p in enumerate(team_picks_sorted)
            if p["hero_id"] == your_hero_id
        ),
        None,
    )
    if my_pick_index is None:
        return None

    team_picks = [
        hero_map[p["hero_id"]]
        for p in team_picks_sorted[:my_pick_index]
        if p["hero_id"] in actually_picked_heroes
    ]

    # Lookup limits based on my_pick_index (0-4 picks; caps at 4)
    opponent_pick_limits = [0, 0, 2, 2, 4]
    limit = opponent_pick_limits[min(my_pick_index, 4)]

    opponent_picks = [
        hero_map[p["hero_id"]]
        for p in opponent_picks_sorted[:limit]
        if p["hero_id"] in actually_picked_heroes
    ]

    win = 1 if match.get("radiant_win") == (your_team == 0) else 0

    return {
        "full_team_picks": [
            hero_map[pick["hero_id"]] for pick in team_picks_sorted
        ],
        "team_picks": team_picks,
        "full_opponent_picks": [
            hero_map[pick["hero_id"]] for pick in opponent_picks_sorted
        ],
        "opponent_picks": opponent_picks,
        "picked_hero": hero_map.get(your_hero_id, "Unknown"),
        "win": win,
        "match_id": match["match_id"],
        "start_time": match.get("start_time", 0),
    }


def fetch_match_details(match_ids, account_id, hero_map, delay_seconds=1.1):
    detailed_decisions = []
    for i, match_id in enumerate(match_ids):
        response = requests.get(API_MATCH_DETAILS_ENDPOINT.format(match_id))
        if response.status_code == 200:
            match = response.json()
            data = parse_draft(match, account_id, hero_map)
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
                "full_team_picks",
                "team_picks",
                "full_opponent_picks",
                "opponent_picks",
                "picked_hero",
                "win",
                "match_id",
                "start_time",
            ]
        )

    # Convert new decisions to DataFrame
    data_rows = [
        {
            "full_team_picks": ",".join(dec["full_team_picks"])
            if dec["full_team_picks"]
            else None,
            "team_picks": ",".join(dec["team_picks"]),
            "full_opponent_picks": ",".join(dec["full_opponent_picks"])
            if dec["full_opponent_picks"]
            else None,
            "opponent_picks": ",".join(dec["opponent_picks"]),
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


def fetch_and_save_new_decisions(account_id):
    hero_map = load_hero_map()
    existing_match_ids, max_start_time = read_existing_matches(CSV_FILE)

    params = {}
    if max_start_time:
        params["date"] = (
            datetime.today() - datetime.fromtimestamp(max_start_time)
        ).days + 1
    else:
        params["date"] = (datetime.today() - CURRENT_PATCH_START).days + 1

    response = requests.get(
        API_MATCHES_ENDPOINT.format(account_id), params=params
    )
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code}")
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

    detailed = fetch_match_details(new_match_ids, account_id, hero_map)
    save_to_csv(CSV_FILE, detailed)
    print(f"Saved {len(detailed)} new decisions.")
    return len(detailed)


if __name__ == "__main__":
    account_id = 381437537
    num_saved = fetch_and_save_new_decisions(account_id)
    print(f"Total new matches saved: {num_saved}")
