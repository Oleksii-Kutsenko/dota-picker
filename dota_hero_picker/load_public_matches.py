import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def query_explorer(sql_query) -> list[dict[str, str]]:
    base_url = "https://api.opendota.com/api/explorer"
    params = {"sql": sql_query}
    response = requests.get(base_url, params=params)
    logger.info(f"Received response: {response.status_code}")
    return response.json()["rows"]


def get_sql(rank_tier) -> str:
    return f"""
    SELECT *
    FROM public_matches
    WHERE lobby_type = 7
    AND avg_rank_tier BETWEEN {rank_tier - 10} AND {rank_tier + 10}
    ORDER BY start_time DESC
    LIMIT 100000;
    """


def main(account_id, matches_path) -> None:
    resp = requests.get(f"https://api.opendota.com/api/players/{account_id}")
    rank_tier = resp.json().get("rank_tier")
    sql = get_sql(rank_tier)

    matches_rows = query_explorer(sql)

    matches_dataframe = pd.DataFrame(matches_rows)
    matches_dataframe = matches_dataframe[
        ["match_id", "radiant_win", "radiant_team", "dire_team"]
    ]
    logger.info(f"Matches Dataframe Size {len(matches_dataframe)}.")

    radiant = matches_dataframe.explode("radiant_team").rename(
        columns={"radiant_team": "hero_id"}
    )
    radiant["is_radiant"] = True
    radiant = radiant.drop(columns=["dire_team"])

    dire = matches_dataframe.explode("dire_team").rename(
        columns={"dire_team": "hero_id"}
    )
    dire["is_radiant"] = False
    dire = dire.drop(columns=["radiant_team"])

    resulting_df = pd.concat([radiant, dire], ignore_index=True)
    logger.info(f"Resulting Dataframe Size {len(resulting_df)}.")

    resulting_df.to_csv(matches_path, index=False)
