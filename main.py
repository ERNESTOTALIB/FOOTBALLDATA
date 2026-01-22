"""
Entry point for the football data ingestion pipeline.

This script orchestrates the workflow of downloading historical match
data from Football‑Data.co.uk, cleaning it and storing it in a
PostgreSQL database.  It can be run locally or scheduled as a cron
job on a hosted platform such as Render.  See README.md for usage
instructions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from data_fetcher import (
    COLUMN_RENAMES,
    LEAGUES,
    fetch_season_data,
    get_default_seasons,
)
from db_manager import init_db, insert_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def row_to_dicts(row: pd.Series) -> Tuple[Dict, Dict, Dict]:
    """Convert a DataFrame row into dictionaries for each table.

    This helper extracts the relevant fields for the ``matches``,
    ``match_stats`` and ``odds_1x2`` tables.  It also handles NaN
    values by converting them to None (psycopg2 will translate None
    into SQL NULL).

    Parameters
    ----------
    row : pandas.Series
        A single row from the cleaned dataframe.

    Returns
    -------
    tuple of (match_dict, stats_dict, odds_dict)
        Three dictionaries ready for insertion into the database.
    """
    def _none_if_nan(value):
        # pandas may represent missing data as numpy.nan or NaT; both evaluate
        # to True with pandas.isna
        return None if pd.isna(value) else value

    match = {
        "league": _none_if_nan(row.get("league")),
        "season": _none_if_nan(row.get("season")),
        "match_date": _none_if_nan(row.get("match_date")),
        "home_team": _none_if_nan(row.get("home_team")),
        "away_team": _none_if_nan(row.get("away_team")),
        "full_time_home_goals": _none_if_nan(row.get("full_time_home_goals")),
        "full_time_away_goals": _none_if_nan(row.get("full_time_away_goals")),
        "full_time_result": _none_if_nan(row.get("full_time_result")),
        "half_time_home_goals": _none_if_nan(row.get("half_time_home_goals")),
        "half_time_away_goals": _none_if_nan(row.get("half_time_away_goals")),
        "half_time_result": _none_if_nan(row.get("half_time_result")),
    }
    stats = {
        "home_shots": _none_if_nan(row.get("home_shots")),
        "away_shots": _none_if_nan(row.get("away_shots")),
        "home_shots_on_target": _none_if_nan(row.get("home_shots_on_target")),
        "away_shots_on_target": _none_if_nan(row.get("away_shots_on_target")),
        "home_corners": _none_if_nan(row.get("home_corners")),
        "away_corners": _none_if_nan(row.get("away_corners")),
        "home_yellows": _none_if_nan(row.get("home_yellows")),
        "away_yellows": _none_if_nan(row.get("away_yellows")),
        "home_reds": _none_if_nan(row.get("home_reds")),
        "away_reds": _none_if_nan(row.get("away_reds")),
    }
    odds = {
        "avg_odds_home": _none_if_nan(row.get("avg_odds_home")),
        "avg_odds_draw": _none_if_nan(row.get("avg_odds_draw")),
        "avg_odds_away": _none_if_nan(row.get("avg_odds_away")),
        "max_odds_home": _none_if_nan(row.get("max_odds_home")),
        "max_odds_draw": _none_if_nan(row.get("max_odds_draw")),
        "max_odds_away": _none_if_nan(row.get("max_odds_away")),
    }
    return match, stats, odds


def main(leagues: Dict[str, str], seasons: List[str]) -> None:
    """Main orchestration function.

    Downloads and stores football data for the specified leagues and
    seasons.  It first ensures the database schema exists, then
    processes each league/season combination in turn.  Matches that
    already exist in the database are skipped automatically during
    insertion.

    Parameters
    ----------
    leagues : dict
        Mapping of league codes to descriptions.
    seasons : list of str
        List of season codes to download.
    """
    logger.info("Starting football data ingestion...")
    # Ensure DB tables exist
    init_db()
    total_inserted_matches = 0

    for league_code, season_code, df in fetch_season_data(leagues, seasons):
        logger.info(
            "Processing %s %s – %d rows", league_code, season_code, len(df)
        )
        matches: List[Dict] = []
        stats: List[Dict] = []
        odds: List[Dict] = []
        for _, row in df.iterrows():
            m, s, o = row_to_dicts(row)
            matches.append(m)
            stats.append(s)
            odds.append(o)
        insert_data(matches, stats, odds)
        total_inserted_matches += len(matches)
        logger.info(
            "Completed %s %s – attempted to insert %d matches", league_code, season_code, len(matches)
        )

    logger.info("Finished ingestion. Processed %d records in total.", total_inserted_matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download football results and load them into a PostgreSQL database."
    )
    parser.add_argument(
        "--leagues",
        nargs="*",
        help=(
            "List of league codes to download (default: %(default)s). "
            "Use codes as defined on Football-Data (e.g. E0, SP1)."
        ),
        default=list(LEAGUES.keys()),
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        help=(
            "List of season codes to download (default: latest 5 seasons). "
            "Season codes are two pairs of digits, e.g. 2223 for 2022/23."
        ),
        default=None,
    )
    args = parser.parse_args()
    # Build the leagues mapping from provided codes
    selected_leagues = {code: LEAGUES.get(code, code) for code in args.leagues}
    selected_seasons = args.seasons if args.seasons else get_default_seasons()
    try:
        main(selected_leagues, selected_seasons)
    except Exception as exc:
        logger.exception("Error during ingestion: %s", exc)
        sys.exit(1)
