"""
Utility functions for downloading and parsing football match data from
Football‑Data.co.uk.  The site publishes CSV files for each league and
season containing match results, basic statistics and betting odds.

This module provides a high level API for retrieving those CSV files,
parsing them into pandas DataFrames and performing minimal cleaning
before handing them off to the database layer.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Dict, Iterable, Iterator, List, Tuple

import pandas as pd  # type: ignore
import requests

logger = logging.getLogger(__name__)

# Base URL template for Football‑Data CSV files.  The path component
# "mmz4281" appears to be fixed for all historic seasons.  The
# ``{season}`` placeholder expects a two‑year code such as "2223" for
# the 2022/23 season, and ``{league}`` expects the league code (e.g.
# "E0" for Premier League).  See https://www.football-data.co.uk/ for
# details.
BASE_URL: str = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Default leagues to scrape.  Keys correspond to the codes used by
# Football‑Data and values are human‑readable descriptions used only
# for logging.  You can modify this dictionary to include additional
# leagues (e.g. "SP2" for Segunda División).
LEAGUES: Dict[str, str] = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

# Mapping from Football‑Data column names to our preferred schema.  See
# https://www.football-data.co.uk/notes.php for the meaning of each
# abbreviation.  Columns not present in this dictionary will be
# preserved as is.  You can extend this mapping to include additional
# statistics or odds columns.
COLUMN_RENAMES: Dict[str, str] = {
    "Date": "match_date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "full_time_home_goals",
    "FTAG": "full_time_away_goals",
    "FTR": "full_time_result",
    "HTHG": "half_time_home_goals",
    "HTAG": "half_time_away_goals",
    "HTR": "half_time_result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    "AvgH": "avg_odds_home",
    "AvgD": "avg_odds_draw",
    "AvgA": "avg_odds_away",
    "MaxH": "max_odds_home",
    "MaxD": "max_odds_draw",
    "MaxA": "max_odds_away",
}

def get_default_seasons(num_seasons: int = 5) -> List[str]:
    """Return a list of season codes for the most recent ``num_seasons``.

    The season code format is two concatenated two‑digit years, e.g.
    "2122" for 2021/22.  The function uses the current date to
    determine the current season and then works backwards ``num_seasons``
    times.

    Parameters
    ----------
    num_seasons : int
        Number of seasons to return (default 5).

    Returns
    -------
    List[str]
        List of season codes ordered from oldest to newest.
    """
    today = datetime.date.today()
    # Determine current season.  Football seasons start roughly in
    # August and finish in May of the following year.  We consider
    # seasons to run from July of one year to June of the next.  If
    # today's month is >= July (7), we treat the season as starting
    # this calendar year.  Otherwise it started last year.
    if today.month >= 7:
        start_year = today.year
    else:
        start_year = today.year - 1

    seasons: List[str] = []
    for i in range(num_seasons):
        start = start_year - (num_seasons - 1 - i)
        end = start + 1
        seasons.append(f"{str(start)[-2:]}{str(end)[-2:]}")
    return seasons


def download_csv(league_code: str, season_code: str) -> bytes:
    """Download the CSV file for a given league and season.

    Raises an exception if the download fails.  Caller should handle
    exceptions (e.g. network errors, 404) and decide how to proceed.

    Parameters
    ----------
    league_code : str
        Football‑Data league code (e.g. "E0").
    season_code : str
        Two‑year code representing the season (e.g. "2223").

    Returns
    -------
    bytes
        Raw CSV content.
    """
    url = BASE_URL.format(season=season_code, league=league_code)
    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def parse_csv_to_df(csv_content: bytes) -> pd.DataFrame:
    """Parse raw CSV content into a pandas DataFrame.

    The function handles common date formats used by Football‑Data by
    delaying date parsing until after renaming and cleaning.

    Parameters
    ----------
    csv_content : bytes
        Raw CSV content as returned by :func:`download_csv`.

    Returns
    -------
    pandas.DataFrame
        DataFrame representation of the CSV.
    """
    from io import BytesIO  # local import avoids unnecessary import when unused
    return pd.read_csv(BytesIO(csv_content))


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalise a dataframe according to our schema.

    This function:
    * Renames columns using :data:`COLUMN_RENAMES`.
    * Converts the ``match_date`` column to ISO date strings.
    * Adds any missing columns defined in :data:`COLUMN_RENAMES` with
      null values (so all tables share the same schema regardless of
      league or season).

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe as parsed from the CSV.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with uniform column names and types.
    """
    # Rename columns.  Only those present in df will be renamed.
    df = df.rename(columns=COLUMN_RENAMES)

    # Convert match_date to datetime.date objects.  Football‑Data uses
    # various formats (e.g. "20/08/2022" for DD/MM/YYYY, or
    # "30/08/92").  We try parsing dayfirst.  Errors='coerce' will
    # produce NaT for unparsable dates.
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(
            df['match_date'], errors='coerce', dayfirst=True
        ).dt.date

    # Ensure all expected columns exist.  For any missing key in
    # COLUMN_RENAMES values, create a column filled with None.
    for canonical_name in COLUMN_RENAMES.values():
        if canonical_name not in df.columns:
            df[canonical_name] = None

    return df


def fetch_season_data(
    leagues: Dict[str, str] | None = None,
    seasons: List[str] | None = None,
) -> Iterator[Tuple[str, str, pd.DataFrame]]:
    """Yield cleaned dataframes for each league/season combination.

    Parameters
    ----------
    leagues : dict, optional
        Mapping of league codes to descriptive names.  If omitted,
        defaults to :data:`LEAGUES`.
    seasons : list of str, optional
        List of season codes (e.g. ["2122", "2223", ...]).  If
        omitted, defaults to the latest five seasons as returned by
        :func:`get_default_seasons`.

    Yields
    ------
    tuple of (league_code, season_code, DataFrame)
        A cleaned dataframe for the given league and season.  The
        dataframe includes all columns defined in
        :data:`COLUMN_RENAMES`, plus any others present in the source
        CSV.
    """
    leagues = leagues or LEAGUES
    seasons = seasons or get_default_seasons()

    for season in seasons:
        for league_code, league_name in leagues.items():
            try:
                content = download_csv(league_code, season)
            except Exception as exc:
                logger.warning(
                    "Failed to download %s %s: %s", league_code, season, exc
                )
                continue
            df_raw = parse_csv_to_df(content)
            df_clean = clean_dataframe(df_raw)
            # Add identifying metadata columns for later use
            df_clean['league'] = league_code
            df_clean['season'] = season
            yield (league_code, season, df_clean)
