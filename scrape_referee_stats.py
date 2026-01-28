#!/usr/bin/env python3
"""
Scrape referee statistics from WhoScored and store them in a PostgreSQL database.

This script uses `requests_html` to render the JavaScript-driven pages from
WhoScored. It collects general referee statistics for a given competition
(e.g., Premier League 2025/2026) and, optionally, per-team referee statistics.

The script expects the following environment variables to be defined:

- WHOSCORED_BASE_URL: The base URL for the WhoScored referee statistics page
  (e.g., https://www.whoscored.com/regions/252/tournaments/2/seasons/10743/stages/24533/refereestatistics/england-premier-league-2025-2026)
- DATABASE_URL: A PostgreSQL connection string for Supabase or other
  Postgres-compatible database.

Dependencies:

- requests_html
- pandas
- psycopg2-binary

You can install these with:

    pip install requests_html pandas psycopg2-binary

Note: requests_html internally uses pyppeteer to execute JavaScript. On the
first run it will download a headless Chromium browser. Ensure your
environment allows this.

"""
import os
import re
import time
import logging
from datetime import datetime

import pandas as pd
from requests_html import HTMLSession
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constants for column names mapping from WhoScored to database-friendly names
COLUMN_MAP = {
    "Referee": "referee_name",
    "Apps": "apps",
    "Fouls pg": "fouls_per_game",
    "Fouls/Tackles": "fouls_tackles_ratio",
    "Pen pg": "penalties_per_game",
    "Yel pg": "yellow_cards_per_game",
    "Yel": "yellow_cards_total",
    "Red pg": "red_cards_per_game",
    "Red": "red_cards_total",
}




def fetch_referee_table(url: str, session: HTMLSession) -> pd.DataFrame:
    """Fetch a referee statistics table from WhoScored.

    Args:
        url: The full URL to the WhoScored page (e.g., with ?show=home).
        session: An HTMLSession instance to reuse connections and cookies.

    Returns:
        A pandas DataFrame containing the referee statistics, with columns
        mapped according to COLUMN_MAP. A new column named 'source_url' will
        contain the URL of the page that was scraped.
    """
    logger.info(f"Fetching {url}")
    response = session.get(url)
    # Render JavaScript. Increase timeout if needed.
    response.html.render(timeout=30)
    # Extract tables via pandas. We search for the table that contains the
    # 'Referee' header.
    tables = pd.read_html(response.html.raw_html)
    df = None
    for tbl in tables:
        if "Referee" in tbl.columns:
            df = tbl.copy()
            break
    if df is None:
        raise ValueError(f"No referee table found on {url}")
    # Drop rank index (first column if integer and unnamed)
    if df.columns[0] == 0 or df.iloc[:, 0].dtype != 'object':
        df = df.drop(columns=df.columns[0])
    # Rename columns to DB-friendly names
    df = df.rename(columns=COLUMN_MAP)
    # Add source URL for debugging/tracking
    df['source_url'] = url
    return df



def scrape_referee_statistics(base_url: str) -> pd.DataFrame:
    """Scrape referee statistics for overall, home, and away contexts.

    Args:
        base_url: The base URL for the referee statistics page (without
            ?show=...) for the desired competition and season.

    Returns:
        A concatenated DataFrame containing statistics for overall, home,
        and away contexts. A new column 'context' indicates which view the
        row came from: 'overall', 'home', or 'away'.
    """
    session = HTMLSession()
    data_frames = []
    contexts = ['overall', 'home', 'away']
    for ctx in contexts:
        url = f"{base_url}?show={ctx}" if ctx != 'overall' else base_url
        try:
            df = fetch_referee_table(url, session)
            df['context'] = ctx
            data_frames.append(df)
        except Exception as exc:
            logger.warning(f"Failed to scrape {ctx} view: {exc}")
    if not data_frames:
        raise ValueError("No referee statistics were scraped.")
    all_df = pd.concat(data_frames, ignore_index=True)
    # Standardize names: lower-case referee names, strip whitespace
    all_df['referee_name'] = all_df['referee_name'].str.strip()
    return all_df



def upsert_referee_stats(df: pd.DataFrame, conn_str: str, league: str, season: str) -> None:
    """Upsert referee statistics into the database.

    Args:
        df: DataFrame containing referee stats with columns matching COLUMN_MAP
            mapped values plus 'context'.
        conn_str: PostgreSQL connection string.
        league: The league name/code (e.g., 'Premier League').
        season: The season string (e.g., '2025/2026').
    """
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    # Ensure table exists
    create_sql = """
    CREATE TABLE IF NOT EXISTS referee_stats (
        id SERIAL PRIMARY KEY,
        league TEXT NOT NULL,
        season TEXT NOT NULL,
        referee_name TEXT NOT NULL,
        context TEXT NOT NULL,
        apps INT,
        fouls_per_game NUMERIC,
        fouls_tackles_ratio NUMERIC,
        penalties_per_game NUMERIC,
        yellow_cards_per_game NUMERIC,
        yellow_cards_total INT,
        red_cards_per_game NUMERIC,
        red_cards_total INT,
        source_url TEXT,
        updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (league, season, referee_name, context)
    );
    """
    cur.execute(create_sql)
    # Prepare records for upsert
    records = []
    for _, row in df.iterrows():
        record = (
            league,
            season,
            row['referee_name'],
            row['context'],
            int(row['apps']) if not pd.isna(row['apps']) else None,
            float(row['fouls_per_game']) if not pd.isna(row['fouls_per_game']) else None,
            float(row['fouls_tackles_ratio']) if not pd.isna(row['fouls_tackles_ratio']) else None,
            float(row['penalties_per_game']) if not pd.isna(row['penalties_per_game']) else None,
            float(row['yellow_cards_per_game']) if not pd.isna(row['yellow_cards_per_game']) else None,
            int(row['yellow_cards_total']) if not pd.isna(row['yellow_cards_total']) else None,
            float(row['red_cards_per_game']) if not pd.isna(row['red_cards_per_game']) else None,
            int(row['red_cards_total']) if not pd.isna(row['red_cards_total']) else None,
            row['source_url'],
        )
        records.append(record)
    insert_sql = """
    INSERT INTO referee_stats (
        league, season, referee_name, context, apps, fouls_per_game,
        fouls_tackles_ratio, penalties_per_game, yellow_cards_per_game,
        yellow_cards_total, red_cards_per_game, red_cards_total, source_url
    ) VALUES %s
    ON CONFLICT (league, season, referee_name, context) DO UPDATE SET
        apps = EXCLUDED.apps,
        fouls_per_game = EXCLUDED.fouls_per_game,
        fouls_tackles_ratio = EXCLUDED.fouls_tackles_ratio,
        penalties_per_game = EXCLUDED.penalties_per_game,
        yellow_cards_per_game = EXCLUDED.yellow_cards_per_game,
        yellow_cards_total = EXCLUDED.yellow_cards_total,
        red_cards_per_game = EXCLUDED.red_cards_per_game,
        red_cards_total = EXCLUDED.red_cards_total,
        source_url = EXCLUDED.source_url,
        updated_at = CURRENT_TIMESTAMP;
    """
    execute_values(cur, insert_sql, records)
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Upserted {len(records)} referee stats records")



def main():
    base_url = os.getenv("WHOSCORED_BASE_URL")
    conn_str = os.getenv("DATABASE_URL")
    league = os.getenv("LEAGUE_NAME", "Premier League")
    season = os.getenv("SEASON", "2025/2026")
    if not base_url or not conn_str:
        raise EnvironmentError(
            "WHOSCORED_BASE_URL and DATABASE_URL environment variables must be set"
        )
    # Scrape referee statistics
    df_stats = scrape_referee_statistics(base_url)
    # Upsert into DB
    upsert_referee_stats(df_stats, conn_str, league, season)


# New main function for multiple leagues

def main():
    import os
    import json
    # Use multi-league environment variables if provided
    conn_str = os.getenv("DATABASE_URL")
    if not conn_str:
        raise EnvironmentError("DATABASE_URL environment variable must be set")

    base_urls_var = os.getenv("WHOSCORED_BASE_URLS")
    league_names_var = os.getenv("LEAGUE_NAMES")
    seasons_var = os.getenv("SEASONS")

    if base_urls_var and league_names_var and seasons_var:
        try:
            base_urls = json.loads(base_urls_var)
            league_names = json.loads(league_names_var)
            seasons = json.loads(seasons_var)
        except json.JSONDecodeError:
            raise ValueError("WHOSCORED_BASE_URLS, LEAGUE_NAMES, and SEASONS must be valid JSON arrays")
        if not (len(base_urls) == len(league_names) == len(seasons)):
            raise ValueError("WHOSCORED_BASE_URLS, LEAGUE_NAMES, and SEASONS arrays must be of equal length")
        for base_url, league, season in zip(base_urls, league_names, seasons):
            df_stats = scrape_referee_statistics(base_url)
            upsert_referee_stats(df_stats, conn_str, league, season)
    else:
        # Fall back to single league variables
        base_url = os.getenv("WHOSCORED_BASE_URL")
        league = os.getenv("LEAGUE_NAME", "Premier League")
        season = os.getenv("SEASON", "2025/2026")
        if not base_url:
            raise EnvironmentError("WHOSCORED_BASE_URL environment variable must be set for single league scrape")
        df_stats = scrape_referee_statistics(base_url)
        upsert_referee_stats(df_stats, conn_str, league, season)



if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
