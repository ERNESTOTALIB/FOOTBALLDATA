#!/usr/bin/env python3
"""
Scrape referee statistics from worldfootball.net for multiple leagues and seasons,
as defined in WORLDFOOTBALL_TARGETS_JSON environment variable (JSON list of objects).
Each object must contain:
{
    "league": "Premier League",
    "season": "2022-2023",
    "url": "https://www.worldfootball.net/competition/co91/england-premier-league/se45794/2022-2023/referees/"
}

The script fetches the referee statistics table and upserts into referee_stats table.

Env vars:
- WORLDFOOTBALL_TARGETS_JSON: JSON list of objects; if not provided, fallback to single
  WORLDFOOTBALL_URL, LEAGUE_NAME, SEASON.
- DATABASE_URL: Postgres connection string.

"""

import os
import json
import logging
import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_worldfootball_table(url: str) -> pd.DataFrame:
    """
    Fetch worldfootball referee stats page and return DataFrame with columns:
    referee_name, country, matches, yellow_cards, yellow_red_cards, red_cards, penalties.
    """
  response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
    response.raise_for_status()
    # Parse all tables on the page
    tables = pd.read_html(response.text)
    df = None
    for tbl in tables:
        # Check if columns contain 'Referee' or 'Name'
        cols = [c.lower() for c in tbl.columns]
        if 'referee' in cols or 'name' in cols:
            df = tbl.copy()
            break
    if df is None:
        raise ValueError(f"No referee table found on {url}")
    # Standardize column names
    rename_map = {}
    for col in df.columns:
        lc = str(col).lower()
        if lc in ('name', 'referee'):
            rename_map[col] = 'referee_name'
        elif lc == 'country':
            rename_map[col] = 'country'
        elif lc == 'matches':
            rename_map[col] = 'matches'
        elif lc == 'yellow':
            rename_map[col] = 'yellow_cards'
        elif lc in ('yellow-red', 'yellow-reds', 'yellow_red'):
            rename_map[col] = 'yellow_red_cards'
        elif lc == 'red':
            rename_map[col] = 'red_cards'
        elif lc in ('11m', 'pen', '11 m'):
            rename_map[col] = 'penalties'
        else:
            # keep as is or drop later
            rename_map[col] = col
    df = df.rename(columns=rename_map)
    # Drop unnamed columns (like '#')
    drop_cols = []
    for col in df.columns:
        if isinstance(col, str) and col.startswith('Unnamed') or col == '#':
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # Ensure required columns exist
    req_cols = ['referee_name', 'matches', 'yellow_cards', 'yellow_red_cards', 'red_cards', 'penalties']
    for col in req_cols:
        if col not in df.columns:
            df[col] = 0
    # Filter out rows without referee_name
    df = df[df['referee_name'].notna()]
    # Convert numeric columns
    for col in ['matches', 'yellow_cards', 'yellow_red_cards', 'red_cards', 'penalties']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

def upsert_worldfootball_stats(df: pd.DataFrame, conn_str: str, league: str, season: str, url: str):
    """
    Upsert referee statistics into referee_stats table.
    """
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    create_sql = """
    CREATE TABLE IF NOT EXISTS referee_stats (
        id SERIAL PRIMARY KEY,
        league TEXT NOT NULL,
        season TEXT NOT NULL,
        referee_name TEXT NOT NULL,
        matches INT,
        yellow_cards INT,
        yellow_red_cards INT,
        red_cards INT,
        penalties INT,
        source_url TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (league, season, referee_name)
    );
    """
    cur.execute(create_sql)
    rows = []
    for _, row in df.iterrows():
        rows.append((
            league,
            season,
            row.get('referee_name'),
            int(row.get('matches', 0)),
            int(row.get('yellow_cards', 0)),
            int(row.get('yellow_red_cards', 0)),
            int(row.get('red_cards', 0)),
            int(row.get('penalties', 0)),
            url
        ))
    upsert_sql = """
    INSERT INTO referee_stats (
        league, season, referee_name,
        matches, yellow_cards, yellow_red_cards, red_cards, penalties, source_url
    )
    VALUES %s
    ON CONFLICT (league, season, referee_name)
    DO UPDATE SET
        matches = EXCLUDED.matches,
        yellow_cards = EXCLUDED.yellow_cards,
        yellow_red_cards = EXCLUDED.yellow_red_cards,
        red_cards = EXCLUDED.red_cards,
        penalties = EXCLUDED.penalties,
        source_url = EXCLUDED.source_url,
        updated_at = CURRENT_TIMESTAMP;
    """
    execute_values(cur, upsert_sql, rows)
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Upserted {len(rows)} rows for {league} {season}")

def process_target(target: dict, conn_str: str):
    league = target.get('league')
    season = target.get('season')
    url = target.get('url')
    if not (league and season and url):
        logger.warning(f"Invalid target specification: {target}")
        return
    logger.info(f"Scraping {league} {season} from {url}")
    try:
        df = parse_worldfootball_table(url)
        upsert_worldfootball_stats(df, conn_str, league, season, url)
    except Exception as e:
        logger.error(f"Failed to process {league} {season}: {e}")

def main():
    conn_str = os.getenv('DATABASE_URL')
    if not conn_str:
        raise ValueError("Missing DATABASE_URL environment variable")
    targets_json = os.getenv('WORLDFOOTBALL_TARGETS_JSON')
    targets = []
    if targets_json:
        try:
            targets = json.loads(targets_json)
            if not isinstance(targets, list):
                raise ValueError("WORLDFOOTBALL_TARGETS_JSON must be a JSON list of objects")
        except Exception as e:
            raise ValueError(f"Invalid WORLDFOOTBALL_TARGETS_JSON: {e}")
    if not targets:
        url = os.getenv('WORLDFOOTBALL_URL')
        league = os.getenv('LEAGUE_NAME')
        season = os.getenv('SEASON')
        if url and league and season:
            targets = [{'league': league, 'season': season, 'url': url}]
        else:
            raise ValueError("No targets specified: set WORLDFOOTBALL_TARGETS_JSON or WORLDFOOTBALL_URL, LEAGUE_NAME, SEASON")
    for t in targets:
        process_target(t, conn_str)

if __name__ == '__main__':
    main()
