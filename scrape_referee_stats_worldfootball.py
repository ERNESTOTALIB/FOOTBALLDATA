#!/usr/bin/env python3
"""
Scrape referee statistics from worldfootball.net for multiple leagues and seasons,
as defined in the WORLDFOOTBALL_TARGETS_JSON environment variable (JSON list of objects).
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
import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

def parse_worldfootball_table(url: str) -> pd.DataFrame:
    """
    Fetch worldfootball referee stats page and return DataFrame with columns:
    referee_name, country, matches, yellow_cards, yellow_red_cards, red_cards, penalties.
    """
    response = requests.get(
        url,
        headers={'User-Agent': 'Mozilla/5.0'},
        timeout=30
    )
    response.raise_for_status()
    # Use pandas to read all tables in the page
    tables = pd.read_html(response.text, flavor='bs4')
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
        elif lc in ('yellow-red', 'yellow reds', 'yellow/red'):
            rename_map[col] = 'yellow_red_cards'
        elif lc == 'red':
            rename_map[col] = 'red_cards'
        elif lc == 'penalties':
            rename_map[col] = 'penalties'
    df = df.rename(columns=rename_map)
    # Ensure numeric columns are numeric
    for col in ['matches', 'yellow_cards', 'yellow_red_cards', 'red_cards', 'penalties']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    # Fill missing columns with zeros
    for col in ['yellow_cards', 'yellow_red_cards', 'red_cards', 'penalties']:
        if col not in df.columns:
            df[col] = 0
    return df[['referee_name', 'country', 'matches', 'yellow_cards', 'yellow_red_cards', 'red_cards', 'penalties']]

def upsert_referee_stats(df: pd.DataFrame, league: str, season: str, conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS referee_stats (
            id SERIAL PRIMARY KEY,
            league VARCHAR,
            season VARCHAR,
            referee_name VARCHAR,
            country VARCHAR,
            matches INTEGER,
            yellow_cards INTEGER,
            yellow_red_cards INTEGER,
            red_cards INTEGER,
            penalties INTEGER,
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(league, season, referee_name)
        );
    """)
    rows = []
    for _, row in df.iterrows():
        rows.append((
            league,
            season,
            row['referee_name'],
            row['country'],
            int(row.get('matches', 0)),
            int(row.get('yellow_cards', 0)),
            int(row.get('yellow_red_cards', 0)),
            int(row.get('red_cards', 0)),
            int(row.get('penalties', 0))
        ))
    query = """
        INSERT INTO referee_stats
            (league, season, referee_name, country, matches, yellow_cards, yellow_red_cards, red_cards, penalties)
        VALUES %s
        ON CONFLICT (league, season, referee_name) DO UPDATE
            SET country = EXCLUDED.country,
                matches = EXCLUDED.matches,
                yellow_cards = EXCLUDED.yellow_cards,
                yellow_red_cards = EXCLUDED.yellow_red_cards,
                red_cards = EXCLUDED.red_cards,
                penalties = EXCLUDED.penalties,
                updated_at = NOW();
    """
    execute_values(cur, query, rows)
    conn.commit()
    cur.close()

def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is required")
    conn = psycopg2.connect(db_url)

    targets_json = os.environ.get("WORLDFOOTBALL_TARGETS_JSON")
    if targets_json:
        targets = json.loads(targets_json)
        for target in targets:
            url = target.get("url")
            league = target.get("league") or target.get("competition") or ""
            season = target.get("season") or ""
            if not url:
                continue
            try:
                df = parse_worldfootball_table(url)
                upsert_referee_stats(df, league, season, conn)
                print(f"Upserted {len(df)} rows for {league} {season}")
            except Exception as e:
                print(f"Failed to process {league} {season}: {e}")
    else:
        url = os.environ.get("WORLDFOOTBALL_URL")
        league = os.environ.get("LEAGUE_NAME", "")
        season = os.environ.get("SEASON", "")
        if not url:
            raise ValueError("WORLDFOOTBALL_TARGETS_JSON or WORLDFOOTBALL_URL must be provided")
        df = parse_worldfootball_table(url)
        upsert_referee_stats(df, league, season, conn)
        print(f"Upserted {len(df)} rows for {league} {season}")
    conn.close()

if __name__ == "__main__":
    main()
