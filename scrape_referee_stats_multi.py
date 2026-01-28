#!/usr/bin/env python3
"""
Scrape referee statistics from WhoScored for multiple leagues and seasons
defined in the environment variable WHOSCORED_TARGETS_JSON, or fallback to arrays or single variables.

This script uses requests_html to render the JavaScript-driven pages
and uses WhoScored StatisticsFeed endpoints to discover season and stage IDs.
The results are stored in a Postgres-compatible database.

Env vars:
- WHOSCORED_TARGETS_JSON: JSON list of objects:
    [{"region_id": 252, "tournament_id": 2, "competition": "Premier League",
      "country": "England", "seasons": ["2022-2023","2023-2024"]}, ...]
- WHOSCORED_BASE_URLS, LEAGUE_NAMES, SEASONS: fallback arrays for static URLs.
- WHOSCORED_BASE_URL, LEAGUE_NAME, SEASON: fallback single.
- DATABASE_URL: Postgres connection string.
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
from requests_html import HTMLSession
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column mapping from WhoScored to DB-friendly names
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
    """Fetch a referee statistics table from WhoScored and return a DataFrame with DB-friendly columns."""
    response = session.get(url)
    # Wait for table to render; increase timeout if necessary
    response.html.render(timeout=30)
    tables = pd.read_html(response.html.raw_html)
    df = None
    for tbl in tables:
        if "Referee" in tbl.columns:
            df = tbl.copy()
            break
    if df is None:
        raise ValueError(f"No referee table found on {url}")
    # Drop first column if it's rank or unnamed index
    if df.columns[0] == 0 or df.iloc[:, 0].dtype == 'object':
        df = df.drop(columns=df.columns[0])
    df = df.rename(columns=COLUMN_MAP)
    df['source_url'] = url
    return df

def scrape_referee_statistics(base_url: str) -> pd.DataFrame:
    """Scrape referee statistics (overall, home, away) from WhoScored and concatenate into one DataFrame."""
    session = HTMLSession()
    contexts = {"overall": "", "home": "?home=1", "away": "?away=1"}
    data_frames = []
    for ctx_name, suffix in contexts.items():
        url = base_url + suffix
        try:
            df = fetch_referee_table(url, session)
            df['context'] = ctx_name
            data_frames.append(df)
        except Exception as exc:
            logger.warning(f"Failed to scrape {ctx_name} from {url}: {exc}")
    if not data_frames:
        raise ValueError(f"No referee statistics were scraped from {base_url}")
    all_df = pd.concat(data_frames, ignore_index=True)
    # Standardize names: strip
    all_df['referee_name'] = all_df['referee_name'].str.strip()
    return all_df

def upsert_referee_stats(df: pd.DataFrame, conn_str: str, league: str, season: str) -> None:
    """Upsert referee statistics into the referee_stats table."""
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
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
        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (league, season, referee_name, context)
    );
    """
    cur.execute(create_sql)
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
        league, season, referee_name, context, apps, fouls_per_game, fouls_tackles_ratio,
        penalties_per_game, yellow_cards_per_game, yellow_cards_total,
        red_cards_per_game, red_cards_total, source_url
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

def get_season_id(session: HTMLSession, tournament_id: int, season_name: str) -> str:
    """Fetch season ID for a tournament and season name using WhoScored StatisticsFeed."""
    url = f"https://www.whoscored.com/StatisticsFeed/1/GetSeasons?tournamentId={tournament_id}"
    r = session.get(url)
    seasons = r.json()
    for s in seasons:
        if s.get('name') == season_name or s.get('name').replace('-', '/') == season_name:
            return str(s['id'])
    raise ValueError(f"Season '{season_name}' not found for tournament {tournament_id}")

def get_stage_id(session: HTMLSession, tournament_id: int, season_id: str) -> str:
    """Fetch stage ID for a tournament season using WhoScored StatisticsFeed."""
    url = f"https://www.whoscored.com/StatisticsFeed/1/GetStages?tournamentId={tournament_id}&seasonId={season_id}"
    r = session.get(url)
    stages = r.json()
    for st in stages:
        return str(st['id'])
    raise ValueError(f"No stage found for season_id {season_id}")

def build_referee_url(region_id: int, tournament_id: int, season_id: str, stage_id: str, country: str, competition: str, season: str) -> str:
    """Construct the WhoScored referee statistics URL."""
    country_slug = country.lower().replace(' ', '-')
    comp_slug = competition.lower().replace(' ', '-')
    season_slug = season.replace('/', '-')
    return f"https://www.whoscored.com/regions/{region_id}/tournaments/{tournament_id}/seasons/{season_id}/stages/{stage_id}/refereestatistics/{country_slug}-{comp_slug}-{season_slug}"

def process_target(target: dict, conn_str: str) -> None:
    """Process a single target from WHOSCORED_TARGETS_JSON."""
    region_id = int(target['region_id'])
    tournament_id = int(target['tournament_id'])
    competition = target.get('competition') or target.get('league')
    country = target.get('country', '')
    seasons = target.get('seasons', [])
    session = HTMLSession()
    for season in seasons:
        try:
            season_id = get_season_id(session, tournament_id, season)
            stage_id = get_stage_id(session, tournament_id, season_id)
            base_url = build_referee_url(region_id, tournament_id, season_id, stage_id, country, competition, season)
            logger.info(f"Scraping {competition} {season}: {base_url}")
            df_stats = scrape_referee_statistics(base_url)
            upsert_referee_stats(df_stats, conn_str, competition, season)
        except Exception as exc:
            logger.warning(f"Failed to scrape {competition} {season}: {exc}")

def main():
    conn_str = os.getenv("DATABASE_URL")
    if not conn_str:
        raise EnvironmentError("DATABASE_URL must be set")
    targets_json = os.getenv("WHOSCORED_TARGETS_JSON")
    if targets_json:
        try:
            targets = json.loads(targets_json)
        except json.JSONDecodeError:
            raise ValueError("WHOSCORED_TARGETS_JSON must be valid JSON")
        for target in targets:
            process_target(target, conn_str)
    else:
        base_urls_var = os.getenv("WHOSCORED_BASE_URLS")
        league_names_var = os.getenv("LEAGUE_NAMES")
        seasons_var = os.getenv("SEASONS")
        if base_urls_var and league_names_var and seasons_var:
            try:
                base_urls = json.loads(base_urls_var)
                league_names = json.loads(league_names_var)
                seasons = json.loads(seasons_var)
            except json.JSONDecodeError:
                raise ValueError("WHOSCORED_BASE_URLS, LEAGUE_NAMES, SEASONS must be valid JSON arrays")
            if not (len(base_urls) == len(league_names) == len(seasons)):
                raise ValueError("WHOSCORED_BASE_URLS, LEAGUE_NAMES, SEASONS arrays must be of equal length")
            for base_url, league, season in zip(base_urls, league_names, seasons):
                logger.info(f"Scraping {league} {season}")
                df_stats = scrape_referee_statistics(base_url)
                upsert_referee_stats(df_stats, conn_str, league, season)
        else:
            base_url = os.getenv("WHOSCORED_BASE_URL")
            league = os.getenv("LEAGUE_NAME", "Premier League")
            season = os.getenv("SEASON", "2025/2026")
            if not base_url:
                raise EnvironmentError("WHOSCORED_BASE_URL must be set for single league scrape")
            df_stats = scrape_referee_statistics(base_url)
            upsert_referee_stats(df_stats, conn_str, league, season)

if __name__ == "__main__":
    main()
