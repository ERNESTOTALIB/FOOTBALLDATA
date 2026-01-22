"""
Database management utilities for the football scraping project.

This module provides functions to connect to PostgreSQL, create the
required tables and insert match data.  It expects a standard
PostgreSQL connection string in the environment variable
``DATABASE_URL``.  The schema is normalised into three tables:

* ``matches`` – basic information about each match (date, teams and
  result).
* ``match_stats`` – per‑match statistics such as shots, corners and
  cards.
* ``odds_1x2`` – average and maximum closing odds for the home win,
  draw and away win markets.

All three tables are linked by a common ``match_id``.  Matches are
identified uniquely by (league, season, match_date, home_team,
away_team).
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Iterable, List, Tuple

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# SQL definitions for the schema.  Declared as constants for clarity.
CREATE_TABLE_MATCHES = """
CREATE TABLE IF NOT EXISTS matches (
    match_id SERIAL PRIMARY KEY,
    league VARCHAR(10),
    season VARCHAR(10),
    match_date DATE,
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    full_time_home_goals INT,
    full_time_away_goals INT,
    full_time_result CHAR(1),
    half_time_home_goals INT,
    half_time_away_goals INT,
    half_time_result CHAR(1),
    UNIQUE (league, season, match_date, home_team, away_team)
);
"""

CREATE_TABLE_STATS = """
CREATE TABLE IF NOT EXISTS match_stats (
    match_id INT PRIMARY KEY,
    home_shots INT,
    away_shots INT,
    home_shots_on_target INT,
    away_shots_on_target INT,
    home_corners INT,
    away_corners INT,
    home_yellows INT,
    away_yellows INT,
    home_reds INT,
    away_reds INT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);
"""

CREATE_TABLE_ODDS = """
CREATE TABLE IF NOT EXISTS odds_1x2 (
    match_id INT PRIMARY KEY,
    avg_odds_home NUMERIC,
    avg_odds_draw NUMERIC,
    avg_odds_away NUMERIC,
    max_odds_home NUMERIC,
    max_odds_draw NUMERIC,
    max_odds_away NUMERIC,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);
"""


def _get_connection():
    """Return a new psycopg2 connection using DATABASE_URL.

    Raises an EnvironmentError if DATABASE_URL is not set.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "DATABASE_URL environment variable is not set. "
            "Please set it to your PostgreSQL connection string."
        )
    return psycopg2.connect(url)


def init_db() -> None:
    """Create database tables if they do not already exist."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_MATCHES)
            cur.execute(CREATE_TABLE_STATS)
            cur.execute(CREATE_TABLE_ODDS)
        conn.commit()
        logger.info("Database tables created or verified.")
    finally:
        conn.close()


def insert_data(
    matches: List[Dict],
    stats: List[Dict],
    odds: List[Dict],
) -> None:
    """Insert lists of match, stats and odds dictionaries into the DB.

    Each list must be the same length and aligned; that is, ``matches[i]``
    corresponds to ``stats[i]`` and ``odds[i]`` for the same match.  The
    function performs an upsert on the ``matches`` table using the
    unique constraint on (league, season, match_date, home_team,
    away_team).  If a match is inserted for the first time, its new
    ``match_id`` is used to insert the corresponding stats and odds.
    If the match already exists, the stats and odds are skipped to
    avoid duplicates.  (You may choose to update stats/odds instead.)

    Parameters
    ----------
    matches : list of dict
        List of match metadata dictionaries for the ``matches`` table.
    stats : list of dict
        List of statistics dictionaries for the ``match_stats`` table.
    odds : list of dict
        List of odds dictionaries for the ``odds_1x2`` table.
    """
    if not (len(matches) == len(stats) == len(odds)):
        raise ValueError(
            "matches, stats and odds lists must be of equal length"
        )

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            for m, s, o in zip(matches, stats, odds):
                # Insert into matches and get match_id
                cur.execute(
                    """
                    INSERT INTO matches (
                        league, season, match_date,
                        home_team, away_team,
                        full_time_home_goals, full_time_away_goals, full_time_result,
                        half_time_home_goals, half_time_away_goals, half_time_result
                    ) VALUES (
                        %(league)s, %(season)s, %(match_date)s,
                        %(home_team)s, %(away_team)s,
                        %(full_time_home_goals)s, %(full_time_away_goals)s, %(full_time_result)s,
                        %(half_time_home_goals)s, %(half_time_away_goals)s, %(half_time_result)s
                    )
                    ON CONFLICT (league, season, match_date, home_team, away_team)
                    DO NOTHING
                    RETURNING match_id;
                    """,
                    m,
                )
                result = cur.fetchone()
                if result is None:
                    # Match already exists; skip inserting stats/odds
                    continue
                match_id = result[0]
                # Add match_id to stats and odds dicts
                s_with_id = dict(s, match_id=match_id)
                o_with_id = dict(o, match_id=match_id)

                # Insert stats
                cur.execute(
                    """
                    INSERT INTO match_stats (
                        match_id, home_shots, away_shots, home_shots_on_target, away_shots_on_target,
                        home_corners, away_corners, home_yellows, away_yellows, home_reds, away_reds
                    ) VALUES (
                        %(match_id)s, %(home_shots)s, %(away_shots)s, %(home_shots_on_target)s, %(away_shots_on_target)s,
                        %(home_corners)s, %(away_corners)s, %(home_yellows)s, %(away_yellows)s, %(home_reds)s, %(away_reds)s
                    );
                    """,
                    s_with_id,
                )
                # Insert odds
                cur.execute(
                    """
                    INSERT INTO odds_1x2 (
                        match_id, avg_odds_home, avg_odds_draw, avg_odds_away,
                        max_odds_home, max_odds_draw, max_odds_away
                    ) VALUES (
                        %(match_id)s, %(avg_odds_home)s, %(avg_odds_draw)s, %(avg_odds_away)s,
                        %(max_odds_home)s, %(max_odds_draw)s, %(max_odds_away)s
                    );
                    """,
                    o_with_id,
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
