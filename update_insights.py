"""
Script to compute and update team statistics and match insights.

This script reads the latest match data from the Supabase database and
computes season‑long and recent‑form statistics for each team.  The
results are written to a `team_stats` table in the same database using
an upsert (insert on conflict do update) strategy.  It is intended to
be run after the match results have been ingested (e.g., in a GitHub
Actions workflow) so that the front‑end always has up‑to‑date
statistics.

Before running this script, ensure that the `DATABASE_URL` environment
variable is set to your Supabase pooler URL.  Also ensure that the
`team_stats` table exists in the database.  A sample schema:

    CREATE TABLE IF NOT EXISTS team_stats (
        league        TEXT,
        season        TEXT,
        team          TEXT,
        avg_goals_for NUMERIC,
        avg_goals_against NUMERIC,
        avg_corners   NUMERIC,
        avg_yellows   NUMERIC,
        avg_reds      NUMERIC,
        recent_goals_for_home NUMERIC,
        recent_goals_against_home NUMERIC,
        recent_goals_for_away NUMERIC,
        recent_goals_against_away NUMERIC,
        recent_corners_home NUMERIC,
        recent_corners_away NUMERIC,
        recent_yellows_home NUMERIC,
        recent_yellows_away NUMERIC,
        recent_reds_home NUMERIC,
        recent_reds_away NUMERIC,
        PRIMARY KEY (league, season, team)
    );

Modify this schema as needed to include additional metrics.
"""

import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


FORM_MATCHES = 10


def weighted_form(matches: pd.DataFrame, goals_for_col: str, goals_against_col: str) -> Dict[str, float]:
    """Compute weighted averages for goals, corners, yellows and reds.

    Parameters
    ----------
    matches : pd.DataFrame
        Subset of matches filtered by team and home/away.
    goals_for_col : str
        Column name for goals scored by the team.
    goals_against_col : str
        Column name for goals conceded by the team.

    Returns
    -------
    dict
        Weighted averages for goals for, goals against, corners, yellows, reds.
    """
    if matches.empty:
        return {
            "goals_for": float("nan"),
            "goals_against": float("nan"),
            "corners": float("nan"),
            "yellows": float("nan"),
            "reds": float("nan"),
        }
    # Sort by date descending and take last FORM_MATCHES
    m = matches.sort_values("match_date", ascending=False).head(FORM_MATCHES)
    weights = np.linspace(FORM_MATCHES, 1, len(m))
    goals_for = (m[goals_for_col] * weights).sum() / weights.sum()
    goals_against = (m[goals_against_col] * weights).sum() / weights.sum()
    corners = ((m["corners"] * weights).sum() / weights.sum())
    yellows = ((m["yellow_cards"] * weights).sum() / weights.sum())
    reds = ((m["red_cards"] * weights).sum() / weights.sum())
    return {
        "goals_for": goals_for,
        "goals_against": goals_against,
        "corners": corners,
        "yellows": yellows,
        "reds": reds,
    }


def compute_team_stats(matches_df: pd.DataFrame, league: str, season: str) -> List[Dict[str, object]]:
    """Compute statistics per team for a given league and season.

    Parameters
    ----------
    matches_df : pd.DataFrame
        DataFrame containing all matches with columns:
        match_date, home_team, away_team, goals_home, goals_away,
        corners_home, corners_away, yellow_home, yellow_away,
        red_home, red_away.
    league : str
        League name.
    season : str
        Season identifier.

    Returns
    -------
    List[dict]
        Each dict contains stats for one team.
    """
    teams = sorted(set(matches_df["home_team"]) | set(matches_df["away_team"]))
    stats = []
    for team in teams:
        home = matches_df[matches_df["home_team"] == team]
        away = matches_df[matches_df["away_team"] == team]
        # Unweighted season averages
        goals_for_season = (
            (home["goals_home"].mean() if not home.empty else 0)
            + (away["goals_away"].mean() if not away.empty else 0)
        ) / 2
        goals_against_season = (
            (home["goals_away"].mean() if not home.empty else 0)
            + (away["goals_home"].mean() if not away.empty else 0)
        ) / 2
        corners_season = (
            (home[["corners_home", "corners_away"]].sum(axis=1).mean() if not home.empty else 0)
            + (away[["corners_home", "corners_away"]].sum(axis=1).mean() if not away.empty else 0)
        ) / 2
        yellows_season = (
            (home[["yellow_home", "yellow_away"]].sum(axis=1).mean() if not home.empty else 0)
            + (away[["yellow_home", "yellow_away"]].sum(axis=1).mean() if not away.empty else 0)
        ) / 2
        reds_season = (
            (home[["red_home", "red_away"]].sum(axis=1).mean() if not home.empty else 0)
            + (away[["red_home", "red_away"]].sum(axis=1).mean() if not away.empty else 0)
        ) / 2
        # Recent form weighted averages
        home_form = weighted_form(
            home.assign(
                corners=home[["corners_home", "corners_away"]].sum(axis=1),
                yellow_cards=home[["yellow_home", "yellow_away"]].sum(axis=1),
                red_cards=home[["red_home", "red_away"]].sum(axis=1),
            ),
            goals_for_col="goals_home",
            goals_against_col="goals_away",
        )
        away_form = weighted_form(
            away.assign(
                corners=away[["corners_home", "corners_away"]].sum(axis=1),
                yellow_cards=away[["yellow_home", "yellow_away"]].sum(axis=1),
                red_cards=away[["red_home", "red_away"]].sum(axis=1),
            ),
            goals_for_col="goals_away",
            goals_against_col="goals_home",
        )
        stats.append({
            "league": league,
            "season": season,
            "team": team,
            "avg_goals_for": goals_for_season,
            "avg_goals_against": goals_against_season,
            "avg_corners": corners_season,
            "avg_yellows": yellows_season,
            "avg_reds": reds_season,
            "recent_goals_for_home": home_form["goals_for"],
            "recent_goals_against_home": home_form["goals_against"],
            "recent_goals_for_away": away_form["goals_for"],
            "recent_goals_against_away": away_form["goals_against"],
            "recent_corners_home": home_form["corners"],
            "recent_corners_away": away_form["corners"],
            "recent_yellows_home": home_form["yellows"],
            "recent_yellows_away": away_form["yellows"],
            "recent_reds_home": home_form["reds"],
            "recent_reds_away": away_form["reds"],
        })
    return stats


def fetch_match_data(conn) -> List[Dict[str, object]]:
    """Fetch match data from the database and return as a DataFrame.

    The function expects that matches are stored in tables `matches` and
    `match_stats` (as created by the scraper).  It joins the tables and
    returns a DataFrame with one row per match containing aggregated
    stats: goals, corners, cards.
    """
    query = """
        SELECT
            m.match_id,
            m.league,
            m.season,
            m.match_date,
            m.home_team,
            m.away_team,
            m.full_time_home_goals AS goals_home,
            m.full_time_away_goals AS goals_away,
            s.home_corners AS corners_home,
            s.away_corners AS corners_away,
            s.home_yellows AS yellow_home,
            s.away_yellows AS yellow_away,
            s.home_reds AS red_home,
            s.away_reds AS red_away
        FROM matches m
        JOIN match_stats s ON m.match_id = s.match_id
    """
    df = pd.read_sql(query, conn)
    return df


def upsert_team_stats(conn, stats: List[Dict[str, object]]) -> None:
    """Insert or update team statistics in the team_stats table."""
    if not stats:
        return
    cols = [
        "league",
        "season",
        "team",
        "avg_goals_for",
        "avg_goals_against",
        "avg_corners",
        "avg_yellows",
        "avg_reds",
        "recent_goals_for_home",
        "recent_goals_against_home",
        "recent_goals_for_away",
        "recent_goals_against_away",
        "recent_corners_home",
        "recent_corners_away",
        "recent_yellows_home",
        "recent_yellows_away",
        "recent_reds_home",
        "recent_reds_away",
    ]
    values = [[stat[col] for col in cols] for stat in stats]
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join([f"{col} = EXCLUDED.{col}" for col in cols[3:]])
    sql = f"""
        INSERT INTO team_stats ({', '.join(cols)})
        VALUES {placeholders}
        ON CONFLICT (league, season, team)
        DO UPDATE SET
            {updates};
    """
    execute_values(conn.cursor(), sql, values)
    conn.commit()


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL environment variable is not set", file=sys.stderr)
        sys.exit(1)
    conn = psycopg2.connect(url)
    matches_df = fetch_match_data(conn)
    if matches_df.empty:
        print("No match data found.")
        conn.close()
        return
    stats_records = []
    # Compute per league and season
    for (league, season), group in matches_df.groupby(["league", "season"]):
        stats = compute_team_stats(group, league, season)
        stats_records.extend(stats)
    upsert_team_stats(conn, stats_records)
    conn.close()
    print(f"Updated stats for {len(stats_records)} team-season entries.")


if __name__ == "__main__":
    main()
