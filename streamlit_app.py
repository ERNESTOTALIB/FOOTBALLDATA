"""
Streamlit dashboard for exploring probabilistic football models.

This app allows you to select a league, season and specific match to
inspect historical statistics such as average goals, corners and
disciplinary records for each team. It also displays head-to-head
results over the past five years and applies a simple weighting to
recent form (last 10 matches) when calculating averages.
"""

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration

DATA_PATHS: Dict[str, str] = {
    "Premier League": "data/premier_league_clean.csv",
    "La Liga": "data/la_liga_clean.csv",
    "Serie A": "data/serie_a_clean.csv",
    "Bundesliga": "data/bundesliga_clean.csv",
    "Ligue 1": "data/ligue_1_clean.csv",
}

# Map human-friendly league names to the codes stored in the database.
# The ingestion pipeline writes the `league` column using codes like E0, SP1, etc.
LEAGUE_NAME_TO_CODE: Dict[str, str] = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
}

# Number of years to look back for head-to-head statistics
H2H_YEARS = 5
# Number of matches to use for form weighting
FORM_MATCHES = 10

# Additional helper functions for per-team metrics and probability calculations
def value_metric_for_team(row: pd.Series, team: str, metric: str):
    """
    Returns the value of a metric (cards or corners) for the specified team in a given match row.

    Parameters
    ----------
    row : pd.Series
        A row from the matches DataFrame.
    team : str
        Team name to extract the metric for.
    metric : str
        One of {"cards", "corners"} indicating which metric to extract.

    Returns
    -------
    float or None
        The value of the metric for the team in that match, or None if the team is not part of the match.
    """
    if metric == "cards":
        if row.get("HomeTeam") == team:
            return row.get("cards_home", None)
        elif row.get("AwayTeam") == team:
            return row.get("cards_away", None)
    elif metric == "corners":
        if row.get("HomeTeam") == team:
            return row.get("corners_home", None)
        elif row.get("AwayTeam") == team:
            return row.get("corners_away", None)
    return None

def calc_over_under_for_subset(
    df_subset: pd.DataFrame,
    category: str,
    scope: str,
    line: float,
    side: str,
    home_team: str,
    away_team: str,
    time_frame: str = "Full match",
) -> Dict[str, object]:
    """
    Calculate over/under probability and list of failures for a given subset.

    Parameters
    ----------
    df_subset : pd.DataFrame
        Filtered DataFrame for the subset.
    category : str
        One of "Goles", "Tarjetas", "Córners".
    scope : str
        Scope selected by user ("Total match", "Home", "Away").
    line : float
        Threshold line (e.g., 2.5).
    side : str
        "over" or "under" indicating the direction of the comparison.
    home_team : str
        Name of the home team in the selected matchup.
    away_team : str
        Name of the away team in the selected matchup.

    Returns
    -------
    dict
        Dictionary with keys: "p" (probability float or None), "hits" (int),
        "n" (int), and "fails" (List[str]) listing opponents and values for failing matches.
    """
    # Determine which time frame column to use for goals
    if category == "Goles":
        if time_frame == "First Half":
            col_name = "goals_first_half"
        elif time_frame == "Second Half":
            col_name = "goals_second_half"
        else:
            col_name = "goals_total"
        series = df_subset[col_name].dropna()
        n = len(series)
        if n == 0:
            return {"p": None, "hits": 0, "n": 0, "fails": []}
        # Define hit condition
        hits_mask = (series > line) if side == "over" else (series < line)
        hits = int(hits_mask.sum())
        fails_mask = ~hits_mask
        fails_list = []
        for idx in series[fails_mask].index:
            row = df_subset.loc[idx]
            # Determine opponent relative to home_team; fallback to away_team
            if row.get("HomeTeam") == home_team:
                opp = row.get("AwayTeam")
            elif row.get("AwayTeam") == home_team:
                opp = row.get("HomeTeam")
            else:
                if row.get("HomeTeam") == away_team:
                    opp = row.get("AwayTeam")
                elif row.get("AwayTeam") == away_team:
                    opp = row.get("HomeTeam")
                else:
                    opp = row.get("AwayTeam")
            val = row.get(col_name)
            fails_list.append(f"{opp} ({val:.0f})")
        return {"p": hits / n if n > 0 else None, "hits": hits, "n": n, "fails": fails_list}

    # Determine metric for cards or corners
    metric = None
    if category == "Tarjetas":
        metric = "cards"
    elif category == "Córners":
        metric = "corners"

    if metric is not None:
        # Choose appropriate column based on scope and time frame
        if metric == "cards":
            if scope == "Total match":
                if time_frame == "First Half":
                    col = "cards_first_half"
                elif time_frame == "Second Half":
                    col = "cards_second_half"
                else:
                    col = "cards_total"
            elif scope == "Home":
                if time_frame == "First Half":
                    col = "cards_home_first_half"
                elif time_frame == "Second Half":
                    col = "cards_home_second_half"
                else:
                    col = "cards_home"
            else:  # Away scope
                if time_frame == "First Half":
                    col = "cards_away_first_half"
                elif time_frame == "Second Half":
                    col = "cards_away_second_half"
                else:
                    col = "cards_away"
        else:  # corners
            if scope == "Total match":
                if time_frame == "First Half":
                    col = "corners_first_half"
                elif time_frame == "Second Half":
                    col = "corners_second_half"
                else:
                    col = "corners_total"
            elif scope == "Home":
                if time_frame == "First Half":
                    col = "corners_home_first_half"
                elif time_frame == "Second Half":
                    col = "corners_home_second_half"
                else:
                    col = "corners_home"
            else:  # Away
                if time_frame == "First Half":
                    col = "corners_away_first_half"
                elif time_frame == "Second Half":
                    col = "corners_away_second_half"
                else:
                    col = "corners_away"

        # Total match scope: simple series of col
        if scope == "Total match":
            series = df_subset[col].dropna()
            n = len(series)
            if n == 0:
                return {"p": None, "hits": 0, "n": 0, "fails": []}
            hits_mask = (series > line) if side == "over" else (series < line)
            hits = int(hits_mask.sum())
            fails_mask = ~hits_mask
            fails_list = []
            for idx in series[fails_mask].index:
                row = df_subset.loc[idx]
                # Determine opponent relative to home_team and away_team
                if row.get("HomeTeam") == home_team:
                    opp = row.get("AwayTeam")
                elif row.get("AwayTeam") == home_team:
                    opp = row.get("HomeTeam")
                else:
                    if row.get("HomeTeam") == away_team:
                        opp = row.get("AwayTeam")
                    elif row.get("AwayTeam") == away_team:
                        opp = row.get("HomeTeam")
                    else:
                        opp = row.get("AwayTeam")
                val = row.get(col)
                fails_list.append(f"{opp} ({val:.0f})")
            return {"p": hits / n if n > 0 else None, "hits": hits, "n": n, "fails": fails_list}

        # Team-specific scopes (Home or Away)
        team_assigned = home_team if scope == "Home" else away_team
        entries = []
        for idx, row in df_subset.iterrows():
            # Determine value based on metric and time frame for the assigned team
            if metric == "cards":
                if row.get("HomeTeam") == team_assigned:
                    v = row.get("cards_home" if time_frame == "Full match" else ("cards_home_first_half" if time_frame == "First Half" else "cards_home_second_half"), None)
                elif row.get("AwayTeam") == team_assigned:
                    v = row.get("cards_away" if time_frame == "Full match" else ("cards_away_first_half" if time_frame == "First Half" else "cards_away_second_half"), None)
                else:
                    v = None
            else:  # corners
                if row.get("HomeTeam") == team_assigned:
                    v = row.get("corners_home" if time_frame == "Full match" else ("corners_home_first_half" if time_frame == "First Half" else "corners_home_second_half"), None)
                elif row.get("AwayTeam") == team_assigned:
                    v = row.get("corners_away" if time_frame == "Full match" else ("corners_away_first_half" if time_frame == "First Half" else "corners_away_second_half"), None)
                else:
                    v = None
            if v is None:
                continue
            entries.append((idx, v, row))
        n = len(entries)
        if n == 0:
            return {"p": None, "hits": 0, "n": 0, "fails": []}
        hits = 0
        fails_list = []
        for idx, v, row in entries:
            ok = (v > line) if side == "over" else (v < line)
            if ok:
                hits += 1
            else:
                # Determine opponent relative to assigned team
                if row.get("HomeTeam") == team_assigned:
                    opp = row.get("AwayTeam")
                else:
                    opp = row.get("HomeTeam")
                fails_list.append(f"{opp} ({v:.0f})")
        return {"p": hits / n if n > 0 else None, "hits": hits, "n": n, "fails": fails_list}

    # Default case: no valid category
    return {"p": None, "hits": 0, "n": 0, "fails": []}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize/standardize column names to the canonical names used by the app.

    Handles:
      - different casing (HomeTeam vs hometeam)
      - snake_case variants (home_team)
      - common alternative names

    Canonical columns used by the app:
      Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HC, AC, HY, AY, HR, AR, League, Season
    """
    if df is None or df.empty:
        return df

    # 1) Strip whitespace and build a lowercase lookup
    cleaned_cols = [str(c).strip() for c in df.columns]
    df.columns = cleaned_cols

    lower_map = {c.lower(): c for c in df.columns}

    # 2) Define canonical mapping by lowercase keys
    canonical_by_lower = {
        "date": "Date",
        "match_date": "Date",
        "utc_date": "Date",
        "fixture_date": "Date",

        "hometeam": "HomeTeam",
        "home_team": "HomeTeam",
        "home": "HomeTeam",
        "team_home": "HomeTeam",

        "awayteam": "AwayTeam",
        "away_team": "AwayTeam",
        "away": "AwayTeam",
        "team_away": "AwayTeam",

        "fthg": "FTHG",
        "full_time_home_goals": "FTHG",
        "home_goals": "FTHG",
        "hg": "FTHG",
        "goals_home": "FTHG",

        "ftag": "FTAG",
        "full_time_away_goals": "FTAG",
        "away_goals": "FTAG",
        "ag": "FTAG",
        "goals_away": "FTAG",

        "ftr": "FTR",
        "full_time_result": "FTR",
        "result": "FTR",

        "hc": "HC",
        "home_corners": "HC",
        "corners_home": "HC",

        "ac": "AC",
        "away_corners": "AC",
        "corners_away": "AC",

        "hy": "HY",
        "home_yellows": "HY",
        "home_yellow_cards": "HY",
        "yellow_home": "HY",

        "ay": "AY",
        "away_yellows": "AY",
        "away_yellow_cards": "AY",
        "yellow_away": "AY",

        "hr": "HR",
        "home_reds": "HR",
        "home_red_cards": "HR",
        "red_home": "HR",

        "ar": "AR",
        "away_reds": "AR",
        "away_red_cards": "AR",
        "red_away": "AR",

        "league": "League",
        "division": "League",

        "season": "Season",
        # Referee mapping
        "referee": "Referee",
        "ref": "Referee",
    }

    rename_dict = {}
    for lower_name, canonical in canonical_by_lower.items():
        if lower_name in lower_map:
            rename_dict[lower_map[lower_name]] = canonical

    if rename_dict:
        df = df.rename(columns=rename_dict)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "FTHG" in df.columns and "FTAG" in df.columns:
        df["goals_total"] = df["FTHG"].fillna(0) + df["FTAG"].fillna(0)
        df["goal_diff"] = df["FTHG"].fillna(0) - df["FTAG"].fillna(0)

    for c in ["HY", "AY", "HR", "AR"]:
        if c not in df.columns:
            df[c] = 0
    df["cards_total"] = df["HY"].fillna(0) + df["AY"].fillna(0) + df["HR"].fillna(0) + df["AR"].fillna(0)
    df["cards_home"] = df["HY"].fillna(0) + df["HR"].fillna(0)
    df["cards_away"] = df["AY"].fillna(0) + df["AR"].fillna(0)

    for c in ["HC", "AC"]:
        if c not in df.columns:
            df[c] = 0
    df["corners_total"] = df["HC"].fillna(0) + df["AC"].fillna(0)
    df["corners_home"] = df["HC"].fillna(0)
    df["corners_away"] = df["AC"].fillna(0)

    # Add first-half and second-half metrics if available, else fallback to equal split
    # Goals: use half-time goals if present
    if "HTHG" in df.columns and "HTAG" in df.columns:
        df["goals_first_half"] = df["HTHG"].fillna(0) + df["HTAG"].fillna(0)
        df["goals_second_half"] = df["goals_total"] - df["goals_first_half"]
    else:
        df["goals_first_half"] = df["goals_total"] / 2.0
        df["goals_second_half"] = df["goals_total"] / 2.0
    # Cards: assume equal split between halves if specific data not available
    df["cards_first_half"] = df["cards_total"] / 2.0
    df["cards_second_half"] = df["cards_total"] / 2.0
    df["cards_home_first_half"] = df["cards_home"] / 2.0
    df["cards_home_second_half"] = df["cards_home"] / 2.0
    df["cards_away_first_half"] = df["cards_away"] / 2.0
    df["cards_away_second_half"] = df["cards_away"] / 2.0
    # Corners: assume equal split between halves if specific data not available
    df["corners_first_half"] = df["corners_total"] / 2.0
    df["corners_second_half"] = df["corners_total"] / 2.0
    df["corners_home_first_half"] = df["corners_home"] / 2.0
    df["corners_home_second_half"] = df["corners_home"] / 2.0
    df["corners_away_first_half"] = df["corners_away"] / 2.0
    df["corners_away_second_half"] = df["corners_away"] / 2.0

    return df


def over_under_prob(df: pd.DataFrame, col: str, line: float, side: str) -> dict:
    s = df[col].dropna()
    n = int(len(s))
    if n == 0:
        return {"p": None, "hits": 0, "n": 0}
    if side == "over":
        hits = int((s > line).sum())
    else:
        hits = int((s < line).sum())
    return {"p": hits / n, "hits": hits, "n": n}


def handicap_prob(df: pd.DataFrame, team_role: str, line: float) -> dict:
    gd = df["goal_diff"].dropna()
    n = int(len(gd))
    if n == 0:
        return {"p": None, "hits": 0, "n": 0}
    if team_role == "home":
        hits = int(((gd + line) > 0).sum())
    else:
        hits = int(((-gd + line) > 0).sum())
    return {"p": hits / n, "hits": hits, "n": n}


def compute_referee_card_stats(
    df: pd.DataFrame,
    referee_name: str,
    home_team: str,
    away_team: str,
    last_n: int = 30,
) -> Dict[str, object]:
    """
    Compute card-related statistics for a given referee.

    This function calculates several metrics based on the last ``last_n``
    matches officiated by ``referee_name`` in the provided DataFrame ``df``.
    It returns the overall average total cards shown, the average cards
    awarded to home and away teams, as well as the average cards given to
    the specific ``home_team`` and ``away_team`` if they have been
    officiated by the referee. It also computes probabilities of the total
    cards exceeding each half-point line from 0.5 to 7.5.

    Parameters
    ----------
    df : pd.DataFrame
        The complete matches DataFrame with normalized column names and
        derived card columns (cards_total, cards_home, cards_away).
    referee_name : str
        Name of the referee to compute statistics for.
    home_team : str
        The home team selected in the matchup (to compute team-specific stats).
    away_team : str
        The away team selected in the matchup (to compute team-specific stats).
    last_n : int, optional
        The number of most recent matches to consider for the referee. Defaults to 30.

    Returns
    -------
    dict
        A dictionary containing the number of matches considered (``n``),
        average total cards (``avg_total_cards``), average cards to home
        teams (``avg_home_cards``), average cards to away teams
        (``avg_away_cards``), average cards awarded to the selected
        ``home_team`` (``avg_cards_to_home_team``) and ``away_team``
        (``avg_cards_to_away_team``) if applicable, as well as a
        dictionary ``over_probs`` mapping each half-point line to the
        probability that total cards exceed that line.
    """
    # Filter matches officiated by the referee
    ref_matches = df[df.get("Referee") == referee_name]
    ref_matches = ref_matches.sort_values("Date")
    # Use the last ``last_n`` matches
    if last_n > 0:
        ref_matches = ref_matches.tail(last_n)
    n_matches = len(ref_matches)
    if n_matches == 0:
        return {
            "n": 0,
            "avg_total_cards": None,
            "avg_home_cards": None,
            "avg_away_cards": None,
            "avg_cards_to_home_team": None,
            "avg_cards_to_away_team": None,
            "matches_home_team": 0,
            "matches_away_team": 0,
            "over_probs": {},
        }
    # Calculate average totals
    avg_total_cards = ref_matches.get("cards_total", pd.Series(dtype=float)).dropna().mean()
    avg_home_cards = ref_matches.get("cards_home", pd.Series(dtype=float)).dropna().mean()
    avg_away_cards = ref_matches.get("cards_away", pd.Series(dtype=float)).dropna().mean()
    # Compute probabilities for each line 0.5 to 7.5
    lines = [x + 0.5 for x in range(0, 8)]
    over_probs = {}
    for line in lines:
        series = ref_matches.get("cards_total", pd.Series(dtype=float)).dropna()
        if len(series) > 0:
            hits = int((series > line).sum())
            over_probs[line] = hits / len(series)
        else:
            over_probs[line] = None
    # Average cards given specifically to the selected teams
    cards_to_home_team = []
    matches_home_team = 0
    if home_team:
        for _, row in ref_matches.iterrows():
            if row.get("HomeTeam") == home_team:
                val = row.get("cards_home")
                if pd.notna(val):
                    cards_to_home_team.append(val)
                    matches_home_team += 1
            elif row.get("AwayTeam") == home_team:
                val = row.get("cards_away")
                if pd.notna(val):
                    cards_to_home_team.append(val)
                    matches_home_team += 1
    avg_cards_to_home_team = (
        np.mean(cards_to_home_team) if cards_to_home_team else None
    )
    # For away_team
    cards_to_away_team = []
    matches_away_team = 0
    if away_team:
        for _, row in ref_matches.iterrows():
            if row.get("HomeTeam") == away_team:
                val = row.get("cards_home")
                if pd.notna(val):
                    cards_to_away_team.append(val)
                    matches_away_team += 1
            elif row.get("AwayTeam") == away_team:
                val = row.get("cards_away")
                if pd.notna(val):
                    cards_to_away_team.append(val)
                    matches_away_team += 1
    avg_cards_to_away_team = (
        np.mean(cards_to_away_team) if cards_to_away_team else None
    )
    return {
        "n": n_matches,
        "avg_total_cards": avg_total_cards,
        "avg_home_cards": avg_home_cards,
        "avg_away_cards": avg_away_cards,
        "avg_cards_to_home_team": avg_cards_to_home_team,
        "avg_cards_to_away_team": avg_cards_to_away_team,
        "matches_home_team": matches_home_team,
        "matches_away_team": matches_away_team,
        "over_probs": over_probs,
    }


def subset_lastN_any(df: pd.DataFrame, team: str, N: int) -> pd.DataFrame:
    t = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date")
    return t.tail(N)


def subset_lastN_home(df: pd.DataFrame, team: str, N: int) -> pd.DataFrame:
    t = df[df["HomeTeam"] == team].sort_values("Date")
    return t.tail(N)


def subset_lastN_away(df: pd.DataFrame, team: str, N: int) -> pd.DataFrame:
    t = df[df["AwayTeam"] == team].sort_values("Date")
    return t.tail(N)


def subset_h2h(df: pd.DataFrame, home: str, away: str, years: int) -> pd.DataFrame:
    if "Date" not in df.columns or df["Date"].isna().all():
        return pd.DataFrame()
    cutoff = df["Date"].max() - pd.DateOffset(years=int(years))
    m = (
        ((df["HomeTeam"] == home) & (df["AwayTeam"] == away))
        | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
    )
    return df[m & (df["Date"] >= cutoff)].sort_values("Date")


def load_data(league: str) -> pd.DataFrame:
    use_db = os.environ.get("USE_DB", "false").lower() == "true"
    db_url = os.environ.get("DATABASE_URL")

    if use_db and db_url:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError:
            st.error(
                "psycopg2 is required to load data from the database. "
                "Add 'psycopg2-binary' to requirements.txt"
            )
            return pd.DataFrame()
        try:
            conn = psycopg2.connect(db_url)
        except Exception as e:
            st.error(f"No se pudo conectar a la base de datos: {e}")
            return pd.DataFrame()

        query = """
            SELECT
              m.match_date AS date,
              m.home_team AS hometeam,
              m.away_team AS awayteam,
              m.full_time_home_goals AS fthg,
              m.full_time_away_goals AS ftag,
              m.full_time_result AS ftr,
              s.home_corners AS hc,
              s.away_corners AS ac,
              s.home_yellows AS hy,
              s.away_yellows AS ay,
              s.home_reds AS hr,
              s.away_reds AS ar,
              m.league AS league,
              m.season AS season
            FROM matches m
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE m.league = %s;
        """

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                league_code = LEAGUE_NAME_TO_CODE.get(league, league)
                cur.execute(query, (league_code,))
                records = cur.fetchall()
            df = pd.DataFrame(records)
        except Exception as e:
            st.error(f"Error al consultar los datos: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

        df = normalize_columns(df)
        required = ["Date", "HomeTeam", "AwayTeam"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(
                f"Faltan columnas requeridas: {missing}. "
                f"Columnas disponibles: {list(df.columns)}"
            )
            return pd.DataFrame()
        return df

    # Fallback to local CSV
    file_path = DATA_PATHS.get(league)
    if not file_path:
        st.error(f"No data path configured for {league}")
        return pd.DataFrame()

    path = Path(file_path)
    if not path.exists():
        st.error(
            f"Data file for {league} not found at {path}. "
            "Place the cleaned CSV in the specified location."
        )
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = normalize_columns(df)

    required = ["Date", "HomeTeam", "AwayTeam"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            f"Faltan columnas requeridas: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )
        return pd.DataFrame()

    return df


def compute_outcome(row: pd.Series) -> str:
    if row["FTHG"] > row["FTAG"]:
        return "H"
    if row["FTHG"] < row["FTAG"]:
        return "A"
    return "D"


def team_form(df: pd.DataFrame, team: str, n: int = FORM_MATCHES) -> pd.DataFrame:
    team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    team_matches.sort_values("Date", inplace=True)
    team_matches = team_matches.tail(n)
    team_matches["Outcome"] = team_matches.apply(compute_outcome, axis=1)
    return team_matches


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    w_sum = weights.sum()
    if w_sum == 0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / w_sum)


def compute_team_stats(df: pd.DataFrame, team: str) -> Dict[str, float]:
    form = team_form(df, team, FORM_MATCHES)
    if form.empty:
        return {}

    weights = np.linspace(1, 2, len(form))
    weights = weights / weights.sum()

    home_matches = form[form["HomeTeam"] == team]
    away_matches = form[form["AwayTeam"] == team]

    def safe_weighted(series: pd.Series, w: np.ndarray) -> float:
        if series is None or series.empty:
            return float("nan")
        w_local = w[-len(series):]
        return weighted_average(series.values.astype(float), w_local.astype(float))

    stats = {
        "weighted_goals_for_home": safe_weighted(home_matches.get("FTHG"), weights),
        "weighted_goals_against_home": safe_weighted(home_matches.get("FTAG"), weights),
        "weighted_goals_for_away": safe_weighted(away_matches.get("FTAG"), weights),
        "weighted_goals_against_away": safe_weighted(away_matches.get("FTHG"), weights),

        "weighted_corners_home": safe_weighted(home_matches.get("HC"), weights),
        "weighted_corners_away": safe_weighted(away_matches.get("AC"), weights),

        "weighted_yellows_home": safe_weighted(home_matches.get("HY"), weights),
        "weighted_yellows_away": safe_weighted(away_matches.get("AY"), weights),

        "weighted_reds_home": safe_weighted(home_matches.get("HR"), weights),
        "weighted_reds_away": safe_weighted(away_matches.get("AR"), weights),
    }
    return stats


def get_h2h(df: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    if "Date" not in df.columns or df["Date"].isna().all():
        return pd.DataFrame()

    cutoff = df["Date"].max() - pd.DateOffset(years=H2H_YEARS)
    mask = (
        (((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
         ((df["HomeTeam"] == away) & (df["AwayTeam"] == home)))
        & (df["Date"] >= cutoff)
    )
    h2h = df.loc[mask].copy()
    h2h.sort_values("Date", inplace=True)
    return h2h


def main() -> None:
    st.title("Football Data Dashboard")
    st.write(
        "Select a league and match to explore probabilities and statistics. "
        "This dashboard uses the last 10 matches for each team to compute weighted averages "
        "and displays head-to-head results over the past five years."
    )

    league = st.sidebar.selectbox("League", list(DATA_PATHS.keys()))
    df = load_data(league)

    if df.empty:
        st.stop()

    df = add_derived_cols(df)

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Season selection
    seasons = sorted(df["Season"].dropna().unique()) if "Season" in df.columns else []
    if not seasons:
        st.error("No hay temporadas disponibles en los datos cargados.")
        st.stop()

    season = st.sidebar.selectbox("Season", seasons)
    season_df = df[df["Season"] == season].copy()
    season_df.sort_values("Date", inplace=True)

    # Match or custom matchup selection
    st.sidebar.markdown("---")
    mode = st.sidebar.radio(
        "Mode",
        ["Select match from dataset", "Custom matchup (Home/Away)"],
        index=1
    )
    calc_scope = st.sidebar.selectbox(
        "Data scope for calculations",
        ["Selected season", "All seasons"],
        index=0
    )
    N_any = st.sidebar.selectbox("N last matches (general)", [10, 15, 20, 30], index=0)
    N_ha = st.sidebar.selectbox("N last matches (home/away)", [5, 10, 15, 20], index=1)
    H2H_years = st.sidebar.selectbox("H2H lookback (years)", [3, 5, 10], index=1)

    base_df = season_df if calc_scope == "Selected season" else df
    base_df = base_df.dropna(subset=["Date"]).sort_values("Date")

    if mode == "Select match from dataset":
        match_options = season_df.apply(
            lambda r: f"{r['Date'].date()} — {r['HomeTeam']} vs {r['AwayTeam']}", axis=1
        ).tolist()
        match_idx = st.sidebar.selectbox(
            "Match", range(len(match_options)), format_func=lambda i: match_options[i]
        )
        match_row = season_df.iloc[match_idx]
        home_team = match_row["HomeTeam"]
        away_team = match_row["AwayTeam"]
    else:
        teams = sorted(list(set(list(base_df["HomeTeam"].dropna()) + list(base_df["AwayTeam"].dropna()))))
        home_team = st.sidebar.selectbox("Home Team", teams, index=0)
        away_candidates = [t for t in teams if t != home_team]
        away_team = st.sidebar.selectbox("Away Team", away_candidates, index=0)
        match_row = None

    st.header(f"{home_team} vs {away_team}")
    st.write(f"League: {league} | Season: {season} | Data scope: {calc_scope}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Match Info", "Team Stats", "Head-to-Head", "Probabilities"
    ])

    with tab1:
        st.subheader("Match details")
        if match_row is not None:
            st.table(pd.DataFrame([match_row]))
        else:
            st.write("Custom matchup: no specific match selected from dataset.")

    with tab2:
        st.subheader("Weighted team statistics (last 10 matches)")
        # Compute weighted averages using matches from the selected season
        season_mask = base_df["Season"] == season
        home_stats = compute_team_stats(base_df[season_mask], home_team)
        away_stats = compute_team_stats(base_df[season_mask], away_team)
        if not home_stats or not away_stats:
            st.warning("No sufficient match history to compute stats for one or both teams.")
        else:
            form_table = pd.DataFrame([
                {
                    "Equipo": home_team,
                    "Goles a favor Casa": f"{home_stats['weighted_goals_for_home']:.2f}",
                    "Goles en contra Casa": f"{home_stats['weighted_goals_against_home']:.2f}",
                    "Goles a favor Fuera": f"{home_stats['weighted_goals_for_away']:.2f}",
                    "Goles en contra Fuera": f"{home_stats['weighted_goals_against_away']:.2f}",
                    "Córners Casa": f"{home_stats['weighted_corners_home']:.2f}",
                    "Córners Fuera": f"{home_stats['weighted_corners_away']:.2f}",
                    "Amarillas Casa": f"{home_stats['weighted_yellows_home']:.2f}",
                    "Amarillas Fuera": f"{home_stats['weighted_yellows_away']:.2f}",
                    "Rojas Casa": f"{home_stats['weighted_reds_home']:.2f}",
                    "Rojas Fuera": f"{home_stats['weighted_reds_away']:.2f}",
                },
                {
                    "Equipo": away_team,
                    "Goles a favor Casa": f"{away_stats['weighted_goals_for_home']:.2f}",
                    "Goles en contra Casa": f"{away_stats['weighted_goals_against_home']:.2f}",
                    "Goles a favor Fuera": f"{away_stats['weighted_goals_for_away']:.2f}",
                    "Goles en contra Fuera": f"{away_stats['weighted_goals_against_away']:.2f}",
                    "Córners Casa": f"{away_stats['weighted_corners_home']:.2f}",
                    "Córners Fuera": f"{away_stats['weighted_corners_away']:.2f}",
                    "Amarillas Casa": f"{away_stats['weighted_yellows_home']:.2f}",
                    "Amarillas Fuera": f"{away_stats['weighted_yellows_away']:.2f}",
                    "Rojas Casa": f"{away_stats['weighted_reds_home']:.2f}",
                    "Rojas Fuera": f"{away_stats['weighted_reds_away']:.2f}",
                },
            ])
            st.table(form_table.set_index("Equipo"))

            # Compute and display average total goals, corners and cards for the teams in their roles
            home_matches_as_home = base_df[(base_df["Season"] == season) & (base_df["HomeTeam"] == home_team)]
            away_matches_as_away = base_df[(base_df["Season"] == season) & (base_df["AwayTeam"] == away_team)]
            def mean_or_nan(series):
                return series.mean() if len(series) > 0 else float("nan")
            avg_home_goals = mean_or_nan(home_matches_as_home.get("goals_total", pd.Series(dtype=float)))
            avg_home_corners = mean_or_nan(home_matches_as_home.get("corners_total", pd.Series(dtype=float)))
            avg_home_cards = mean_or_nan(home_matches_as_home.get("cards_total", pd.Series(dtype=float)))
            avg_away_goals = mean_or_nan(away_matches_as_away.get("goals_total", pd.Series(dtype=float)))
            avg_away_corners = mean_or_nan(away_matches_as_away.get("corners_total", pd.Series(dtype=float)))
            avg_away_cards = mean_or_nan(away_matches_as_away.get("cards_total", pd.Series(dtype=float)))
            avg_totals = pd.DataFrame([
                {
                    "Equipo": home_team,
                    "Media Goles Totales (como local)": f"{avg_home_goals:.2f}",
                    "Media Córners Totales (como local)": f"{avg_home_corners:.2f}",
                    "Media Tarjetas Totales (como local)": f"{avg_home_cards:.2f}",
                },
                {
                    "Equipo": away_team,
                    "Media Goles Totales (como visitante)": f"{avg_away_goals:.2f}",
                    "Media Córners Totales (como visitante)": f"{avg_away_corners:.2f}",
                    "Media Tarjetas Totales (como visitante)": f"{avg_away_cards:.2f}",
                },
            ])
            st.write("Average totals by position (historical):")
            st.table(avg_totals.set_index("Equipo"))

    with tab3:
        # Show head-to-head matches across the selected data scope using the H2H_years parameter
        st.subheader(f"Head-to-head results (last {H2H_years} years)")
        h2h = subset_h2h(base_df, home_team, away_team, H2H_years)
        if h2h.empty:
            st.write("No head-to-head matches found in the selected period.")
        else:
            display_cols = [
                "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                "HC", "AC", "HY", "AY", "HR", "AR"
            ]
            existing_cols = [c for c in display_cols if c in h2h.columns]
            st.dataframe(h2h[existing_cols].reset_index(drop=True))

            # Summarize outcomes from perspective of the selected home team
            def h2h_result(row: pd.Series) -> str:
                # Determine winner of the match
                if row["FTHG"] > row["FTAG"]:
                    winner = row["HomeTeam"]
                elif row["FTHG"] < row["FTAG"]:
                    winner = row["AwayTeam"]
                else:
                    return "Draw"
                return "HomeTeam" if winner == home_team else "AwayTeam"

            results = h2h.apply(h2h_result, axis=1)
            counts = results.value_counts()
            st.write("H2H Summary:")
            st.write(counts)

    with tab4:
        st.subheader("Empirical probability calculator")
        # Build subsets
        subsets = {
            f"HOME last {N_any} (any venue)": subset_lastN_any(base_df, home_team, N_any),
            f"HOME last {N_ha} (home)": subset_lastN_home(base_df, home_team, N_ha),
            f"AWAY last {N_any} (any venue)": subset_lastN_any(base_df, away_team, N_any),
            f"AWAY last {N_ha} (away)": subset_lastN_away(base_df, away_team, N_ha),
            f"H2H last {H2H_years} years": subset_h2h(base_df, home_team, away_team, H2H_years),
        }
        vol = pd.DataFrame([{"Subset": k, "N": len(v)} for k, v in subsets.items()])
        st.dataframe(vol, hide_index=True)
        st.markdown("---")

        # Choose category and time frame
        cat = st.selectbox("Category", ["Goles", "Tarjetas", "Córners", "Hándicap"], index=0)
        # Time frame selection for goals, cards and corners
        time_frame = st.selectbox("Time frame", ["Full match", "First Half", "Second Half"], index=0)

        if cat in ["Goles", "Tarjetas", "Córners"]:
            side_label = st.radio("Over / Under", ["Over", "Under"], horizontal=True)
            side_key = "over" if side_label == "Over" else "under"

            if cat == "Goles":
                scope = st.selectbox("Scope", ["Total match"], index=0)
                col = "goals_total"
                lines = [x + 0.5 for x in range(0, 6)]  # 0.5 to 5.5
            elif cat == "Tarjetas":
                scope = st.selectbox("Scope", ["Total match", "Home", "Away"], index=0)
                if scope == "Total match":
                    col = "cards_total"
                    lines = [x + 0.5 for x in range(0, 8)]  # 0.5 to 7.5
                elif scope == "Home":
                    col = "cards_home"
                    lines = [x + 0.5 for x in range(0, 5)]  # 0.5 to 4.5
                else:
                    col = "cards_away"
                    lines = [x + 0.5 for x in range(0, 5)]
            else:  # Córners
                scope = st.selectbox("Scope", ["Total match", "Home", "Away"], index=0)
                if scope == "Total match":
                    col = "corners_total"
                    lines = [x + 0.5 for x in range(0, 15)]  # 0.5 to 14.5
                elif scope == "Home":
                    col = "corners_home"
                    lines = [x + 0.5 for x in range(0, 11)]  # 0.5 to 10.5
                else:
                    col = "corners_away"
                    lines = [x + 0.5 for x in range(0, 11)]


            line_sel = st.select_slider("Line", options=lines, value=lines[2])

            # Show referee statistics for cards
            if cat == "Tarjetas":
                # Collect list of referees from the full dataset
                refs = df.get("Referee").dropna().unique() if "Referee" in df.columns else []
                if len(refs) > 0:
                    selected_ref = st.selectbox(
                        "Referee (optional)",
                        sorted(list(refs)),
                        index=0,
                        key="ref_select",
                    )
                    # Compute referee card stats using the full dataset (not just base_df) so we have enough matches
                    ref_stats = compute_referee_card_stats(df, selected_ref, home_team, away_team, last_n=30)
                    st.markdown(
                        f"**Referee card stats for {selected_ref} (last {ref_stats['n']} matches):**"
                    )
                    if ref_stats["n"] == 0:
                        st.write("No matches found for this referee in the dataset.")
                    else:
                        # Display probability of exceeding each line
                        prob_rows = []
                        for ln, prob in ref_stats["over_probs"].items():
                            if prob is None:
                                prob_txt = "-"
                            else:
                                prob_txt = f"{prob*100:.1f}%"
                            prob_rows.append({"Line": ln, "P(> line)": prob_txt})
                        st.table(pd.DataFrame(prob_rows))
                        # Display average cards summary
                        summary_rows = [
                            {
                                "Metric": "Average total cards",
                                "Value": f"{ref_stats['avg_total_cards']:.2f}"
                                if ref_stats['avg_total_cards'] is not None
                                else "-",
                            },
                            {
                                "Metric": "Average home cards",
                                "Value": f"{ref_stats['avg_home_cards']:.2f}"
                                if ref_stats['avg_home_cards'] is not None
                                else "-",
                            },
                            {
                                "Metric": "Average away cards",
                                "Value": f"{ref_stats['avg_away_cards']:.2f}"
                                if ref_stats['avg_away_cards'] is not None
                                else "-",
                            },
                            {
                                "Metric": f"Average cards to {home_team}",
                                "Value": f"{ref_stats['avg_cards_to_home_team']:.2f}"
                                if ref_stats['avg_cards_to_home_team'] is not None
                                else "-",
                            },
                            {
                                "Metric": f"Average cards to {away_team}",
                                "Value": f"{ref_stats['avg_cards_to_away_team']:.2f}"
                                if ref_stats['avg_cards_to_away_team'] is not None
                                else "-",
                            },
                        ]
                        st.table(pd.DataFrame(summary_rows))

            # Prepare session state for over/under results
            if "prob_out" not in st.session_state:
                st.session_state["prob_out"] = None
            # When the user clicks calculate, compute probability and fails for each subset
            if st.button("Calculate", key="calc_btn"):
                rows = []
                for name, sdf in subsets.items():
                    # Use custom function to compute probability and failures, passing time_frame
                    res = calc_over_under_for_subset(
                        sdf,
                        cat,
                        scope,
                        float(line_sel),
                        side_key,
                        home_team,
                        away_team,
                        time_frame,
                    )
                    if res["p"] is None:
                        prob_txt = "-"
                    else:
                        prob_txt = f"{res['p']*100:.1f}% ({res['hits']}/{res['n']})"
                    fails_txt = ", ".join(res.get("fails", [])) if res.get("fails") else ""
                    rows.append({
                        "Subset": name,
                        "Probability": prob_txt,
                        "N": res.get("n", 0),
                        "Fails": fails_txt,
                    })
                st.session_state["prob_out"] = rows
            # Display results if available
            if st.session_state.get("prob_out"):
                out = pd.DataFrame(st.session_state["prob_out"])
                st.dataframe(out, hide_index=True)

        else:
            # Handicap
            team_role = st.selectbox("Team", ["Home", "Away"], index=0)
            team_role_key = "home" if team_role == "Home" else "away"
            h_line = st.selectbox("Handicap line", [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], index=2)
            if "handicap_out" not in st.session_state:
                st.session_state["handicap_out"] = None
            if st.button("Calculate", key="handicap_btn"):
                rows = []
                for name, sdf in subsets.items():
                    if "goal_diff" not in sdf.columns:
                        rows.append({"Subset": name, "Probability": "-", "N": len(sdf)})
                        continue
                    res = handicap_prob(sdf, team_role_key, float(h_line))
                    if res["p"] is None:
                        prob_txt = "-"
                    else:
                        prob_txt = f"{res['p']*100:.1f}% ({res['hits']}/{res['n']})"
                    rows.append({"Subset": name, "Probability": prob_txt, "N": res["n"]})
                st.session_state["handicap_out"] = rows
            if st.session_state.get("handicap_out"):
                out = pd.DataFrame(st.session_state["handicap_out"])
                st.dataframe(out, hide_index=True)

    st.caption("Data source: local cleaned CSVs or Supabase/PostgreSQL (if USE_DB=true).")


if __name__ == "__main__":
    main()
