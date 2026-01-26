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
    original_cols = list(df.columns)
    cleaned_cols = [str(c).strip() for c in original_cols]
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
    }

    # 3) Build a rename dict from existing columns to canonical columns
    rename_dict = {}
    for lower_name, canonical in canonical_by_lower.items():
        if lower_name in lower_map:
            rename_dict[lower_map[lower_name]] = canonical

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # 4) Parse/ensure Date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


@st.cache_data
def load_data(league: str) -> pd.DataFrame:
    """Load match data for the given league (DB if enabled, else local CSV)."""
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
    """Compute full-time result of a match based on goals."""
    if row["FTHG"] > row["FTAG"]:
        return "H"
    if row["FTHG"] < row["FTAG"]:
        return "A"
    return "D"


def team_form(df: pd.DataFrame, team: str, n: int = FORM_MATCHES) -> pd.DataFrame:
    """Return last n matches for team with outcome and goals."""
    team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    team_matches.sort_values("Date", inplace=True)
    team_matches = team_matches.tail(n)
    team_matches["Outcome"] = team_matches.apply(compute_outcome, axis=1)
    return team_matches


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted average with fallback if weights sum to zero."""
    if len(values) == 0:
        return float("nan")
    w_sum = weights.sum()
    if w_sum == 0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / w_sum)


def compute_team_stats(df: pd.DataFrame, team: str) -> Dict[str, float]:
    """Compute weighted stats for a team based on last N matches."""
    form = team_form(df, team, FORM_MATCHES)
    if form.empty:
        return {}

    # Newest matches get higher weight
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
    """Get head-to-head matches between two teams over past H2H_YEARS."""
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

    # Season selection
    seasons = sorted(df["Season"].dropna().unique()) if "Season" in df.columns else []
    if not seasons:
        st.error("No hay temporadas disponibles en los datos cargados.")
        st.stop()

    season = st.sidebar.selectbox("Season", seasons)
    season_df = df[df["Season"] == season].copy()
    season_df.sort_values("Date", inplace=True)

    # Match selection
    match_options = season_df.apply(
        lambda r: f"{r['Date'].date()} — {r['HomeTeam']} vs {r['AwayTeam']}", axis=1
    ).tolist()

    if not match_options:
        st.error("No hay partidos disponibles para la temporada seleccionada.")
        st.stop()

    match_idx = st.sidebar.selectbox(
        "Match", range(len(match_options)), format_func=lambda i: match_options[i]
    )
    match_row = season_df.iloc[match_idx]
    home_team = match_row["HomeTeam"]
    away_team = match_row["AwayTeam"]

    st.header(f"{home_team} vs {away_team}")
    st.write(f"Date: {match_row['Date'].date()} | Season: {season} | League: {league}")

    tab1, tab2, tab3 = st.tabs(["Match Info", "Team Stats", "Head-to-Head"])

    with tab1:
        st.subheader("Match details")
        st.table(pd.DataFrame([match_row]))

    with tab2:
        st.subheader("Weighted team statistics (last 10 matches)")
        home_stats = compute_team_stats(season_df, home_team)
        away_stats = compute_team_stats(season_df, away_team)

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

    with tab3:
        st.subheader("Head-to-head results (past 5 years)")
        h2h = get_h2h(df, home_team, away_team)
        if h2h.empty:
            st.write("No head-to-head matches found in the selected period.")
        else:
            display_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HC", "AC", "HY", "AY", "HR", "AR"]
            existing_cols = [c for c in display_cols if c in h2h.columns]
            st.dataframe(h2h[existing_cols].reset_index(drop=True))

    st.caption("Data source: local cleaned CSVs or Supabase/PostgreSQL (if USE_DB=true).")


if __name__ == "__main__":
    main()
