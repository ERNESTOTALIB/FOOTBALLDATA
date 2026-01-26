"""
Streamlit dashboard for exploring probabilistic football models.

This app allows you to select a league, season and specific match to
inspect historical statistics such as average goals, corners and
disciplinary records for each team.  It also displays head-to-head
results over the past five years and applies a simple weighting to
recent form (last 10 matches) when calculating averages.

To use this app you will need to provide a cleaned dataset in CSV
format.  The expected columns are based on the `football-datasets`
repository and include:

    Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR,
    Referee, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR,
    League, Season

See the accompanying documentation for instructions on obtaining
and cleaning the raw data from football-datasets.  Once the CSV
files for each league and season are prepared, you can load them
into this app by adjusting the `DATA_PATHS` dictionary below.

The app does not connect directly to Supabase by default.  If you
prefer to read data from Supabase, replace the `load_data` function
with a version that queries your database using psycopg2 or an
SQLAlchemy engine.  Remember to set the DATABASE_URL environment
variable with your pooler URL (see earlier instructions).
"""

import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
#
# Update this dictionary with the relative or absolute paths to your cleaned
# CSV files for each league.  These files should contain data from the
# last five seasons with the columns described above.  If your files
# live in a subdirectory, include that in the path (e.g., "data/pl.csv").

DATA_PATHS: Dict[str, str] = {
    "Premier League": "data/premier_league_clean.csv",
    "La Liga": "data/la_liga_clean.csv",
    "Serie A": "data/serie_a_clean.csv",
    "Bundesliga": "data/bundesliga_clean.csv",
    "Ligue 1": "data/ligue_1_clean.csv",
}

# Map human-friendly league names to the codes stored in the database.
# The ingestion pipeline writes the ``league`` column using codes like E0, SP1, etc.
# See data_fetcher.py::LEAGUES for the canonical mapping.
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


@st.cache_data
def load_data(league: str) -> pd.DataFrame:
    """Load match data for the given league.

    Depending on configuration this function reads from a PostgreSQL database
    or from a local CSV file.  When connecting to the database, the league
    argument will be mapped to its code before filtering the query.

    Parameters
    ----------
    league : str
        Name of the league (must exist in DATA_PATHS).

    Returns
    -------
    pd.DataFrame
        DataFrame containing all matches for the league across the
        available seasons.
    """
    use_db = os.environ.get("USE_DB", "false").lower() == "true"
    db_url = os.environ.get("DATABASE_URL")
    if use_db and db_url:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError:
            st.error(
                "psycopg2 is required to load data from the database.\n"
                "Install it by adding 'psycopg2-binary' to your requirements."
            )
            return pd.DataFrame()
        try:
            conn = psycopg2.connect(db_url)
        except Exception as e:
            st.error(f"No se pudo conectar a la base de datos: {e}")
            return pd.DataFrame()
        query = """
            SELECT
              m.match_date AS Date,
              m.home_team AS HomeTeam,
              m.away_team AS AwayTeam,
              m.full_time_home_goals AS FTHG,
              m.full_time_away_goals AS FTAG,
              m.full_time_result AS FTR,
              s.home_corners AS HC,
              s.away_corners AS AC,
              s.home_yellows AS HY,
              s.away_yellows AS AY,
              s.home_reds AS HR,
              s.away_reds AS AR,
              m.league AS League,
              m.season AS Season
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

        # Normalize column names if needed (snake_case to expected names)
        if "Date" not in df.columns:
            for c in ["date", "match_date", "utc_date", "fixture_date"]:
                if c in df.columns:
                    df["Date"] = df[c]
                    break

        if "HomeTeam" not in df.columns:
            for c in ["home_team", "home", "team_home"]:
                if c in df.columns:
                    df["HomeTeam"] = df[c]
                    break

        if "AwayTeam" not in df.columns:
            for c in ["away_team", "away", "team_away"]:
                if c in df.columns:
                    df["AwayTeam"] = df[c]
                    break

        # Goals
        if "FTHG" not in df.columns:
            for c in ["home_goals", "home_score", "hg", "goals_home"]:
                if c in df.columns:
                    df["FTHG"] = df[c]
                    break

        if "FTAG" not in df.columns:
            for c in ["away_goals", "away_score", "ag", "goals_away"]:
                if c in df.columns:
                    df["FTAG"] = df[c]
                    break

        # Corners
        if "HC" not in df.columns:
            for c in ["home_corners", "corners_home"]:
                if c in df.columns:
                    df["HC"] = df[c]
                    break

        if "AC" not in df.columns:
            for c in ["away_corners", "corners_away"]:
                if c in df.columns:
                    df["AC"] = df[c]
                    break

        # Yellow cards
        if "HY" not in df.columns:
            for c in ["home_yellow", "home_yellow_cards", "yellow_home"]:
                if c in df.columns:
                    df["HY"] = df[c]
                    break

        if "AY" not in df.columns:
            for c in ["away_yellow", "away_yellow_cards", "yellow_away"]:
                if c in df.columns:
                    df["AY"] = df[c]
                    break

        # Red cards
        if "HR" not in df.columns:
            for c in ["home_red", "home_red_cards", "red_home"]:
                if c in df.columns:
                    df["HR"] = df[c]
                    break

        if "AR" not in df.columns:
            for c in ["away_red", "away_red_cards", "red_away"]:
                if c in df.columns:
                    df["AR"] = df[c]
                    break

        # Validate essential columns; if missing, report and stop
        required = ["Date", "HomeTeam", "AwayTeam"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(
                f"Faltan columnas requeridas: {missing}. Columnas disponibles: {list(df.columns)}"
            )
            return pd.DataFrame()

        # Coerce Date to datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df

    # Fallback to local CSV
    file_path = DATA_PATHS.get(league)
    if not file_path:
        st.error(f"No data path configured for {league}")
        return pd.DataFrame()
    path = Path(file_path)
    if not path.exists():
        st.error(
            f"Data file for {league} not found at {path}.\n"
            "Please place the cleaned CSV in the specified location."
        )
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"])
    # Ensure expected columns exist; if not, warn the user.
    required_cols = {
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HC",
        "AC",
        "HY",
        "AY",
        "HR",
        "AR",
    }
    missing = required_cols - set(df.columns)
    if missing:
        st.warning(f"Missing expected columns: {missing}")
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

    # Stats split by home/away
    home_matches = form[form["HomeTeam"] == team]
    away_matches = form[form["AwayTeam"] == team]

    def safe_weighted(series: pd.Series, w: np.ndarray) -> float:
        if series.empty:
            return float("nan")
        w_local = w[-len(series) :]
        return weighted_average(series.values.astype(float), w_local.astype(float))

    stats = {
        "weighted_goals_for_home": safe_weighted(home_matches["FTHG"], weights),
        "weighted_goals_against_home": safe_weighted(home_matches["FTAG"], weights),
        "weighted_goals_for_away": safe_weighted(away_matches["FTAG"], weights),
        "weighted_goals_against_away": safe_weighted(away_matches["FTHG"], weights),
        "weighted_corners_home": safe_weighted(home_matches.get("HC", pd.Series(dtype=float)), weights),
        "weighted_corners_away": safe_weighted(away_matches.get("AC", pd.Series(dtype=float)), weights),
        "weighted_yellows_home": safe_weighted(home_matches.get("HY", pd.Series(dtype=float)), weights),
        "weighted_yellows_away": safe_weighted(away_matches.get("AY", pd.Series(dtype=float)), weights),
        "weighted_reds_home": safe_weighted(home_matches.get("HR", pd.Series(dtype=float)), weights),
        "weighted_reds_away": safe_weighted(away_matches.get("AR", pd.Series(dtype=float)), weights),
    }
    return stats


def get_h2h(df: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Get head-to-head matches between two teams over past H2H_YEARS."""
    cutoff = df["Date"].max() - pd.DateOffset(years=H2H_YEARS)
    mask = (
        ((df["HomeTeam"] == home) & (df["AwayTeam"] == away))
        | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
    ) & (df["Date"] >= cutoff)
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

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Season selection
    seasons = sorted(df["Season"].dropna().unique())
    season = st.sidebar.selectbox("Season", seasons)
    season_df = df[df["Season"] == season].copy()
    season_df.sort_values("Date", inplace=True)

    # Match selection
    match_options = season_df.apply(
        lambda r: f"{r['Date'].date()} — {r['HomeTeam']} vs {r['AwayTeam']}", axis=1
    ).tolist()
    match_idx = st.sidebar.selectbox("Match", range(len(match_options)), format_func=lambda i: match_options[i])
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

            # Simple win/draw/loss count from perspective of home_team
            def h2h_result(row: pd.Series) -> str:
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

    st.caption("Data source: local cleaned CSVs or Supabase/PostgreSQL (if USE_DB=true).")


if __name__ == "__main__":
    main()
