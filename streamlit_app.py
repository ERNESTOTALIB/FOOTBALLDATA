"""
Streamlit dashboard for exploring probabilistic football models.

This app allows you to select a league, season and specific match to
inspect historical statistics such as average goals, corners and
disciplinary records for each team.  It also displays head‑to‑head
results over the past five years and applies a simple weighting to
recent form (last 10 matches) when calculating averages.

To use this app you will need to provide a cleaned dataset in CSV
format.  The expected columns are based on the `football‑datasets`
repository and include:

    Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR,
    Referee, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR,
    League, Season

See the accompanying documentation for instructions on obtaining
and cleaning the raw data from football‑datasets.  Once the CSV
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

#
# By default this app loads data from pre‑cleaned CSV files located on disk.
# To connect directly to your Supabase/PostgreSQL database, set the
# environment variable ``USE_DB`` to ``true`` (case insensitive) and
# define ``DATABASE_URL`` with your pooler connection string.  When
# USE_DB is enabled, the app will query the ``matches``, ``match_stats``
# and ``team_stats`` tables directly from the database instead of
# reading local CSVs.  Otherwise it falls back to loading from
# DATA_PATHS.

DATA_PATHS: Dict[str, str] = {
    "Premier League": "data/premier_league_clean.csv",
    "La Liga": "data/la_liga_clean.csv",
    "Serie A": "data/serie_a_clean.csv",
    "Bundesliga": "data/bundesliga_clean.csv",
    "Ligue 1": "data/ligue_1_clean.csv",
}

# Number of years to look back for head‑to‑head statistics
H2H_YEARS = 5
# Number of matches to use for form weighting
FORM_MATCHES = 10


@st.cache_data
def load_data(league: str) -> pd.DataFrame:
    """Load cleaned match data for the given league.

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
    """Load match data for the given league.

    If the environment variable USE_DB is set to "true" (case
    insensitive) and DATABASE_URL is defined, this function will
    connect to the database and query the matches and match_stats
    tables for the specified league.  Otherwise it loads the data
    from local CSV files defined in DATA_PATHS.

    Parameters
    ----------
    league : str
        Name of the league (must exist in DATA_PATHS when not using DB).

    Returns
    -------
    pd.DataFrame
        DataFrame containing all matches for the league across the
        available seasons.  The returned columns are aligned to
        HomeTeam, AwayTeam, FTHG, FTAG, HC, AC, HY, AY, HR, AR,
        FTR, Date, League, Season as expected by the rest of the app.
    """
    use_db = os.environ.get("USE_DB", "false").lower() == "true"
    db_url = os.environ.get("DATABASE_URL")
    if use_db and db_url:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError:
            st.error("psycopg2 is required to load data from the database.\n"
                     "Install it by adding 'psycopg2-binary' to your requirements.")
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
                cur.execute(query, (league,))
                records = cur.fetchall()
            df = pd.DataFrame(records)
        except Exception as e:
            st.error(f"Error al consultar los datos: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
        # Ensure date column is datetime
        # --- Normaliza nombres de columnas (DB suele venir en snake_case) ---
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
        
        # Goles
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
        
        # Córners (si tu app los usa)
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
        
        # Amarillas
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
        
        # Rojas
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
        
        # --- Validaciones mínimas para evitar errores silenciosos ---
        required = ["Date", "HomeTeam", "AwayTeam"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Faltan columnas requeridas: {missing}. Columnas disponibles: {list(df.columns)}")
        
        # --- Parseo de tipos ---
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
    required = {
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
    missing = required - set(df.columns)
    if missing:
        st.warning(f"Missing expected columns: {missing}")
    return df


def team_form(df: pd.DataFrame, team: str, home: bool) -> pd.DataFrame:
    """Return the last FORM_MATCHES matches for the team with a simple weight.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset for a league.
    team : str
        The team to filter on.
    home : bool
        Whether to consider home (True) or away (False) matches.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the last N matches with weights. Includes
        columns for goals, corners and cards.
    """
    if home:
        matches = df[df["HomeTeam"] == team].sort_values("Date", ascending=False).head(FORM_MATCHES)
        goals_for = matches["FTHG"]
        goals_against = matches["FTAG"]
        corners = matches[["HC", "AC"]].sum(axis=1)
        yellows = matches[["HY", "AY"]].sum(axis=1)
        reds = matches[["HR", "AR"]].sum(axis=1)
    else:
        matches = df[df["AwayTeam"] == team].sort_values("Date", ascending=False).head(FORM_MATCHES)
        goals_for = matches["FTAG"]
        goals_against = matches["FTHG"]
        corners = matches[["HC", "AC"]].sum(axis=1)
        yellows = matches[["HY", "AY"]].sum(axis=1)
        reds = matches[["HR", "AR"]].sum(axis=1)
    # Create weights: most recent match gets weight=FORM_MATCHES, then descending
    weights = np.linspace(FORM_MATCHES, 1, len(matches))
    return pd.DataFrame({
        "Date": matches["Date"],
        "GoalsFor": goals_for,
        "GoalsAgainst": goals_against,
        "Corners": corners,
        "Yellows": yellows,
        "Reds": reds,
        "Weight": weights,
    })


def weighted_average(series: pd.Series, weights: pd.Series) -> float:
    """Compute weighted average of a series. Return NaN if empty."""
    if series.empty:
        return float("nan")
    return (series * weights).sum() / weights.sum()


def compute_team_stats(df: pd.DataFrame, team: str) -> Dict[str, float]:
    """Calculate season averages and weighted form for a team.

    Returns a dictionary with keys:
        - goals_for
        - goals_against
        - corners
        - yellows
        - reds
        - goals_for_home / goals_against_home
        - goals_for_away / goals_against_away
        - weighted_* for recent form (home and away)
    """
    stats = {}
    home_matches = df[df["HomeTeam"] == team]
    away_matches = df[df["AwayTeam"] == team]
    # Season averages (unweighted)
    stats["goals_for_season"] = (home_matches["FTHG"].mean() + away_matches["FTAG"].mean()) / 2
    stats["goals_against_season"] = (home_matches["FTAG"].mean() + away_matches["FTHG"].mean()) / 2
    stats["corners_season"] = (home_matches[["HC", "AC"]].sum(axis=1).mean() + away_matches[["HC", "AC"]].sum(axis=1).mean()) / 2
    stats["yellows_season"] = (home_matches[["HY", "AY"]].sum(axis=1).mean() + away_matches[["HY", "AY"]].sum(axis=1).mean()) / 2
    stats["reds_season"] = (home_matches[["HR", "AR"]].sum(axis=1).mean() + away_matches[["HR", "AR"]].sum(axis=1).mean()) / 2
    # Home/away averages
    stats["goals_for_home"] = home_matches["FTHG"].mean()
    stats["goals_against_home"] = home_matches["FTAG"].mean()
    stats["goals_for_away"] = away_matches["FTAG"].mean()
    stats["goals_against_away"] = away_matches["FTHG"].mean()
    # Weighted recent form
    form_home = team_form(df, team, home=True)
    form_away = team_form(df, team, home=False)
    stats["weighted_goals_for_home"] = weighted_average(form_home["GoalsFor"], form_home["Weight"])
    stats["weighted_goals_against_home"] = weighted_average(form_home["GoalsAgainst"], form_home["Weight"])
    stats["weighted_goals_for_away"] = weighted_average(form_away["GoalsFor"], form_away["Weight"])
    stats["weighted_goals_against_away"] = weighted_average(form_away["GoalsAgainst"], form_away["Weight"])
    stats["weighted_corners_home"] = weighted_average(form_home["Corners"], form_home["Weight"])
    stats["weighted_corners_away"] = weighted_average(form_away["Corners"], form_away["Weight"])
    stats["weighted_yellows_home"] = weighted_average(form_home["Yellows"], form_home["Weight"])
    stats["weighted_yellows_away"] = weighted_average(form_away["Yellows"], form_away["Weight"])
    stats["weighted_reds_home"] = weighted_average(form_home["Reds"], form_home["Weight"])
    stats["weighted_reds_away"] = weighted_average(form_away["Reds"], form_away["Weight"])
    return stats


def get_h2h(df: pd.DataFrame, team1: str, team2: str) -> pd.DataFrame:
    """Return head‑to‑head matches between team1 and team2 in the last H2H_YEARS.

    The returned DataFrame includes date, home team, away team and final score.
    """
    cutoff = pd.to_datetime("today") - pd.DateOffset(years=H2H_YEARS)
    mask = (
        ((df["HomeTeam"] == team1) & (df["AwayTeam"] == team2))
        | ((df["HomeTeam"] == team2) & (df["AwayTeam"] == team1))
    )
    h2h = df[mask & (df["Date"] >= cutoff)][
        ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    ].sort_values("Date", ascending=False)
    return h2h


def main() -> None:
    st.title("Análisis Probabilístico de Fútbol – Top 5 Ligas")
    st.sidebar.header("Configuración de la consulta")
    # Selección de liga
    league = st.sidebar.selectbox("Liga", list(DATA_PATHS.keys()))
    df = load_data(league)
    if df.empty:
        st.warning("No hay datos disponibles para la liga seleccionada."
                   " Asegúrate de cargar los ficheros CSV limpiados en la ruta indicada.")
        return
    # Selección de temporada (opcional)
    seasons = sorted(df["Season"].dropna().unique())
    season = st.sidebar.selectbox("Temporada", seasons + ["Todas"])
    filtered_df = df if season == "Todas" else df[df["Season"] == season]
    teams = sorted(
        set(filtered_df["HomeTeam"]) | set(filtered_df["AwayTeam"])
    )
    # Selección de partidos (equipos)
    home_team = st.sidebar.selectbox("Equipo local", teams)
    # Excluir al equipo local de las opciones de visitante
    away_teams = [t for t in teams if t != home_team]
    away_team = st.sidebar.selectbox("Equipo visitante", away_teams)
    # Mostrar estadísticas al confirmar
    if st.sidebar.button("Mostrar análisis"):
        st.header(f"{home_team} vs {away_team}")
        # Calcular estadísticas para ambos equipos
        home_stats = compute_team_stats(filtered_df, home_team)
        away_stats = compute_team_stats(filtered_df, away_team)
        # Tabla de promedios de temporada
        st.subheader("Medias de la temporada")
        season_table = pd.DataFrame([
            {
                "Equipo": home_team,
                "Goles a favor": f"{home_stats['goals_for_season']:.2f}",
                "Goles en contra": f"{home_stats['goals_against_season']:.2f}",
                "Córners": f"{home_stats['corners_season']:.2f}",
                "Amarillas": f"{home_stats['yellows_season']:.2f}",
                "Rojas": f"{home_stats['reds_season']:.2f}",
            },
            {
                "Equipo": away_team,
                "Goles a favor": f"{away_stats['goals_for_season']:.2f}",
                "Goles en contra": f"{away_stats['goals_against_season']:.2f}",
                "Córners": f"{away_stats['corners_season']:.2f}",
                "Amarillas": f"{away_stats['yellows_season']:.2f}",
                "Rojas": f"{away_stats['reds_season']:.2f}",
            },
        ])
        st.table(season_table.set_index("Equipo"))
        # Forma reciente ponderada
        st.subheader(f"Forma reciente ponderada (últimos {FORM_MATCHES} partidos)")
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
        # Enfrentamientos directos
        st.subheader(f"Enfrentamientos directos (últimos {H2H_YEARS} años)")
        h2h = get_h2h(filtered_df, home_team, away_team)
        if h2h.empty:
            st.info("No hay enfrentamientos recientes entre estos equipos en el periodo analizado.")
        else:
            # Construir una columna Resultado concatenando goles
            h2h_display = h2h.copy()
            h2h_display["Resultado"] = h2h_display["FTHG"].astype(str) + "-" + h2h_display["FTAG"].astype(str)
            h2h_display = h2h_display[["Date", "HomeTeam", "AwayTeam", "Resultado"]]
            h2h_display.columns = ["Fecha", "Local", "Visitante", "Marcador"]
            st.dataframe(h2h_display)
    st.write(""
        "Notas:\n"
        "- Esta app usa promedios simples y ponderados para ilustrar estadísticas de los equipos.\n"
        "- El modelo probabilístico de apuestas (por ejemplo Poisson) deberá calcularse aparte y añadirse como columnas adicionales en tu dataset limpiado."
    )


if __name__ == "__main__":
    main()
