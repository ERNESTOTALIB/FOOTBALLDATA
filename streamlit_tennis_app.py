import os
import pandas as pd
import streamlit as st
import psycopg2

@st.cache_data
def get_conn():
    """
    Establishes a connection to the PostgreSQL database using the DATABASE_URL
    environment variable. Connections are cached by Streamlit to improve
    performance across interactions.
    """
    return psycopg2.connect(os.environ.get("DATABASE_URL"))

@st.cache_data
def get_players():
    """
    Fetches the list of players from the `players` table.
    The table is expected to contain at least `player_id`, `name` and `ranking` fields.
    """
    conn = get_conn()
    df = pd.read_sql("SELECT player_id, name FROM players ORDER BY ranking", conn)
    conn.close()
    return df

@st.cache_data
def get_recent_matches(player_id: int, limit: int = 10) -> pd.DataFrame:
    """
    Retrieves the most recent matches for a given player.

    Args:
        player_id (int): The unique identifier of the player.
        limit (int): Number of recent matches to fetch.

    Returns:
        pd.DataFrame: DataFrame containing match date, opponent, result, surface and tournament.
    """
    conn = get_conn()
    query = f"""
        SELECT date,
               CASE WHEN player1_id = {player_id} THEN player2_name ELSE player1_name END AS opponent,
               CASE WHEN winner_id = {player_id} THEN 'W' ELSE 'L' END AS result,
               surface,
               tournament
        FROM matches
        WHERE player1_id = {player_id} OR player2_id = {player_id}
        ORDER BY date DESC
        LIMIT {limit};
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

@st.cache_data
def get_h2h(player1_id: int, player2_id: int) -> pd.Series:
    """
    Computes head-to-head statistics between two players.

    Args:
        player1_id (int): The unique identifier of the first player.
        player2_id (int): The unique identifier of the second player.

    Returns:
        pd.Series: Series containing the number of matches, wins for player1 and wins for player2.
    """
    conn = get_conn()
    query = f"""
        SELECT COUNT(*) AS matches,
               SUM(CASE WHEN winner_id = {player1_id} THEN 1 ELSE 0 END) AS player1_wins,
               SUM(CASE WHEN winner_id = {player2_id} THEN 1 ELSE 0 END) AS player2_wins
        FROM matches
        WHERE (player1_id = {player1_id} AND player2_id = {player2_id})
           OR (player1_id = {player2_id} AND player2_id = {player1_id});
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df.iloc[0]

def main() -> None:
    """Main entry point for the Streamlit tennis dashboard."""
    st.set_page_config(page_title="Tennis Dashboard", page_icon="🎾", layout="wide")
    st.title("Tennis Dashboard - ATP Top 100")

    # Load players
    players_df = get_players()
    if players_df.empty:
        st.error("No players found. Please ensure the database is populated with tennis data.")
        return

    player_names = players_df["name"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        player1_name = st.selectbox("Select Player 1", player_names)
    with col2:
        default_index = 1 if len(player_names) > 1 else 0
        player2_name = st.selectbox("Select Player 2", player_names, index=default_index)

    player1_id = players_df.loc[players_df["name"] == player1_name, "player_id"].iloc[0]
    player2_id = players_df.loc[players_df["name"] == player2_name, "player_id"].iloc[0]

    # Display recent matches for each player
    st.subheader(f"{player1_name} - Recent Matches")
    recent1 = get_recent_matches(player1_id)
    if recent1.empty:
        st.write("No recent matches found for this player.")
    else:
        st.dataframe(recent1)

    st.subheader(f"{player2_name} - Recent Matches")
    recent2 = get_recent_matches(player2_id)
    if recent2.empty:
        st.write("No recent matches found for this player.")
    else:
        st.dataframe(recent2)

    # Head-to-head comparison
    st.subheader(f"Head-to-Head: {player1_name} vs {player2_name}")
    h2h_stats = get_h2h(player1_id, player2_id)
    if int(h2h_stats["matches"]) == 0:
        st.write("These players have not faced each other yet.")
    else:
        st.metric(label="Total Matches", value=int(h2h_stats["matches"]))
        st.metric(label=f"{player1_name} Wins", value=int(h2h_stats["player1_wins"]))
        st.metric(label=f"{player2_name} Wins", value=int(h2h_stats["player2_wins"]))

if __name__ == "__main__":
    main()
