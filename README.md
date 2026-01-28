# Sports Analytics Platform

This repository contains code to ingest, process, and visualize football and tennis match data.

## Football Module

The football component uses historical data from [Football‑Data.co.uk](https://football-data.co.uk) and enriches it with match events and referee statistics via the API-Football (API-Sports). The Streamlit front-end calculates empirical probabilities for various markets (goals, cards, corners, handicap) and allows the user to simulate virtual matches.

### Running the Football Dashboard

Install the dependencies from `requirements.txt`. Create a `.streamlit/secrets.toml` file or export environment variables with your database URL and API keys. To launch the dashboard locally:

```bash
streamlit run streamlit_app.py
```

### Ingesting Football Data

The ingestion pipeline pulls recent fixtures from API-Football and loads them into Supabase. To ingest data manually, run:

```bash
python ingest_api_sports.py
```

In production the ingestion is automated via GitHub Actions (see `.github/workflows/ingest_api_sports.yml`). Ensure that the following secrets are set in **GitHub → Settings → Secrets and variables → Actions**:

- `API_SPORTS_BASE` – Base URL for the API (e.g. `https://api-football-v1.p.rapidapi.com/v3`).
-# Sports Analytics Platform

This repository contains code to ingest, process, and visualize football and tennis match data.

## Football Module

The football component uses historical data from [Football‑Data.co.uk](https://football-data.co.uk) and enriches it with match events and referee statistics via the API-Football (API-Sports). The Streamlit front-end calculates empirical probabilities for various markets (goals, cards, corners, handicap) and allows the user to simulate virtual matches.

### Running the Football Dashboard

Install the dependencies from `requirements.txt`. Create a `.streamlit/secrets.toml` file or export environment variables with your database URL and API keys. To launch the dashboard locally:

```bash
streamlit run streamlit_app.py
```

### Ingesting Football Data

The ingestion pipeline pulls recent fixtures from API-Football and loads them into Supabase. To ingest data manually, run:

```bash
python ingest_api_sports.py
```

In production the ingestion is automated via GitHub Actions (see `.github/workflows/ingest_api_sports.yml`). Ensure that the following secrets are set in **GitHub → Settings → Secrets and variables → Actions**:

- `API_SPORTS_BASE` – Base URL for the API (e.g. `https://api-football-v1.p.rapidapi.com/v3`).
- `API_SPORTS_KEY` – Your RapidAPI key for API-Football.
- `DATABASE_URL` – Connection string for your Supabase PostgreSQL database.

## Tennis Module (New)

The tennis module aims to collect comprehensive statistics for ATP Top 100 players and provide a dashboard to compare any two players. Data is fetched using the API-Sports Tennis endpoints and stored in Supabase. A skeleton ingestion script (`ingest_tennis_data.py`) and Streamlit dashboard (`streamlit_tennis_app.py`) are provided.

### Setting Up Tennis Tables

Run the SQL script in [`sql/create_tennis_tables.sql`](sql/create_tennis_tables.sql) on your PostgreSQL database to create the necessary tables (`players`, `matches`, `match_stats`, `tennis_fixtures_enriched`) before ingesting data.

### Running the Tennis Dashboard

After ingesting data, start the dashboard with:

```bash
streamlit run streamlit_tennis_app.py
```

The UI allows you to select two players and shows recent matches, head‑to‑head results and basic statistics. Future enhancements (e.g. detailed stats, social media sentiment) can be added once data is available.

### Automating Tennis Ingestion

The workflow `.github/workflows/ingest_tennis_data.yml` runs daily at 04:00 UTC to download the latest tennis fixtures via API-Sports. To enable it, set the same secrets used for football (`API_SPORTS_BASE`, `API_SPORTS_KEY`, `DATABASE_URL`) in your GitHub repository. If the tennis API uses a different base URL, create an additional secret (`API_SPORTS_BASE_TENNIS`) and update the workflow accordingly.

## Development Notes

- This project uses **Supabase** for storage. If running locally, ensure your database is reachable and update `DATABASE_URL`.
- Dependencies are listed in `requirements.txt`. Use `pip install -r requirements.txt` to set up your environment.
- The ingestion scripts and workflows are designed to fail gracefully if API endpoints are unavailable or data is missing.
- Feel free to contribute by adding new data sources (e.g. WTA, Challenger circuits, social media sentiment) or improving the dashboard.
 `API_SPORTS_KEY` – Your RapidAPI key for API-Football.
- `DATABASE_URL` – Connection string for your Supabase PostgreSQL database.

## Tennis Module (New)

The tennis module aims to collect comprehensive statistics for ATP Top 100 players and provide a dashboard to compare any two players. Data is fetched using the API-Sports Tennis endpoints and stored in Supabase. A skeleton ingestion script (`ingest_tennis_data.py`) and Streamlit dashboard (`streamlit_tennis_app.py`) are provided.

### Setting Up Tennis Tables

Run the SQL script in [`sql/create_tennis_tables.sql`](sql/create_tennis_tables.sql) on your PostgreSQL database to create the necessary tables (`players`, `matches`, `match_stats`, `tennis_fixtures_enriched`) before ingesting data.

### Running the Tennis Dashboard

After ingesting data, start the dashboard with:

```bash
streamlit run streamlit_tennis_app.py
```

The UI allows you to select two players and shows recent matches, head‑to‑head results and basic statistics. Future enhancements (e.g. detailed stats, social media sentiment) can be added once data is available.

### Automating Tennis Ingestion

The workflow `.github/workflows/ingest_tennis_data.yml` runs daily at 04:00 UTC to download the latest tennis fixtures via API-Sports. To enable it, set the same secrets used for football (`API_SPORTS_BASE`, `API_SPORTS_KEY`, `DATABASE_URL`) in your GitHub repository. If the tennis API uses a different base URL, create an additional secret (`API_SPORTS_BASE_TENNIS`) and update the workflow accordingly.

## Development Notes

- This project uses **Supabase** for storage. If running locally, ensure your database is reachable and update `DATABASE_URL`.
- Dependencies are listed in `requirements.txt`. Use `pip install -r requirements.txt` to set up your environment.
- The ingestion scripts and workflows are designed to fail gracefully if API endpoints are unavailable or data is missing.
- Feel free to contribute by adding new data sources (e.g. WTA, Challenger circuits, social media sentiment) or improving the dashboard.
