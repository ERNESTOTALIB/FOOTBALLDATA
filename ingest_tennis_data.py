import os
import requests
import json
import psycopg2
from psycopg2.extras import execute_values


def get_env(key, default=None):
    value = os.getenv(key)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"Missing required environment variable {key}")
    return value

API_BASE = get_env("API_SPORTS_BASE")
API_KEY = get_env("API_SPORTS_KEY")
DATABASE_URL = get_env("DATABASE_URL")

# Construct base URL: ensure https and remove trailing slash
if API_BASE.startswith("http"):
    BASE_URL = API_BASE.rstrip("/")
else:
    BASE_URL = "https://" + API_BASE.rstrip("/")

HOST = API_BASE.replace("https://", "").split("/")[0]

# Tennis-specific base: some APIs have separate path for tennis (e.g. /tennis)
# Adjust TENNIS_BASE_URL if your API uses a different prefix
TENNIS_BASE_URL = f"{BASE_URL}/tennis"


def fetch_last_matches(count: int = 10) -> dict:
    """
    Fetch the last `count` tennis fixtures from the API.
    """
    url = f"{TENNIS_BASE_URL}/fixtures"
    params = {"last": count}
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": HOST,
    }
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def upsert_fixtures(fixtures):
    """
    Insert or update tennis fixtures in the database.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    rows = []
    for f in fixtures:
        # Many APIs nest fixture information under 'fixture', similar to football.
        # Adjust this extraction depending on the actual structure returned by the tennis API.
        fixture_id = f.get("fixture", {}).get("id") or f.get("id")
        rows.append((fixture_id, json.dumps(f)))
    insert_sql = """
        INSERT INTO tennis_fixtures_enriched (fixture_id, data)
        VALUES %s
        ON CONFLICT (fixture_id) DO UPDATE SET data = excluded.data;
    """
    execute_values(cur, insert_sql, rows)
    conn.commit()
    cur.close()
    conn.close()


def main():
    data = fetch_last_matches(20)
    fixtures = data.get("response", data.get("results", []))
    if fixtures:
        upsert_fixtures(fixtures)
        print(f"Upserted {len(fixtures)} tennis fixtures")
    else:
        print("No tennis fixtures fetched")


if __name__ == "__main__":
    main()
