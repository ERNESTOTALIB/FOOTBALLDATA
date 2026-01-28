import os
import requests
import json
import psycopg2
from psycopg2.extras import execute_values


def get_env(key, default=None):
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Missing required environment variable {key}")
    return value


API_BASE = get_env("API_SPORTS_BASE")
API_KEY = get_env("API_SPORTS_KEY")
DATABASE_URL = get_env("DATABASE_URL")

# Construct base URL and host for API-Football (API-Sports)
# Remove scheme if provided and ensure we prepend https:// only once
if API_BASE.startswith("http"):
    BASE_URL = API_BASE.rstrip("/")
else:
    BASE_URL = "https://" + API_BASE.rstrip("/")

HOST = API_BASE.replace("https://", "").split("/")[0]

def fetch_last_matches(count: int = 10) -> dict:
    """
    Fetch the last `count` fixtures from the API-Football service.

    Args:
        count (int): Number of fixtures to fetch.

    Returns:
        dict: JSON response from the API.
    """
    url = f"{BASE_URL}/fixtures"
    params = {"last": count}
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": HOST,
    }
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()



def upsert_fixtures(fixtures):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    # Insert or update each fixture as JSON data.
    rows = [(f["fixture"]["id"], json.dumps(f)) for f in fixtures]
    insert_sql = """
        INSERT INTO fixtures_enriched (fixture_id, data)
        VALUES %s
        ON CONFLICT (fixture_id) DO UPDATE SET data = excluded.data;
    """
    execute_values(cur, insert_sql, rows)
    conn.commit()
    cur.close()
    conn.close()


def main():
    data = fetch_last_matches(10)
    fixtures = data.get("response", [])
    if fixtures:
        upsert_fixtures(fixtures)
        print(f"Upserted {len(fixtures)} fixtures")
    else:
        print("No fixtures fetched")


if __name__ == "__main__":
    main()
