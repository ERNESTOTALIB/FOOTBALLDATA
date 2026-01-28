-- SQL script to create tennis tables for top 100 players analytics
-- This script defines tables for players, matches and enriched fixtures.

-- Table of tennis players
CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    player_api_id VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    country VARCHAR(50),
    age INTEGER,
    ranking INTEGER,
    ranking_points INTEGER,
    hand VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table of tennis matches
CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    fixture_id VARCHAR(50) NOT NULL UNIQUE,
    match_date DATE NOT NULL,
    tournament VARCHAR(100),
    category VARCHAR(50),
    surface VARCHAR(20),
    round VARCHAR(50),
    player1_api_id VARCHAR(50) NOT NULL,
    player2_api_id VARCHAR(50) NOT NULL,
    winner_api_id VARCHAR(50),
    score TEXT,
    sets_won_player1 INTEGER,
    sets_won_player2 INTEGER,
    duration_minutes INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Optionally store detailed stats per match (aces, double faults etc.)
CREATE TABLE IF NOT EXISTS match_stats (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    player_api_id VARCHAR(50),
    aces INTEGER,
    double_faults INTEGER,
    first_serve_percent NUMERIC,
    first_serve_points_won_percent NUMERIC,
    break_points_saved_percent NUMERIC,
    second_serve_points_won_percent NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing enriched fixtures from API (per fixture) if needed
CREATE TABLE IF NOT EXISTS tennis_fixtures_enriched (
    fixture_id VARCHAR(50) PRIMARY KEY,
    referee VARCHAR(100),
    first_half_stats JSONB,
    second_half_stats JSONB,
    events JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
