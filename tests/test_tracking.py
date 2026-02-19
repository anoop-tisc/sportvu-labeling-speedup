"""Tests for tracking analytics tools using real game data."""

import pytest
from src.sportvu_loader import load_game
from src.tracking_tools import (
    get_player_positions,
    get_ball_handler,
    detect_passes,
    get_defensive_matchups,
    get_player_trajectory,
    get_ball_trajectory,
)

GAME_PATH = "data/0021500492.json"


@pytest.fixture(scope="module")
def game():
    return load_game(GAME_PATH)


class TestGetPlayerPositions:
    def test_returns_10_players(self, game):
        result = get_player_positions(game, 1, 660.0)
        assert "error" not in result
        assert len(result["players"]) == 10

    def test_has_ball_data(self, game):
        result = get_player_positions(game, 1, 660.0)
        assert result["ball"] is not None
        assert "x" in result["ball"]
        assert "z" in result["ball"]

    def test_players_have_positions(self, game):
        result = get_player_positions(game, 1, 660.0)
        for p in result["players"]:
            assert "position" in p
            assert "landmark" in p
            assert "distance_to_basket" in p
            assert "team" in p
            assert "name" in p

    def test_invalid_period(self, game):
        result = get_player_positions(game, 9, 660.0)
        assert "error" in result


class TestGetBallHandler:
    def test_returns_handlers(self, game):
        result = get_ball_handler(game, 1, 690.0, 660.0)
        assert "handlers" in result
        assert len(result["handlers"]) > 0

    def test_handlers_have_names(self, game):
        result = get_ball_handler(game, 1, 690.0, 660.0)
        named = [h for h in result["handlers"] if h.get("handler")]
        assert len(named) > 0
        for h in named:
            assert "team" in h
            assert "start_gc" in h
            assert "end_gc" in h

    def test_handler_at_tipoff(self, game):
        # Right after tipoff in Q1, someone should have the ball
        result = get_ball_handler(game, 1, 712.0, 708.0)
        named = [h for h in result["handlers"] if h.get("handler")]
        assert len(named) > 0


class TestDetectPasses:
    def test_finds_passes(self, game):
        # 30-second window should have some passes
        result = detect_passes(game, 1, 690.0, 660.0)
        assert "passes" in result
        assert len(result["passes"]) > 0

    def test_pass_has_structure(self, game):
        result = detect_passes(game, 1, 690.0, 660.0)
        if result["passes"]:
            p = result["passes"][0]
            assert "passer" in p
            assert "receiver" in p
            assert "passer_team" in p
            assert "time_gc" in p

    def test_same_team_passes(self, game):
        result = detect_passes(game, 1, 690.0, 660.0)
        for p in result["passes"]:
            assert p["passer_team"] == p["receiver_team"]


class TestDefensiveMatchups:
    def test_returns_matchups(self, game):
        result = get_defensive_matchups(game, 1, 660.0)
        assert "matchups" in result
        assert len(result["matchups"]) == 5  # 5 offensive players

    def test_matchup_structure(self, game):
        result = get_defensive_matchups(game, 1, 660.0)
        for m in result["matchups"]:
            assert "offensive_player" in m
            assert "defensive_player" in m
            assert "coverage" in m
            assert m["coverage"] in ("tight", "moderate", "loose/help", "unguarded")

    def test_with_window(self, game):
        result = get_defensive_matchups(game, 1, 660.0, window_seconds=2)
        assert "matchups" in result
        assert len(result["matchups"]) == 5


class TestPlayerTrajectory:
    def test_known_player(self, game):
        result = get_player_trajectory(game, "Kyle Lowry", 1, 690.0, 660.0)
        assert "error" not in result
        assert result["player"] == "Kyle Lowry"
        assert len(result["samples"]) > 0

    def test_has_speed_and_distance(self, game):
        result = get_player_trajectory(game, "DeRozan", 1, 690.0, 660.0)
        assert result["total_distance_ft"] > 0
        assert result["avg_speed_ft_s"] >= 0

    def test_unknown_player(self, game):
        result = get_player_trajectory(game, "LeBron James", 1, 690.0, 660.0)
        assert "error" in result


class TestBallTrajectory:
    def test_returns_samples(self, game):
        result = get_ball_trajectory(game, 1, 690.0, 660.0)
        assert "error" not in result
        assert len(result["samples"]) > 0

    def test_samples_have_height(self, game):
        result = get_ball_trajectory(game, 1, 690.0, 660.0)
        for s in result["samples"]:
            assert "z" in s
            assert "state" in s
