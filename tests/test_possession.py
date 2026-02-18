"""Tests for PBP parsing and possession identification."""

import pytest
from src.pbp import parse_clock, fetch_pbp
from src.sportvu_loader import load_game
from src.possession import identify_possessions

GAME_PATH = "data/0021500492.json"
GAME_ID = "0021500492"


class TestParseClock:
    def test_minutes_and_seconds(self):
        assert parse_clock("8:45") == 525.0

    def test_twelve_minutes(self):
        assert parse_clock("12:00") == 720.0

    def test_seconds_only(self):
        assert parse_clock("33.0") == 33.0

    def test_zero(self):
        assert parse_clock("0:00") == 0.0

    def test_empty(self):
        assert parse_clock("") == 0.0


class TestFetchPbp:
    def test_loads_cached_data(self):
        events = fetch_pbp(GAME_ID)
        assert len(events) > 400  # Should be ~431

    def test_events_have_required_fields(self):
        events = fetch_pbp(GAME_ID)
        for e in events[:10]:
            assert "event_type" in e
            assert "period" in e
            assert "game_clock" in e
            assert "description" in e

    def test_four_periods(self):
        events = fetch_pbp(GAME_ID)
        periods = set(e["period"] for e in events)
        assert periods == {1, 2, 3, 4}


class TestIdentifyPossessions:
    @pytest.fixture(scope="class")
    def possessions(self):
        game = load_game(GAME_PATH)
        events = fetch_pbp(GAME_ID)
        return identify_possessions(events, game)

    def test_reasonable_count(self, possessions):
        # A typical NBA game has ~180-220 possessions
        assert 100 < len(possessions) < 300

    def test_possession_has_required_fields(self, possessions):
        p = possessions[0]
        assert p.period > 0
        assert p.start_gc > p.end_gc  # clock counts down
        assert p.duration > 0
        assert p.team_abbr in ("TOR", "CHA")

    def test_all_four_periods(self, possessions):
        periods = set(p.period for p in possessions)
        assert periods == {1, 2, 3, 4}

    def test_teams_alternate(self, possessions):
        # Most consecutive possessions should alternate teams
        # (not all, because of offensive rebounds)
        alternations = 0
        for i in range(1, min(len(possessions), 50)):
            if possessions[i].team_abbr != possessions[i - 1].team_abbr:
                alternations += 1
        # At least 60% should alternate
        assert alternations > 25
