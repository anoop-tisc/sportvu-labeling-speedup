"""Fetch and cache play-by-play data from ESPN API."""

import json
import re
from pathlib import Path
from src.config import DATA_DIR


# ESPN game IDs for known SportVU games (NBA game ID -> ESPN event ID)
ESPN_GAME_MAP = {
    "0021500492": "400828379",  # CHA @ TOR, 2016-01-01
}


def parse_clock(clock_str: str) -> float:
    """Parse a clock string like '8:45' or '33.0' into total seconds."""
    if not clock_str:
        return 0.0
    clock_str = clock_str.strip()
    if ":" in clock_str:
        parts = clock_str.split(":")
        return int(parts[0]) * 60.0 + float(parts[1])
    return float(clock_str)


def fetch_pbp(game_id: str, cache_dir: str | None = None, espn_event_id: str | None = None) -> list[dict]:
    """Fetch play-by-play data for a game from ESPN, caching locally.

    Returns list of normalized PBP event dicts.
    """
    cache_dir = cache_dir or DATA_DIR
    cache_path = Path(cache_dir) / f"pbp_{game_id}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    # Check for raw ESPN cache
    espn_cache = Path(cache_dir) / f"espn_pbp_{game_id}.json"
    if espn_cache.exists():
        with open(espn_cache) as f:
            espn_plays = json.load(f)
        events = _normalize_espn_plays(espn_plays)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(events, f, indent=2)
        return events

    # Fetch from ESPN API
    event_id = espn_event_id or ESPN_GAME_MAP.get(game_id)
    if not event_id:
        print(f"No ESPN event ID known for game {game_id}.")
        print(f"Provide it via --espn-id or add to ESPN_GAME_MAP in pbp.py")
        return []

    try:
        import requests
        url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        espn_plays = data.get("plays", [])
    except Exception as e:
        print(f"Warning: Could not fetch PBP from ESPN: {e}")
        print(f"To manually provide PBP data, place ESPN plays JSON at: {espn_cache}")
        return []

    # Save raw ESPN data for reference
    with open(espn_cache, "w") as f:
        json.dump(espn_plays, f, indent=2)

    events = _normalize_espn_plays(espn_plays)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(events, f, indent=2)

    return events


def _normalize_espn_plays(espn_plays: list[dict]) -> list[dict]:
    """Convert ESPN play format to our normalized PBP event format."""
    events = []
    for i, play in enumerate(espn_plays):
        clock_str = play.get("clock", {}).get("displayValue", "0:00")
        game_clock = parse_clock(clock_str)
        period = play.get("period", {}).get("number", 0)

        # Map ESPN type to simplified event type
        espn_type_id = int(play.get("type", {}).get("id", 0))
        event_type = _map_espn_type(espn_type_id, play)

        team_id = play.get("team", {}).get("id")

        # Extract participant names from text
        text = play.get("text", "")

        event = {
            "event_num": i,
            "event_type": event_type,
            "espn_type_id": espn_type_id,
            "espn_type_text": play.get("type", {}).get("text", ""),
            "period": period,
            "game_clock": game_clock,
            "clock_str": clock_str,
            "description": text,
            "espn_team_id": team_id,
            "scoring_play": play.get("scoringPlay", False),
            "shooting_play": play.get("shootingPlay", False),
            "score_value": play.get("scoreValue", 0),
            "away_score": play.get("awayScore", 0),
            "home_score": play.get("homeScore", 0),
            "coordinate": play.get("coordinate"),
            "sequence_number": play.get("sequenceNumber"),
        }
        events.append(event)

    return events


# ESPN play type -> simplified event type mapping
# Simplified types: 1=made shot, 2=missed shot, 3=free throw, 4=rebound,
# 5=turnover, 6=foul, 7=violation, 8=substitution, 9=timeout,
# 10=jump ball, 12=period start, 13=period end

_MADE_SHOT_TYPES = {
    93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
    109, 110, 111, 113, 114, 115, 118, 119, 120, 122, 124, 125, 128, 129,
    130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
}
_SHOT_TYPES = _MADE_SHOT_TYPES | {91, 92}  # 91=no shot, 92=jump shot (used for misses too)
_FREE_THROW_TYPES = {97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108}
_REBOUND_TYPES = {155, 156}  # 155=defensive, 156=offensive
_TURNOVER_TYPES = {62, 63, 64, 67, 70, 71, 84, 87}
_FOUL_TYPES = {21, 23, 29, 35, 42, 43, 44, 45}
_TIMEOUT_TYPES = {16, 17, 581}
_SUB_TYPES = {584}
_PERIOD_END_TYPES = {412, 402}  # 412=end period, 402=end game


def _map_espn_type(espn_type_id: int, play: dict) -> int:
    """Map ESPN play type ID to simplified event type."""
    if espn_type_id == 11:
        return 10  # jump ball
    if espn_type_id in _PERIOD_END_TYPES:
        return 13  # period end

    if espn_type_id in _FREE_THROW_TYPES:
        return 3  # free throw
    if espn_type_id in _REBOUND_TYPES:
        return 4  # rebound
    if espn_type_id in _TURNOVER_TYPES:
        return 5  # turnover
    if espn_type_id in _FOUL_TYPES:
        return 6  # foul
    if espn_type_id in _TIMEOUT_TYPES:
        return 9  # timeout
    if espn_type_id in _SUB_TYPES:
        return 8  # substitution
    if espn_type_id == 8:  # delay of game
        return 7  # violation
    if espn_type_id == 12:  # kicked ball
        return 7  # violation

    # Shot types: determine made/missed from text and scoringPlay flag
    if espn_type_id in _SHOT_TYPES or play.get("shootingPlay"):
        if play.get("scoringPlay"):
            return 1  # made shot
        return 2  # missed shot

    # Default
    return 0


def get_pbp_events(events: list[dict], period: int, gc_start: float, gc_end: float) -> list[dict]:
    """Filter PBP events for a time window. gc_start > gc_end (clock counts down)."""
    return [
        e for e in events
        if e["period"] == period and gc_end <= e["game_clock"] <= gc_start
    ]


if __name__ == "__main__":
    import sys
    game_id = sys.argv[1] if len(sys.argv) > 1 else "0021500492"
    events = fetch_pbp(game_id)
    print(f"Loaded {len(events)} PBP events")
    if events:
        for e in events[:20]:
            type_names = {1: "MADE", 2: "MISS", 3: "FT", 4: "REB", 5: "TO",
                         6: "FOUL", 7: "VIOL", 8: "SUB", 9: "TO", 10: "JB", 13: "END"}
            tname = type_names.get(e["event_type"], f"?{e['event_type']}")
            print(f"  P{e['period']} {e['clock_str']:>5s} [{tname:4s}] {e['description'][:80]}")
