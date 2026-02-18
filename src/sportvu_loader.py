"""Load, deduplicate, and index SportVU tracking data."""

import json
import bisect
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Player:
    player_id: int
    first_name: str
    last_name: str
    jersey: str
    team_id: int
    team_abbr: str
    position: str

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"


@dataclass
class Moment:
    period: int
    epoch_ms: int
    game_clock: float
    shot_clock: float  # may be None
    players: list  # list of [team_id, player_id, x, y, z]


@dataclass
class GameData:
    game_id: str
    game_date: str
    home_team: dict  # {name, teamid, abbreviation}
    visitor_team: dict
    roster: dict  # player_id -> Player
    roster_by_name: dict  # lowercase full name -> Player
    roster_by_jersey: dict  # (team_id, jersey) -> Player
    timelines: dict  # period -> list of Moment, sorted by game_clock descending
    timeline_clocks: dict  # period -> list of game_clock values (parallel to timelines)


def load_game(json_path: str) -> GameData:
    """Load a SportVU game JSON file into an indexed GameData structure."""
    path = Path(json_path)
    with open(path) as f:
        raw = json.load(f)

    game_id = raw["gameid"]
    game_date = raw["gamedate"]
    events = raw["events"]

    # Build roster from first event (all events have same roster)
    home_info = events[0]["home"]
    visitor_info = events[0]["visitor"]

    home_team = {
        "name": home_info["name"],
        "teamid": home_info["teamid"],
        "abbreviation": home_info["abbreviation"],
    }
    visitor_team = {
        "name": visitor_info["name"],
        "teamid": visitor_info["teamid"],
        "abbreviation": visitor_info["abbreviation"],
    }

    roster = {}
    roster_by_name = {}
    roster_by_jersey = {}

    for team_info in [home_info, visitor_info]:
        for p in team_info["players"]:
            player = Player(
                player_id=p["playerid"],
                first_name=p["firstname"],
                last_name=p["lastname"],
                jersey=p["jersey"],
                team_id=team_info["teamid"],
                team_abbr=team_info["abbreviation"],
                position=p["position"],
            )
            roster[player.player_id] = player
            roster_by_name[player.full_name.lower()] = player
            roster_by_jersey[(team_info["teamid"], p["jersey"])] = player

    # Deduplicate moments across events by (period, epoch_ms)
    seen = set()
    all_moments = []

    for event in events:
        for m in event["moments"]:
            period = m[0]
            epoch_ms = m[1]
            key = (period, epoch_ms)
            if key in seen:
                continue
            seen.add(key)
            moment = Moment(
                period=period,
                epoch_ms=epoch_ms,
                game_clock=m[2],
                shot_clock=m[3] if m[3] is not None else 0.0,
                players=m[5],
            )
            all_moments.append(moment)

    # Build sorted timelines per period (game_clock descending)
    periods = sorted(set(m.period for m in all_moments))
    timelines = {}
    timeline_clocks = {}

    for period in periods:
        period_moments = [m for m in all_moments if m.period == period]
        # Sort by game_clock descending (clock counts down)
        period_moments.sort(key=lambda m: -m.game_clock)
        # Remove exact game_clock duplicates within period (keep first = highest epoch_ms)
        deduped = []
        last_gc = None
        for m in period_moments:
            if m.game_clock != last_gc:
                deduped.append(m)
                last_gc = m.game_clock
        timelines[period] = deduped
        # Clocks in descending order for binary search
        timeline_clocks[period] = [m.game_clock for m in deduped]

    return GameData(
        game_id=game_id,
        game_date=game_date,
        home_team=home_team,
        visitor_team=visitor_team,
        roster=roster,
        roster_by_name=roster_by_name,
        roster_by_jersey=roster_by_jersey,
        timelines=timelines,
        timeline_clocks=timeline_clocks,
    )


def find_moment(game: GameData, period: int, target_gc: float) -> Moment | None:
    """Find the moment closest to target_gc in the given period using binary search.

    Clocks are sorted descending, so we search accordingly.
    """
    if period not in game.timeline_clocks:
        return None
    clocks = game.timeline_clocks[period]
    if not clocks:
        return None

    # Clocks are descending. We want the closest value.
    # bisect on negated values to use ascending bisect
    target_neg = -target_gc
    neg_clocks = [-c for c in clocks]  # ascending
    idx = bisect.bisect_left(neg_clocks, target_neg)

    # Check idx and idx-1 for closest
    best_idx = idx
    if idx >= len(clocks):
        best_idx = len(clocks) - 1
    elif idx > 0:
        if abs(clocks[idx] - target_gc) > abs(clocks[idx - 1] - target_gc):
            best_idx = idx - 1

    return game.timelines[period][best_idx]


def get_moments(game: GameData, period: int, gc_start: float, gc_end: float) -> list[Moment]:
    """Get all moments in [gc_end, gc_start] (clock counts down, so start > end).

    Returns moments sorted by game_clock descending.
    """
    if period not in game.timeline_clocks:
        return []

    clocks = game.timeline_clocks[period]
    moments = game.timelines[period]

    # clocks are descending: clocks[0] is highest
    # We want gc_end <= clock <= gc_start
    # In descending list, gc_start comes first, gc_end comes last
    # Find first index where clock <= gc_start
    start_idx = 0
    for i, c in enumerate(clocks):
        if c <= gc_start:
            start_idx = i
            break

    # Find last index where clock >= gc_end
    end_idx = len(clocks) - 1
    for i in range(len(clocks) - 1, -1, -1):
        if clocks[i] >= gc_end:
            end_idx = i
            break

    if start_idx > end_idx:
        return []

    return moments[start_idx : end_idx + 1]


def resolve_player(game: GameData, player_id: int) -> Player | None:
    """Look up player info by ID."""
    return game.roster.get(player_id)


def resolve_player_by_name(game: GameData, name: str) -> Player | None:
    """Look up player by name (case-insensitive, partial match)."""
    name_lower = name.lower()
    # Exact match
    if name_lower in game.roster_by_name:
        return game.roster_by_name[name_lower]
    # Partial match (last name)
    for full_name, player in game.roster_by_name.items():
        if name_lower in full_name:
            return player
    return None


def get_team_players(game: GameData, team_id: int) -> list[Player]:
    """Get all players on a team."""
    return [p for p in game.roster.values() if p.team_id == team_id]


def team_id_for_abbr(game: GameData, abbr: str) -> int | None:
    """Get team ID from abbreviation."""
    abbr = abbr.upper()
    if game.home_team["abbreviation"] == abbr:
        return game.home_team["teamid"]
    if game.visitor_team["abbreviation"] == abbr:
        return game.visitor_team["teamid"]
    return None


def print_game_info(game: GameData):
    """Print basic game information."""
    print(f"Game: {game.game_id}")
    print(f"Date: {game.game_date}")
    print(f"Home: {game.home_team['name']} ({game.home_team['abbreviation']})")
    print(f"Visitor: {game.visitor_team['name']} ({game.visitor_team['abbreviation']})")
    print(f"Periods: {sorted(game.timelines.keys())}")
    for p in sorted(game.timelines.keys()):
        tl = game.timelines[p]
        print(f"  Period {p}: {len(tl)} moments, clock {tl[0].game_clock:.2f} → {tl[-1].game_clock:.2f}")
    print(f"Roster: {len(game.roster)} players")
    for team_info in [game.visitor_team, game.home_team]:
        tid = team_info["teamid"]
        players = get_team_players(game, tid)
        print(f"  {team_info['abbreviation']}:")
        for p in sorted(players, key=lambda x: x.jersey):
            print(f"    #{p.jersey} {p.full_name} ({p.position})")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/0021500492.json"
    print(f"Loading {path}...")
    game = load_game(path)
    print_game_info(game)
