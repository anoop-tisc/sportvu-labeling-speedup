"""Identify possession boundaries from play-by-play data."""

import re
from dataclasses import dataclass, field


@dataclass
class Possession:
    team_id: str  # ESPN team ID (string)
    team_abbr: str
    period: int
    start_gc: float  # game clock at start (higher)
    end_gc: float    # game clock at end (lower)
    events: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.start_gc - self.end_gc

    def summary(self) -> str:
        dur = self.duration
        descs = [e["description"] for e in self.events if e.get("description")]
        event_str = "; ".join(descs[-3:]) if descs else "no events"
        if len(event_str) > 100:
            event_str = event_str[:97] + "..."
        return (
            f"P{self.period} {_fmt_clock(self.start_gc)}→{_fmt_clock(self.end_gc)} "
            f"({dur:.1f}s) {self.team_abbr}: {event_str}"
        )


def _fmt_clock(gc: float) -> str:
    """Format game clock seconds as M:SS."""
    mins = int(gc) // 60
    secs = int(gc) % 60
    return f"{mins}:{secs:02d}"


def identify_possessions(pbp_events: list[dict], game=None,
                         espn_home_id: str = None, espn_away_id: str = None,
                         home_abbr: str = None, away_abbr: str = None) -> list[Possession]:
    """Parse PBP events to identify possession boundaries.

    A possession ends on:
      - Made field goal (type 1)
      - Turnover (type 5)
      - Defensive rebound after miss (type 4 with espn_type_id 155)
      - End of period (type 13)
      - Last free throw made (type 3, "X of X" pattern)

    Args:
        pbp_events: Normalized PBP events from pbp.py
        game: Optional GameData for team info
        espn_home_id: ESPN team ID for home team
        espn_away_id: ESPN team ID for away team
        home_abbr: Home team abbreviation
        away_abbr: Away team abbreviation
    """
    # Resolve team IDs
    if game and not espn_home_id:
        # Auto-detect ESPN team IDs from PBP data
        team_ids = set()
        for e in pbp_events:
            tid = e.get("espn_team_id")
            if tid:
                team_ids.add(tid)
        team_ids = sorted(team_ids)
        if len(team_ids) == 2:
            espn_home_id = team_ids[0]
            espn_away_id = team_ids[1]
            # We need to figure out which is home/away from context
            # For now, use the game data
            home_abbr = home_abbr or game.home_team["abbreviation"]
            away_abbr = away_abbr or game.visitor_team["abbreviation"]
        else:
            espn_home_id = espn_home_id or "HOME"
            espn_away_id = espn_away_id or "AWAY"

    home_abbr = home_abbr or "HOME"
    away_abbr = away_abbr or "AWAY"

    def team_abbr(tid):
        if tid == espn_home_id:
            return home_abbr
        if tid == espn_away_id:
            return away_abbr
        return "UNK"

    def other_team(tid):
        if tid == espn_home_id:
            return espn_away_id
        return espn_home_id

    # Sort events by period, then game_clock descending
    sorted_events = sorted(
        pbp_events,
        key=lambda e: (e["period"], -e["game_clock"], e["event_num"])
    )

    possessions = []
    current_team_id = None
    current_start_gc = None
    current_events = []
    current_period = None

    def end_possession(end_gc):
        nonlocal current_team_id, current_start_gc, current_events
        if current_team_id and current_start_gc is not None:
            possessions.append(Possession(
                team_id=current_team_id,
                team_abbr=team_abbr(current_team_id),
                period=current_period,
                start_gc=current_start_gc,
                end_gc=end_gc,
                events=list(current_events),
            ))
        current_team_id = None
        current_start_gc = None
        current_events = []

    for event in sorted_events:
        etype = event["event_type"]
        period = event["period"]
        gc = event["game_clock"]
        evt_team = event.get("espn_team_id")

        # Period change
        if period != current_period:
            if current_period is not None:
                end_possession(0.0)
            current_period = period
            current_team_id = None
            current_start_gc = 720.0  # Start of quarter
            current_events = []

        # Skip non-possession events
        if etype in (8, 9, 7):  # sub, timeout, violation
            continue

        # Jump ball → new possession
        if etype == 10:
            end_possession(gc)
            # Determine who gets possession from text
            text = event.get("description", "")
            gains_match = re.search(r"\((.+?) gains possession\)", text)
            if gains_match and evt_team:
                current_team_id = evt_team
            elif evt_team:
                current_team_id = evt_team
            current_start_gc = gc
            current_events = [event]
            continue

        # Period end
        if etype == 13:
            end_possession(gc)
            continue

        # If we don't know who has ball, infer from first action
        if current_team_id is None and evt_team:
            current_team_id = evt_team
            if current_start_gc is None:
                current_start_gc = gc

        current_events.append(event)

        # Made shot → possession ends, other team gets ball
        if etype == 1:
            end_possession(gc)
            if evt_team:
                current_team_id = other_team(evt_team)
            current_start_gc = gc
            current_events = []
            continue

        # Turnover → possession ends, other team gets ball
        if etype == 5:
            end_possession(gc)
            if evt_team:
                current_team_id = other_team(evt_team)
            current_start_gc = gc
            current_events = []
            continue

        # Defensive rebound → possession changes
        if etype == 4:
            espn_tid = event.get("espn_type_id", 0)
            if espn_tid == 155:  # defensive rebound
                end_possession(gc)
                if evt_team:
                    current_team_id = evt_team
                current_start_gc = gc
                current_events = [event]
                continue
            # Offensive rebound — same team keeps possession, no break

        # Free throw: if last FT, possession may change
        if etype == 3:
            desc = event.get("description", "")
            ft_match = re.search(r"(\d+) of (\d+)", desc)
            if ft_match and ft_match.group(1) == ft_match.group(2):
                # Last free throw — possession ends
                end_possession(gc)
                if evt_team:
                    current_team_id = other_team(evt_team)
                current_start_gc = gc
                current_events = []
                continue

    # End any remaining possession
    if current_team_id:
        end_possession(0.0)

    # Filter out zero-duration and very short possessions
    possessions = [p for p in possessions if p.duration > 0.5]

    return possessions


def list_possessions(possessions: list[Possession], period: int | None = None) -> list[str]:
    """Get formatted list of possessions, optionally filtered by period."""
    result = []
    for i, p in enumerate(possessions):
        if period is not None and p.period != period:
            continue
        result.append(f"  [{i:3d}] {p.summary()}")
    return result


if __name__ == "__main__":
    import sys
    from src.sportvu_loader import load_game
    from src.pbp import fetch_pbp

    game_id = sys.argv[1] if len(sys.argv) > 1 else "0021500492"
    json_path = sys.argv[2] if len(sys.argv) > 2 else f"data/{game_id}.json"

    game = load_game(json_path)
    events = fetch_pbp(game_id)

    if not events:
        print("No PBP data available.")
        sys.exit(1)

    possessions = identify_possessions(events, game)
    print(f"Found {len(possessions)} possessions")
    for line in list_possessions(possessions, period=1):
        print(line)
