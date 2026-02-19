"""Core analytics tools for SportVU tracking data.

Each function returns structured dicts suitable for the Claude agent.
"""

import math
from src.config import (
    BALL_CONTROL_ENTRY_RADIUS, BALL_CONTROL_EXIT_RADIUS,
    MIN_HANDLER_FRAMES, DRIBBLE_HEIGHT_MAX, PASS_SPEED_THRESHOLD,
    LOOSE_BALL_RADIUS, REGAIN_WINDOW, MATCHUP_TIGHT, MATCHUP_MODERATE,
    MATCHUP_LOOSE, FPS,
)
from src.sportvu_loader import GameData, Moment, find_moment, get_moments, resolve_player
from src.court import describe_position, normalize_coords, detect_attacking_basket, distance_to_basket
from src.pbp import get_pbp_events


def _dist2d(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _fmt_clock(gc: float) -> str:
    mins = int(gc) // 60
    secs = gc - mins * 60
    return f"{mins}:{secs:04.1f}"


def _get_ball_and_players(moment: Moment):
    """Split moment players into ball entry and player entries."""
    ball = None
    players = []
    for entry in moment.players:
        if entry[0] == -1:
            ball = entry
        else:
            players.append(entry)
    return ball, players


def _resolve_player_info(game: GameData, team_id: int, player_id: int) -> dict:
    """Resolve player info into a clean dict."""
    p = resolve_player(game, player_id)
    if p:
        return {
            "player_id": player_id,
            "name": p.full_name,
            "jersey": p.jersey,
            "team": p.team_abbr,
            "position": p.position,
        }
    return {
        "player_id": player_id,
        "name": f"Unknown ({player_id})",
        "jersey": "?",
        "team": str(team_id),
        "position": "?",
    }


def get_player_positions(game: GameData, period: int, game_clock: float,
                         attacking_basket: dict | None = None,
                         offense_team_id: int | None = None) -> dict:
    """Get positions of all players and ball at a specific game clock.

    All positions are described from the offensive team's perspective.
    If offense_team_id is provided, that team's basket direction is used
    for all 10 players. Otherwise falls back to ball proximity.

    Returns snapshot with landmarks, distances, and raw coordinates.
    """
    moment = find_moment(game, period, game_clock)
    if not moment:
        return {"error": f"No tracking data for period {period}, clock {game_clock}"}

    if attacking_basket is None:
        attacking_basket = detect_attacking_basket(game, period)

    ball, players = _get_ball_and_players(moment)

    # Determine offensive perspective — one direction for all players
    if offense_team_id is not None:
        offense_attacking_right = attacking_basket.get(offense_team_id, False)
    elif ball:
        # Fallback: closest player to ball
        bx, by = ball[2], ball[3]
        min_dist = float("inf")
        offense_attacking_right = False
        for entry in players:
            d = _dist2d(bx, by, entry[2], entry[3])
            if d < min_dist:
                min_dist = d
                offense_attacking_right = attacking_basket.get(entry[0], False)
    else:
        offense_attacking_right = False

    result = {
        "period": period,
        "game_clock": round(moment.game_clock, 2),
        "shot_clock": round(moment.shot_clock, 1) if moment.shot_clock else None,
        "ball": None,
        "players": [],
    }

    if ball:
        result["ball"] = {
            "x": round(ball[2], 1),
            "y": round(ball[3], 1),
            "z": round(ball[4], 1),
        }

    for entry in players:
        team_id, player_id, x, y, z = entry[0], entry[1], entry[2], entry[3], entry[4]
        info = _resolve_player_info(game, team_id, player_id)

        # Normalize ALL players from offensive team's perspective
        nx, ny = normalize_coords(x, y, offense_attacking_right)
        pos = describe_position(nx, ny)

        info.update({
            "x": round(x, 1),
            "y": round(y, 1),
            "position": pos["description"],
            "landmark": pos["landmark"],
            "distance_to_basket": pos["distance_to_basket"],
        })
        result["players"].append(info)

    return result


def get_ball_handler(game: GameData, period: int, gc_start: float, gc_end: float) -> dict:
    """Detect ball handler over a time range using proximity with hysteresis.

    Returns timeline of handlers with gaps labeled.
    """
    moments = get_moments(game, period, gc_start, gc_end)
    if not moments:
        return {"error": f"No tracking data for period {period}, {gc_start}-{gc_end}"}

    # Track handler per frame
    frame_handlers = []  # (game_clock, player_id, team_id, distance)
    current_handler_id = None

    for moment in moments:
        ball, players = _get_ball_and_players(moment)
        if not ball:
            frame_handlers.append((moment.game_clock, None, None, None))
            continue

        bx, by, bz = ball[2], ball[3], ball[4]

        # Ball too high = airborne
        if bz > DRIBBLE_HEIGHT_MAX:
            frame_handlers.append((moment.game_clock, None, None, None))
            current_handler_id = None
            continue

        # Find closest player
        min_dist = float("inf")
        closest_pid = None
        closest_tid = None
        for entry in players:
            tid, pid, px, py, pz = entry[0], entry[1], entry[2], entry[3], entry[4]
            d = _dist2d(bx, by, px, py)
            if d < min_dist:
                min_dist = d
                closest_pid = pid
                closest_tid = tid

        # Hysteresis logic
        if current_handler_id is not None:
            # Check if current handler still has it
            current_dist = None
            for entry in players:
                if entry[1] == current_handler_id:
                    current_dist = _dist2d(bx, by, entry[2], entry[3])
                    break

            if current_dist is not None and current_dist <= BALL_CONTROL_EXIT_RADIUS:
                # Current handler retains
                frame_handlers.append((moment.game_clock, current_handler_id, closest_tid, current_dist))
                # Update team for current handler
                for entry in players:
                    if entry[1] == current_handler_id:
                        frame_handlers[-1] = (moment.game_clock, current_handler_id, entry[0], current_dist)
                        break
                continue
            else:
                current_handler_id = None

        # Try to acquire new handler
        if min_dist <= BALL_CONTROL_ENTRY_RADIUS:
            current_handler_id = closest_pid
            frame_handlers.append((moment.game_clock, closest_pid, closest_tid, min_dist))
        else:
            frame_handlers.append((moment.game_clock, None, None, min_dist))

    # Consolidate into handler segments with minimum duration
    segments = []
    if not frame_handlers:
        return {"handlers": [], "period": period, "gc_start": gc_start, "gc_end": gc_end}

    seg_start = 0
    for i in range(1, len(frame_handlers)):
        if frame_handlers[i][1] != frame_handlers[seg_start][1]:
            segments.append((seg_start, i - 1))
            seg_start = i
    segments.append((seg_start, len(frame_handlers) - 1))

    # Filter by minimum frames
    timeline = []
    for start_idx, end_idx in segments:
        pid = frame_handlers[start_idx][1]
        tid = frame_handlers[start_idx][2]
        gc_s = frame_handlers[start_idx][0]
        gc_e = frame_handlers[end_idx][0]
        n_frames = end_idx - start_idx + 1

        if pid is None:
            # Gap — ball in flight or loose
            timeline.append({
                "handler": None,
                "status": "ball in flight / loose",
                "start_gc": round(gc_s, 2),
                "end_gc": round(gc_e, 2),
                "duration": round(gc_s - gc_e, 2),
            })
        elif n_frames >= MIN_HANDLER_FRAMES:
            info = _resolve_player_info(game, tid, pid)
            timeline.append({
                "handler": info["name"],
                "team": info["team"],
                "jersey": info["jersey"],
                "start_gc": round(gc_s, 2),
                "end_gc": round(gc_e, 2),
                "duration": round(gc_s - gc_e, 2),
            })
        else:
            # Too short — treat as transition
            timeline.append({
                "handler": None,
                "status": "brief contact",
                "start_gc": round(gc_s, 2),
                "end_gc": round(gc_e, 2),
                "duration": round(gc_s - gc_e, 2),
            })

    # Merge adjacent gaps
    merged = []
    for seg in timeline:
        if (merged and merged[-1].get("handler") is None and seg.get("handler") is None):
            merged[-1]["end_gc"] = seg["end_gc"]
            merged[-1]["duration"] = round(merged[-1]["start_gc"] - seg["end_gc"], 2)
        else:
            merged.append(seg)

    return {
        "handlers": merged,
        "period": period,
        "gc_start": round(gc_start, 2),
        "gc_end": round(gc_end, 2),
    }


def detect_passes(game: GameData, period: int, gc_start: float, gc_end: float,
                  attacking_basket: dict | None = None) -> dict:
    """Detect passes in a time range.

    A pass = handler A → gap → handler B (same team).
    """
    handler_data = get_ball_handler(game, period, gc_start, gc_end)
    handlers = handler_data.get("handlers", [])

    if attacking_basket is None:
        attacking_basket = detect_attacking_basket(game, period)

    passes = []
    turnovers = []

    for i in range(len(handlers) - 2):
        seg_a = handlers[i]
        seg_gap = handlers[i + 1]
        seg_b = handlers[i + 2]

        # Need: handler A, gap, handler B
        if seg_a.get("handler") is None or seg_gap.get("handler") is not None or seg_b.get("handler") is None:
            continue

        passer = seg_a["handler"]
        receiver = seg_b["handler"]
        passer_team = seg_a.get("team", "")
        receiver_team = seg_b.get("team", "")

        # Same player regaining ball quickly = dribble
        if passer == receiver and seg_gap["duration"] < REGAIN_WINDOW:
            continue

        gap_duration = seg_gap["duration"]

        # Get passer and receiver positions
        passer_moment = find_moment(game, period, seg_a["end_gc"])
        receiver_moment = find_moment(game, period, seg_b["start_gc"])

        passer_pos = None
        receiver_pos = None
        pass_distance = None

        if passer_moment and receiver_moment:
            # Find passer position at pass time
            for entry in passer_moment.players:
                p = resolve_player(game, entry[1])
                if p and p.full_name == passer:
                    attacking_right = attacking_basket.get(entry[0], False)
                    nx, ny = normalize_coords(entry[2], entry[3], attacking_right)
                    passer_pos = describe_position(nx, ny)["description"]
                    passer_xy = (entry[2], entry[3])
                    break

            # Find receiver position at catch time
            for entry in receiver_moment.players:
                p = resolve_player(game, entry[1])
                if p and p.full_name == receiver:
                    attacking_right = attacking_basket.get(entry[0], False)
                    nx, ny = normalize_coords(entry[2], entry[3], attacking_right)
                    receiver_pos = describe_position(nx, ny)["description"]
                    receiver_xy = (entry[2], entry[3])
                    break

            if passer_pos and receiver_pos:
                pass_distance = round(_dist2d(*passer_xy, *receiver_xy), 1)

        pass_info = {
            "passer": passer,
            "receiver": receiver,
            "passer_team": passer_team,
            "receiver_team": receiver_team,
            "time_gc": round(seg_a["end_gc"], 2),
            "flight_time": round(gap_duration, 2),
            "passer_position": passer_pos,
            "receiver_position": receiver_pos,
            "distance_ft": pass_distance,
        }

        if passer_team == receiver_team:
            passes.append(pass_info)
        else:
            pass_info["type"] = "turnover"
            turnovers.append(pass_info)

    return {
        "passes": passes,
        "turnovers": turnovers,
        "period": period,
        "gc_start": round(gc_start, 2),
        "gc_end": round(gc_end, 2),
    }


def get_defensive_matchups(game: GameData, period: int, game_clock: float,
                           window_seconds: float = 0,
                           attacking_basket: dict | None = None) -> dict:
    """Get defensive matchups at a specific time.

    For each offensive player, finds the closest defender.
    If window > 0, uses majority vote across frames.
    """
    if attacking_basket is None:
        attacking_basket = detect_attacking_basket(game, period)

    # Determine offensive team from ball handler
    moment = find_moment(game, period, game_clock)
    if not moment:
        return {"error": f"No tracking data for period {period}, clock {game_clock}"}

    ball, players = _get_ball_and_players(moment)
    if not ball:
        return {"error": "No ball data in this moment"}

    bx, by = ball[2], ball[3]

    # Find ball handler's team = offensive team
    min_dist = float("inf")
    offensive_team_id = None
    for entry in players:
        d = _dist2d(bx, by, entry[2], entry[3])
        if d < min_dist:
            min_dist = d
            offensive_team_id = entry[0]

    if offensive_team_id is None:
        return {"error": "Could not determine offensive team"}

    if window_seconds > 0:
        moments_range = get_moments(game, period, game_clock + window_seconds / 2,
                                     game_clock - window_seconds / 2)
    else:
        moments_range = [moment]

    # Count matchups across frames
    matchup_counts = {}  # offensive_pid -> {defensive_pid: count}

    for m in moments_range:
        _, m_players = _get_ball_and_players(m)
        offense = [e for e in m_players if e[0] == offensive_team_id]
        defense = [e for e in m_players if e[0] != offensive_team_id and e[0] != -1]

        for off_entry in offense:
            off_pid = off_entry[1]
            if off_pid not in matchup_counts:
                matchup_counts[off_pid] = {}

            min_d = float("inf")
            closest_def = None
            for def_entry in defense:
                d = _dist2d(off_entry[2], off_entry[3], def_entry[2], def_entry[3])
                if d < min_d:
                    min_d = d
                    closest_def = (def_entry[1], def_entry[0], min_d)

            if closest_def:
                def_pid = closest_def[0]
                matchup_counts[off_pid][def_pid] = matchup_counts[off_pid].get(def_pid, 0) + 1

    # Resolve matchups (majority vote)
    matchups = []
    for off_pid, def_counts in matchup_counts.items():
        best_def_pid = max(def_counts, key=def_counts.get)

        # Get distance from the reference moment
        off_entry = None
        def_entry = None
        for e in players:
            if e[1] == off_pid:
                off_entry = e
            if e[1] == best_def_pid:
                def_entry = e

        dist = None
        coverage = "unknown"
        if off_entry and def_entry:
            dist = round(_dist2d(off_entry[2], off_entry[3], def_entry[2], def_entry[3]), 1)
            if dist < MATCHUP_TIGHT:
                coverage = "tight"
            elif dist < MATCHUP_MODERATE:
                coverage = "moderate"
            elif dist < MATCHUP_LOOSE:
                coverage = "loose/help"
            else:
                coverage = "unguarded"

        off_info = _resolve_player_info(game, offensive_team_id, off_pid)
        def_team_id = def_entry[0] if def_entry else None
        def_info = _resolve_player_info(game, def_team_id, best_def_pid)

        matchups.append({
            "offensive_player": off_info["name"],
            "offensive_team": off_info["team"],
            "defensive_player": def_info["name"],
            "defensive_team": def_info["team"],
            "distance_ft": dist,
            "coverage": coverage,
        })

    return {
        "matchups": matchups,
        "offensive_team": game.home_team["abbreviation"] if offensive_team_id == game.home_team["teamid"] else game.visitor_team["abbreviation"],
        "period": period,
        "game_clock": round(game_clock, 2),
    }


def get_player_trajectory(game: GameData, player_name: str, period: int,
                          gc_start: float, gc_end: float,
                          attacking_basket: dict | None = None) -> dict:
    """Get a player's trajectory over a time range.

    Returns sampled positions at ~0.5s intervals with speed and zone transitions.
    """
    from src.sportvu_loader import resolve_player_by_name
    player = resolve_player_by_name(game, player_name)
    if not player:
        return {"error": f"Player '{player_name}' not found"}

    if attacking_basket is None:
        attacking_basket = detect_attacking_basket(game, period)

    moments = get_moments(game, period, gc_start, gc_end)
    if not moments:
        return {"error": f"No tracking data for period {period}, {gc_start}-{gc_end}"}

    attacking_right = attacking_basket.get(player.team_id, False)

    # Extract player positions
    raw_points = []
    for m in moments:
        for entry in m.players:
            if entry[1] == player.player_id:
                raw_points.append({
                    "gc": m.game_clock,
                    "x": entry[2],
                    "y": entry[3],
                })
                break

    if not raw_points:
        return {"error": f"Player {player_name} not on court in this range"}

    # Compute speeds between consecutive points
    for i in range(1, len(raw_points)):
        dt = raw_points[i - 1]["gc"] - raw_points[i]["gc"]  # positive since clock descends
        if dt > 0:
            dx = raw_points[i]["x"] - raw_points[i - 1]["x"]
            dy = raw_points[i]["y"] - raw_points[i - 1]["y"]
            speed = math.sqrt(dx * dx + dy * dy) / dt  # ft/s
            raw_points[i]["speed_ft_s"] = round(speed, 1)
        else:
            raw_points[i]["speed_ft_s"] = 0.0
    if raw_points:
        raw_points[0]["speed_ft_s"] = 0.0

    # Sample at ~0.5s intervals
    sample_interval = 0.5
    sampled = []
    last_gc = None
    total_distance = 0.0

    for i, pt in enumerate(raw_points):
        if i > 0:
            total_distance += _dist2d(pt["x"], pt["y"],
                                      raw_points[i - 1]["x"], raw_points[i - 1]["y"])

        if last_gc is None or (last_gc - pt["gc"]) >= sample_interval:
            nx, ny = normalize_coords(pt["x"], pt["y"], attacking_right)
            pos = describe_position(nx, ny)
            sampled.append({
                "game_clock": round(pt["gc"], 2),
                "x": round(pt["x"], 1),
                "y": round(pt["y"], 1),
                "position": pos["description"],
                "landmark": pos["landmark"],
                "speed_ft_s": pt["speed_ft_s"],
            })
            last_gc = pt["gc"]

    # Position transitions
    transitions = []
    for i in range(1, len(sampled)):
        if sampled[i]["position"] != sampled[i - 1]["position"]:
            transitions.append({
                "from_position": sampled[i - 1]["position"],
                "to_position": sampled[i]["position"],
                "game_clock": sampled[i]["game_clock"],
            })

    avg_speed = sum(pt["speed_ft_s"] for pt in raw_points) / len(raw_points) if raw_points else 0

    return {
        "player": player.full_name,
        "team": player.team_abbr,
        "period": period,
        "gc_start": round(gc_start, 2),
        "gc_end": round(gc_end, 2),
        "total_distance_ft": round(total_distance, 1),
        "avg_speed_ft_s": round(avg_speed, 1),
        "samples": sampled,
        "position_transitions": transitions,
    }


def get_ball_trajectory(game: GameData, period: int, gc_start: float, gc_end: float) -> dict:
    """Get ball trajectory over a time range, including height and state."""
    moments = get_moments(game, period, gc_start, gc_end)
    if not moments:
        return {"error": f"No tracking data for period {period}, {gc_start}-{gc_end}"}

    raw_points = []
    for m in moments:
        ball, players = _get_ball_and_players(m)
        if ball:
            raw_points.append({
                "gc": m.game_clock,
                "x": ball[2],
                "y": ball[3],
                "z": ball[4],
            })

    if not raw_points:
        return {"error": "No ball data in this range"}

    # Sample at ~0.5s intervals
    sampled = []
    last_gc = None

    for i, pt in enumerate(raw_points):
        # Compute speed
        speed = 0.0
        if i > 0:
            dt = raw_points[i - 1]["gc"] - pt["gc"]
            if dt > 0:
                d = _dist2d(pt["x"], pt["y"], raw_points[i - 1]["x"], raw_points[i - 1]["y"])
                speed = d / dt

        # Determine ball state
        if pt["z"] > 12:
            state = "shot arc"
        elif pt["z"] > DRIBBLE_HEIGHT_MAX:
            state = "in flight"
        elif speed > PASS_SPEED_THRESHOLD:
            state = "in flight"
        else:
            state = "held/dribbled"

        if last_gc is None or (last_gc - pt["gc"]) >= 0.5:
            sampled.append({
                "game_clock": round(pt["gc"], 2),
                "x": round(pt["x"], 1),
                "y": round(pt["y"], 1),
                "z": round(pt["z"], 1),
                "speed_ft_s": round(speed, 1),
                "state": state,
            })
            last_gc = pt["gc"]

    return {
        "period": period,
        "gc_start": round(gc_start, 2),
        "gc_end": round(gc_end, 2),
        "samples": sampled,
    }


def get_play_by_play(game: GameData, pbp_events: list[dict],
                     period: int, gc_start: float, gc_end: float) -> dict:
    """Get PBP events for a time window."""
    events = get_pbp_events(pbp_events, period, gc_start, gc_end)
    return {
        "events": [
            {
                "game_clock": e["game_clock"],
                "clock_str": e["clock_str"],
                "description": e["description"],
                "event_type": e["espn_type_text"],
            }
            for e in events
        ],
        "period": period,
        "gc_start": round(gc_start, 2),
        "gc_end": round(gc_end, 2),
    }


def get_possession_summary(game: GameData, pbp_events: list[dict],
                           period: int, gc_start: float, gc_end: float) -> dict:
    """Composite summary of a possession: handler, passes, matchups, PBP, key positions."""
    attacking_basket = detect_attacking_basket(game, period)

    handler = get_ball_handler(game, period, gc_start, gc_end)
    passes = detect_passes(game, period, gc_start, gc_end, attacking_basket)
    pbp = get_play_by_play(game, pbp_events, period, gc_start, gc_end)

    # Positions at start, middle, and end
    gc_mid = (gc_start + gc_end) / 2
    pos_start = get_player_positions(game, period, gc_start, attacking_basket)
    pos_mid = get_player_positions(game, period, gc_mid, attacking_basket)
    pos_end = get_player_positions(game, period, gc_end, attacking_basket)

    # Matchups at start
    matchups = get_defensive_matchups(game, period, gc_start, window_seconds=1,
                                       attacking_basket=attacking_basket)

    return {
        "period": period,
        "gc_start": round(gc_start, 2),
        "gc_end": round(gc_end, 2),
        "duration": round(gc_start - gc_end, 2),
        "ball_handler_timeline": handler.get("handlers", []),
        "passes": passes.get("passes", []),
        "turnovers": passes.get("turnovers", []),
        "play_by_play": pbp.get("events", []),
        "matchups_at_start": matchups.get("matchups", []),
        "positions_at_start": pos_start.get("players", []),
        "positions_at_middle": pos_mid.get("players", []),
        "positions_at_end": pos_end.get("players", []),
    }
