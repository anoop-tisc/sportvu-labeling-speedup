"""Court zone definitions and coordinate normalization.

All zone classification uses coordinates normalized to a canonical half-court
where the basket is at (5.25, 25) and the player faces the basket from mid-court.

Left/Right convention: from the offensive player's perspective facing the basket.
  - Left  = y < 25 (near sideline when basket at x=5.25)
  - Right = y > 25 (far sideline when basket at x=5.25)
"""

import math
from src.config import (
    COURT_LENGTH, COURT_WIDTH, LEFT_BASKET, RESTRICTED_AREA_RADIUS,
    PAINT_Y_MIN, PAINT_Y_MAX, PAINT_DEPTH,
    THREE_POINT_ARC_RADIUS, THREE_POINT_CORNER_DIST,
)

BASKET = LEFT_BASKET  # Normalized basket position: (5.25, 25.0)


def distance_to_basket(x: float, y: float) -> float:
    """Euclidean distance from (x, y) to the basket at (5.25, 25)."""
    return math.sqrt((x - BASKET[0]) ** 2 + (y - BASKET[1]) ** 2)


def is_beyond_three_point_line(x: float, y: float) -> bool:
    """Check if a point is beyond the three-point line (normalized coords)."""
    dist = distance_to_basket(x, y)
    # Corner three: straight line at 22 ft, below y=11 or above y=39
    if y < 11.0 or y > 39.0:
        return dist > THREE_POINT_CORNER_DIST
    return dist > THREE_POINT_ARC_RADIUS


def normalize_coords(x: float, y: float, attacking_right: bool) -> tuple[float, float]:
    """Normalize coordinates so the offensive basket is at (5.25, 25).

    If team is attacking the right basket (88.75, 25), mirror x.
    If team is attacking the left basket, mirror y so that left/right
    from the player's perspective (facing the basket) stays consistent
    with classify_zone's convention (y < 25 = left, y > 25 = right).
    """
    if attacking_right:
        x = COURT_LENGTH - x
    else:
        y = COURT_WIDTH - y
    return x, y


def classify_zone(x: float, y: float) -> dict:
    """Classify a normalized (x, y) position into a basketball zone.

    Coordinates must be normalized so the basket is at (5.25, 25).

    Returns dict with:
      - zone: human-readable zone name
      - side: 'left', 'right', or 'center'
      - distance_to_basket: float (feet)
      - beyond_arc: bool
    """
    dist = distance_to_basket(x, y)
    beyond_arc = is_beyond_three_point_line(x, y)

    # Determine left/right/center
    if y < 22:
        side = "left"
    elif y > 28:
        side = "right"
    else:
        side = "center"

    # Priority 1: The logo (center circle, radius 6ft around midcourt)
    dist_to_center = math.sqrt((x - 47) ** 2 + (y - 25) ** 2)
    if dist_to_center <= 6:
        return {"zone": "the logo", "side": "center", "distance_to_basket": round(dist, 1), "beyond_arc": True}

    # Priority 2: Restricted area
    if dist <= RESTRICTED_AREA_RADIUS:
        return {"zone": "restricted area", "side": side, "distance_to_basket": round(dist, 1), "beyond_arc": False}

    # Priority 3: Elbows (at the corners of the free-throw line, check before paint)
    if 17 <= x <= 22:
        if 14 <= y <= 19:
            return {"zone": "left elbow", "side": "left", "distance_to_basket": round(dist, 1), "beyond_arc": False}
        if 31 <= y <= 36:
            return {"zone": "right elbow", "side": "right", "distance_to_basket": round(dist, 1), "beyond_arc": False}

    # Priority 4: Paint — split into low post (near basket) and high post (near FT line)
    if x <= PAINT_DEPTH and PAINT_Y_MIN <= y <= PAINT_Y_MAX:
        if y < 25:
            paint_side = "left"
            zone = "left low post" if x <= 10 else "left high post"
        elif y > 25:
            paint_side = "right"
            zone = "right low post" if x <= 10 else "right high post"
        else:
            paint_side = "center"
            zone = "left high post" if x <= 10 else "right high post"
        return {"zone": zone, "side": paint_side, "distance_to_basket": round(dist, 1), "beyond_arc": False}

    # Priority 5: Short corner (baseline area outside paint, inside 3pt line)
    if x <= 14 and not beyond_arc and not (PAINT_Y_MIN <= y <= PAINT_Y_MAX):
        if y < 25:
            return {"zone": "left short corner", "side": "left", "distance_to_basket": round(dist, 1), "beyond_arc": False}
        else:
            return {"zone": "right short corner", "side": "right", "distance_to_basket": round(dist, 1), "beyond_arc": False}

    # Priority 6: Three-point zones
    if beyond_arc:
        # Corner threes
        if y < 11 and x < 14:
            return {"zone": "left corner", "side": "left", "distance_to_basket": round(dist, 1), "beyond_arc": True}
        if y > 39 and x < 14:
            return {"zone": "right corner", "side": "right", "distance_to_basket": round(dist, 1), "beyond_arc": True}

        # Wing — split into near 3pt line vs near half court at x=35
        if y < 22:
            zone = "left wing" if x <= 35 else "left wing (deep)"
            return {"zone": zone, "side": "left", "distance_to_basket": round(dist, 1), "beyond_arc": True}
        if y > 28:
            zone = "right wing" if x <= 35 else "right wing (deep)"
            return {"zone": zone, "side": "right", "distance_to_basket": round(dist, 1), "beyond_arc": True}

        # Top of the arc — split into near 3pt line vs near half court at x=35
        zone = "top of the arc" if x <= 35 else "in front of the logo"
        return {"zone": zone, "side": "center", "distance_to_basket": round(dist, 1), "beyond_arc": True}

    # Priority 7: Mid-range
    if y < 22:
        zone = "left mid-range"
        mr_side = "left"
    elif y > 28:
        zone = "right mid-range"
        mr_side = "right"
    else:
        zone = "top of the key"
        mr_side = "center"

    return {"zone": zone, "side": mr_side, "distance_to_basket": round(dist, 1), "beyond_arc": False}


def detect_attacking_basket(game, period: int) -> dict[int, bool]:
    """Determine which basket each team attacks in a given period.

    Returns {team_id: attacking_right} where attacking_right means
    the team shoots at basket (88.75, 25).

    Logic: In period 1, look at early moments and see which side of the court
    the ball tends to be when each team has it closest. Teams swap sides
    at halftime (period 3).
    """
    from src.sportvu_loader import get_moments

    # Sample first 2 minutes of the period
    moments = get_moments(game, period, 720.0, 600.0)
    if not moments:
        # Try wider range
        moments = get_moments(game, period, 720.0, 500.0)
    if not moments:
        # Default assumption: home attacks right in periods 1-2
        home_id = game.home_team["teamid"]
        visitor_id = game.visitor_team["teamid"]
        if period <= 2:
            return {home_id: True, visitor_id: False}
        else:
            return {home_id: False, visitor_id: True}

    # For each moment, find who is closest to ball, determine their team,
    # and note which half of the court the ball is in
    team_x_sums = {}  # team_id -> (sum_of_ball_x, count)

    for moment in moments:
        ball = None
        players = []
        for entry in moment.players:
            if entry[0] == -1:
                ball = entry
            else:
                players.append(entry)

        if ball is None:
            continue

        ball_x, ball_y = ball[2], ball[3]

        # Find closest player to ball
        min_dist = float("inf")
        closest_team = None
        for p in players:
            dx = p[2] - ball_x
            dy = p[3] - ball_y
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d
                closest_team = p[0]

        if closest_team is not None and min_dist < 5.0:
            if closest_team not in team_x_sums:
                team_x_sums[closest_team] = [0.0, 0]
            team_x_sums[closest_team][0] += ball_x
            team_x_sums[closest_team][1] += 1

    # Team with lower average x is attacking the left basket (5.25, 25)
    # i.e., attacking_right = False
    home_id = game.home_team["teamid"]
    visitor_id = game.visitor_team["teamid"]

    result = {}
    for tid in [home_id, visitor_id]:
        if tid in team_x_sums and team_x_sums[tid][1] > 0:
            avg_x = team_x_sums[tid][0] / team_x_sums[tid][1]
            # If avg ball x < 47 when this team has ball, they're attacking left basket
            result[tid] = avg_x > 47  # attacking right basket
        else:
            result[tid] = tid == home_id  # default

    # If both teams ended up on the same side, override with convention
    if len(result) == 2:
        vals = list(result.values())
        if vals[0] == vals[1]:
            result[home_id] = True
            result[visitor_id] = False

    # Adjust for period: teams swap at halftime
    base_period = 1 if period <= 2 else 3
    if period > 2 and base_period == 3:
        # Use period 1 detection, then flip
        # Actually, we've already sampled from the correct period
        pass

    return result


if __name__ == "__main__":
    # Quick smoke test with known coordinates
    test_points = [
        (5.25, 25.0, "basket center → restricted area"),
        (3.0, 25.0, "under basket → restricted area"),
        (7.0, 20.0, "paint near basket → left low post"),
        (7.0, 30.0, "paint near basket → right low post"),
        (15.0, 20.0, "paint near FT line → left high post"),
        (15.0, 30.0, "paint near FT line → right high post"),
        (19.0, 17.0, "left elbow area"),
        (19.0, 33.0, "right elbow area"),
        (5.0, 3.0, "baseline outside paint → left short corner"),
        (5.0, 47.0, "baseline outside paint → right short corner"),
        (5.0, 8.0, "left corner three"),
        (5.0, 42.0, "right corner three"),
        (30.0, 10.0, "left wing"),
        (30.0, 40.0, "right wing"),
        (40.0, 10.0, "left wing (deep)"),
        (40.0, 40.0, "right wing (deep)"),
        (30.0, 25.0, "top of the arc"),
        (15.0, 15.0, "left mid-range"),
        (15.0, 35.0, "right mid-range"),
        (25.0, 25.0, "top of the key"),
        (30.0, 25.0, "top of the arc"),
        (40.0, 25.0, "in front of the logo"),
        (47.0, 25.0, "the logo"),
    ]
    for x, y, desc in test_points:
        result = classify_zone(x, y)
        print(f"  ({x:5.1f}, {y:5.1f}) → {result['zone']:25s} [{result['side']:6s}] dist={result['distance_to_basket']:5.1f}ft  | {desc}")
