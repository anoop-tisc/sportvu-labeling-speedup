"""Court landmark definitions and coordinate normalization.

Position description uses named court landmarks (points and lines) rather than
hard-edged zones.  Coordinates are normalized to a canonical half-court where
the basket is at (5.25, 25) and the player faces the basket from mid-court.

Left/Right convention: from the offensive player's perspective facing the basket.
  - Left  = y < 25 (near sideline when basket at x=5.25)
  - Right = y > 25 (far sideline when basket at x=5.25)
"""

import math
from dataclasses import dataclass, field
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
    with the convention (y < 25 = left, y > 25 = right).
    """
    if attacking_right:
        x = COURT_LENGTH - x
    else:
        y = COURT_WIDTH - y
    return x, y


# ---------------------------------------------------------------------------
# Landmark definitions
# ---------------------------------------------------------------------------

LANDMARK_RADIUS = 1.5  # ft — proximity threshold for points and line tolerance


@dataclass
class LandmarkPoint:
    name: str
    x: float
    y: float


@dataclass
class LandmarkLine:
    name: str
    kind: str  # "segment", "arc", or "composite"
    # Segment: (x1,y1)→(x2,y2)
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    # Arc: center + radius + angle range (radians)
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    angle_start: float = 0.0
    angle_end: float = 0.0
    # Composite: list of sub-lines (populated after construction)
    parts: list = field(default_factory=list)


LANDMARK_POINTS: list[LandmarkPoint] = [
    LandmarkPoint("basket", 5.25, 25),
    LandmarkPoint("left block", 7, 20),
    LandmarkPoint("right block", 7, 30),
    LandmarkPoint("left elbow", 19, 17),
    LandmarkPoint("right elbow", 19, 33),
    LandmarkPoint("free throw line center", 19, 25),
    LandmarkPoint("top of the key", 25, 25),
    LandmarkPoint("left short corner", 8, 13),
    LandmarkPoint("right short corner", 8, 37),
    LandmarkPoint("left corner", 5, 3),
    LandmarkPoint("right corner", 5, 47),
    LandmarkPoint("left wing", 28, 8),
    LandmarkPoint("right wing", 28, 42),
    LandmarkPoint("left deep wing", 40, 8),
    LandmarkPoint("right deep wing", 40, 42),
    LandmarkPoint("top of the arc", 29, 25),
    LandmarkPoint("the logo", 47, 25),
]

# Three-point arc angle range: the arc portion (excluding the straight corner segments).
# Corner segments run at y=3 (left) and y=47 (right) from the baseline.
# Arc center is at (5.25, 25), radius 23.75.
# The arc meets the corner lines where y=3 and y=47:
#   angle_left  = math.asin((25 - 3) / 23.75)   ≈ 1.176 rad
#   angle_right = math.asin((47 - 25) / 23.75)   ≈ 1.176 rad
# Angles measured from positive-x axis (0 = toward half-court).
_THREE_ARC_ANGLE = math.acos((25 - 3) / THREE_POINT_ARC_RADIUS)  # half-angle from center

_three_point_line = LandmarkLine("three-point line", "composite")
_three_point_line.parts = [
    # Left corner segment: baseline up from y=0 side, x goes from 0 to where arc starts
    LandmarkLine("three-point line", "segment",
                 x1=0, y1=3, x2=5.25 + THREE_POINT_ARC_RADIUS * math.cos(math.pi / 2 + _THREE_ARC_ANGLE), y2=3),
    # Arc portion
    LandmarkLine("three-point line", "arc",
                 cx=5.25, cy=25, radius=THREE_POINT_ARC_RADIUS,
                 angle_start=-math.pi / 2 + _THREE_ARC_ANGLE,
                 angle_end=math.pi / 2 - _THREE_ARC_ANGLE),
    # Right corner segment
    LandmarkLine("three-point line", "segment",
                 x1=0, y1=47, x2=5.25 + THREE_POINT_ARC_RADIUS * math.cos(math.pi / 2 + _THREE_ARC_ANGLE), y2=47),
]

LANDMARK_LINES: list[LandmarkLine] = [
    LandmarkLine("baseline", "segment", x1=0, y1=0, x2=0, y2=50),
    LandmarkLine("left sideline", "segment", x1=0, y1=0, x2=47, y2=0),
    LandmarkLine("right sideline", "segment", x1=0, y1=50, x2=47, y2=50),
    LandmarkLine("left lane line", "segment", x1=0, y1=17, x2=19, y2=17),
    LandmarkLine("right lane line", "segment", x1=0, y1=33, x2=19, y2=33),
    LandmarkLine("free throw line", "segment", x1=19, y1=17, x2=19, y2=33),
    _three_point_line,
    LandmarkLine("half-court line", "segment", x1=47, y1=0, x2=47, y2=50),
    LandmarkLine("restricted area", "arc",
                 cx=5.25, cy=25, radius=RESTRICTED_AREA_RADIUS,
                 angle_start=-math.pi / 2, angle_end=math.pi / 2),
]


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _distance_to_segment(px: float, py: float,
                         x1: float, y1: float, x2: float, y2: float) -> float:
    """Shortest distance from point (px, py) to segment (x1,y1)→(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _distance_to_arc(px: float, py: float, cx: float, cy: float,
                     radius: float, angle_start: float, angle_end: float) -> float:
    """Shortest distance from point (px, py) to a circular arc.

    The arc spans from angle_start to angle_end (radians, measured from
    the positive-x axis going counter-clockwise around (cx, cy)).
    """
    # Angle from center to point
    angle = math.atan2(py - cy, px - cx)

    # Normalize angles into [0, 2pi)
    def _norm(a):
        return a % (2 * math.pi)

    a_s = _norm(angle_start)
    a_e = _norm(angle_end)
    a_p = _norm(angle)

    # Check if point's angle falls within arc range
    if a_s <= a_e:
        on_arc = a_s <= a_p <= a_e
    else:
        on_arc = a_p >= a_s or a_p <= a_e

    if on_arc:
        # Distance is simply |dist_to_center - radius|
        dist_to_center = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        return abs(dist_to_center - radius)
    else:
        # Distance to nearest arc endpoint
        ex1 = cx + radius * math.cos(angle_start)
        ey1 = cy + radius * math.sin(angle_start)
        ex2 = cx + radius * math.cos(angle_end)
        ey2 = cy + radius * math.sin(angle_end)
        d1 = math.sqrt((px - ex1) ** 2 + (py - ey1) ** 2)
        d2 = math.sqrt((px - ex2) ** 2 + (py - ey2) ** 2)
        return min(d1, d2)


def _distance_to_line(px: float, py: float, line: LandmarkLine) -> float:
    """Distance from a point to a landmark line (segment, arc, or composite)."""
    if line.kind == "segment":
        return _distance_to_segment(px, py, line.x1, line.y1, line.x2, line.y2)
    elif line.kind == "arc":
        return _distance_to_arc(px, py, line.cx, line.cy, line.radius,
                                line.angle_start, line.angle_end)
    elif line.kind == "composite":
        return min(_distance_to_line(px, py, part) for part in line.parts)
    return float("inf")


# ---------------------------------------------------------------------------
# Main position description
# ---------------------------------------------------------------------------

def describe_position(x: float, y: float) -> dict:
    """Describe a normalized (x, y) position using the nearest court landmark.

    Coordinates must be normalized so the basket is at (5.25, 25).

    Returns dict with:
      - description: human-readable (e.g. "at the left elbow")
      - landmark: landmark name
      - relation: "at" | "on" | "near"
      - distance: distance to the landmark (ft)
      - side: "left" | "right" | "center"
      - distance_to_basket: float (feet)
    """
    dist_basket = distance_to_basket(x, y)

    # Side
    if y < 22:
        side = "left"
    elif y > 28:
        side = "right"
    else:
        side = "center"

    # 1. Check landmark points (most specific)
    best_point = None
    best_point_dist = float("inf")
    for lp in LANDMARK_POINTS:
        d = math.sqrt((x - lp.x) ** 2 + (y - lp.y) ** 2)
        if d < best_point_dist:
            best_point_dist = d
            best_point = lp

    if best_point_dist <= LANDMARK_RADIUS:
        return {
            "description": f"at the {best_point.name}",
            "landmark": best_point.name,
            "relation": "at",
            "distance": round(best_point_dist, 1),
            "side": side,
            "distance_to_basket": round(dist_basket, 1),
        }

    # 2. Check landmark lines
    best_line = None
    best_line_dist = float("inf")
    for ll in LANDMARK_LINES:
        d = _distance_to_line(x, y, ll)
        if d < best_line_dist:
            best_line_dist = d
            best_line = ll

    if best_line_dist <= LANDMARK_RADIUS:
        return {
            "description": f"on the {best_line.name}",
            "landmark": best_line.name,
            "relation": "on",
            "distance": round(best_line_dist, 1),
            "side": side,
            "distance_to_basket": round(dist_basket, 1),
        }

    # 3. Fallback: nearest landmark (point or line, whichever closer)
    if best_point_dist <= best_line_dist:
        return {
            "description": f"near the {best_point.name}",
            "landmark": best_point.name,
            "relation": "near",
            "distance": round(best_point_dist, 1),
            "side": side,
            "distance_to_basket": round(dist_basket, 1),
        }
    else:
        return {
            "description": f"near the {best_line.name}",
            "landmark": best_line.name,
            "relation": "near",
            "distance": round(best_line_dist, 1),
            "side": side,
            "distance_to_basket": round(dist_basket, 1),
        }


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
    # Smoke test: exercise all three relation types
    test_points = [
        # "at" — on a landmark point
        (5.25, 25.0, "should be 'at the basket'"),
        (19.0, 17.0, "should be 'at the left elbow'"),
        (19.0, 33.0, "should be 'at the right elbow'"),
        (7.0, 20.0, "should be 'at the left block'"),
        (5.0, 3.0, "should be 'at the left corner'"),
        (47.0, 25.0, "should be 'at the the logo'"),
        # "on" — on a landmark line
        (0.0, 25.0, "should be 'on the baseline'"),
        (10.0, 17.0, "should be 'on the left lane line'"),
        (19.0, 25.0, "at FT line center (point takes priority)"),
        # "near" — fallback
        (15.0, 10.0, "should be 'near' something"),
        (35.0, 25.0, "should be 'near' something"),
        (12.0, 25.0, "should be 'near' something"),
    ]
    for x, y, desc in test_points:
        result = describe_position(x, y)
        print(f"  ({x:5.1f}, {y:5.1f}) → {result['description']:35s} "
              f"[{result['relation']:4s}] dist={result['distance']:4.1f}ft  "
              f"basket={result['distance_to_basket']:5.1f}ft  | {desc}")
