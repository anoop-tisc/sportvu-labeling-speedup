"""Render SportVU tracking data as MP4 videos.

Draws an overhead court view with colored player/ball dots, jersey numbers,
game clock, shot clock, score, and PBP commentary text.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from mpl_toolkits.mplot3d import Axes3D
from subprocess import Popen, PIPE
from pathlib import Path

from src.sportvu_loader import GameData, Moment, get_moments, resolve_player_by_name

# Output dimensions
WIDTH = 1280
HEIGHT = 720
DPI = 100
FIG_W = WIDTH / DPI
FIG_H = HEIGHT / DPI
FPS = 25


def draw_court(ax, color="white", lw=1.5, zorder=0):
    """Draw full NBA court lines on *ax*."""
    elements = [
        Rectangle((0, -50), width=94, height=50, color=color,
                  zorder=zorder, fill=False, lw=lw),
        # Left basket
        Circle((5.35, -25), radius=.75, lw=lw, fill=False,
               color=color, zorder=zorder),
        Rectangle((4, -28), 0, 6, lw=lw, color=color, zorder=zorder),
        # Left paint
        Rectangle((0, -33), 19, 16, lw=lw, fill=False,
                  color=color, zorder=zorder),
        Rectangle((0, -31), 19, 12, lw=lw, fill=False,
                  color=color, zorder=zorder),
        Circle((19, -25), radius=6, lw=lw, fill=False,
               color=color, zorder=zorder),
        # Left 3-pt
        Rectangle((0, -3), 14, 0, lw=lw, color=color, zorder=zorder),
        Rectangle((0, -47), 14, 0, lw=lw, color=color, zorder=zorder),
        Arc((5, -25), 47.5, 47.5, theta1=292, theta2=68, lw=lw,
            color=color, zorder=zorder),
        # Right basket
        Circle((88.65, -25), radius=.75, lw=lw, fill=False,
               color=color, zorder=zorder),
        Rectangle((90, -28), 0, 6, lw=lw, color=color, zorder=zorder),
        # Right paint
        Rectangle((75, -33), 19, 16, lw=lw, fill=False,
                  color=color, zorder=zorder),
        Rectangle((75, -31), 19, 12, lw=lw, fill=False,
                  color=color, zorder=zorder),
        Circle((75, -25), radius=6, lw=lw, fill=False,
               color=color, zorder=zorder),
        # Right 3-pt
        Rectangle((80, -3), 14, 0, lw=lw, color=color, zorder=zorder),
        Rectangle((80, -47), 14, 0, lw=lw, color=color, zorder=zorder),
        Arc((89, -25), 47.5, 47.5, theta1=112, theta2=248,
            lw=lw, color=color, zorder=zorder),
        # Half court
        Rectangle((47, -50), 0, 50, lw=lw, color=color, zorder=zorder),
        Circle((47, -25), radius=6, lw=lw, fill=False,
               color=color, zorder=zorder),
        Circle((47, -25), radius=2, lw=lw, fill=False,
               color=color, zorder=zorder),
    ]
    for el in elements:
        ax.add_patch(el)
    return ax


def _circle_pts(cx, cy, r, theta1=0, theta2=360, n=64):
    """Generate (x, y) arrays for a circle/arc at z=0."""
    t = np.linspace(np.radians(theta1), np.radians(theta2), n)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def draw_court_3d(ax, color="black", lw=1.0):
    """Draw full NBA court lines on a 3D axes at z=0, plus hoops at z=10."""
    z0 = lambda n: np.zeros(n)
    RIM_HEIGHT = 10.0
    RIM_RADIUS = 0.75
    BACKBOARD_WIDTH = 6.0

    # --- Court boundary ---
    for x0, y0, x1, y1 in [
        (0, -50, 94, -50), (94, -50, 94, 0), (94, 0, 0, 0), (0, 0, 0, -50),
    ]:
        ax.plot([x0, x1], [y0, y1], [0, 0], color=color, lw=lw)

    # --- Half court line ---
    ax.plot([47, 47], [-50, 0], [0, 0], color=color, lw=lw)

    # Half court circles
    for r in [6, 2]:
        cx, cy = _circle_pts(47, -25, r)
        ax.plot(cx, cy, z0(len(cx)), color=color, lw=lw)

    # --- Helper for mirrored elements (left at bx, right at 94-bx) ---
    def _draw_side(bx, basket_x, sign):
        # Paint outer
        px = bx if sign == 1 else 94 - 19
        for x0, y0, x1, y1 in [
            (px, -33, px + 19, -33), (px + 19, -33, px + 19, -17),
            (px + 19, -17, px, -17), (px, -17, px, -33),
        ]:
            ax.plot([x0, x1], [y0, y1], [0, 0], color=color, lw=lw)
        # Paint inner
        for x0, y0, x1, y1 in [
            (px, -31, px + 19, -31), (px + 19, -31, px + 19, -19),
            (px + 19, -19, px, -19), (px, -19, px, -31),
        ]:
            ax.plot([x0, x1], [y0, y1], [0, 0], color=color, lw=lw)
        # FT circle
        ft_x = 19 if sign == 1 else 75
        cx, cy = _circle_pts(ft_x, -25, 6)
        ax.plot(cx, cy, z0(len(cx)), color=color, lw=lw)

        # Basket circle on floor
        cx, cy = _circle_pts(basket_x, -25, RIM_RADIUS)
        ax.plot(cx, cy, z0(len(cx)), color=color, lw=lw * 0.5, alpha=0.3)

        # Backboard on floor
        bb_x = 4 if sign == 1 else 90
        ax.plot([bb_x, bb_x], [-28, -22], [0, 0], color=color, lw=lw)

        # 3-pt corner lines
        corner_x = 14 if sign == 1 else 80
        base_x = 0 if sign == 1 else 94
        ax.plot([base_x, corner_x], [-3, -3], [0, 0], color=color, lw=lw)
        ax.plot([base_x, corner_x], [-47, -47], [0, 0], color=color, lw=lw)

        # 3-pt arc
        arc_cx = 5 if sign == 1 else 89
        t1, t2 = (292, 428) if sign == 1 else (112, 248)
        cx, cy = _circle_pts(arc_cx, -25, 23.75, t1, t2, n=80)
        ax.plot(cx, cy, z0(len(cx)), color=color, lw=lw)

        # === 3D Hoop at rim height ===
        # Rim ring
        cx, cy = _circle_pts(basket_x, -25, RIM_RADIUS, n=32)
        ax.plot(cx, cy, np.full(len(cx), RIM_HEIGHT), color='#ff6600', lw=1.5)

        # Backboard (vertical rectangle)
        bb_hw = BACKBOARD_WIDTH / 2
        bb_h = 3.5  # backboard height in feet
        bb_bottom = RIM_HEIGHT - 0.5
        ax.plot([bb_x, bb_x], [-25 - bb_hw, -25 + bb_hw],
                [bb_bottom, bb_bottom], color='#aaa', lw=1.5)
        ax.plot([bb_x, bb_x], [-25 - bb_hw, -25 + bb_hw],
                [bb_bottom + bb_h, bb_bottom + bb_h], color='#aaa', lw=1.5)
        ax.plot([bb_x, bb_x], [-25 - bb_hw, -25 - bb_hw],
                [bb_bottom, bb_bottom + bb_h], color='#aaa', lw=1.5)
        ax.plot([bb_x, bb_x], [-25 + bb_hw, -25 + bb_hw],
                [bb_bottom, bb_bottom + bb_h], color='#aaa', lw=1.5)

        # Vertical post from rim to backboard (connector)
        ax.plot([basket_x, bb_x], [-25, -25],
                [RIM_HEIGHT, RIM_HEIGHT], color='#aaa', lw=1.0)

    _draw_side(0, 5.35, 1)      # left
    _draw_side(75, 88.65, -1)    # right

    return ax


def get_commentary(pbp_events, period, game_clock, depth=10, max_lines=4):
    """Return (commentary_script, score_str) for the given moment."""
    game_time = (period - 1) * 720 + (720 - game_clock)

    lines = []
    score_str = "0 - 0"

    for evt in pbp_events:
        evt_gt = (evt.get("period", 0) - 1) * 720 + (720 - evt.get("game_clock", 0))
        if not (game_time - depth <= evt_gt <= game_time + 2):
            continue
        desc = evt.get("description", "")
        if desc and len(lines) < max_lines:
            # Truncate long descriptions
            lines.append(desc[:80])
        away = evt.get("away_score", 0)
        home = evt.get("home_score", 0)
        if away or home:
            score_str = f"{away} - {home}"

    return "\n".join(lines) if lines else "", score_str


def _extract_frame_data(moment, game, team_colors, highlight_player_id=None):
    """Extract scatter plot data from a Moment."""
    home_id = game.home_team["teamid"]
    x_pos, y_pos, colors, sizes, edge_widths = [], [], [], [], []
    jerseys = []  # (x, y, jersey_str) for non-ball entries

    for entry in moment.players:
        tid, pid, x, y = entry[0], entry[1], entry[2], entry[3]
        z = entry[4] if len(entry) > 4 else 0.0

        x_pos.append(x)
        y_adj = -y  # Court drawn from y=-50 to y=0; negate to flip width axis
        y_pos.append(y_adj)
        colors.append(team_colors.get(tid, "gray"))

        if tid == -1:  # Ball
            sizes.append(max(150 - 2 * (z - 5) ** 2, 10))
        else:
            sizes.append(200)

        if highlight_player_id is not None and pid == highlight_player_id:
            edge_widths.append(4)
        else:
            edge_widths.append(0.5)

        # Jersey labels for players (not ball)
        if tid != -1:
            player = game.roster.get(pid)
            if player:
                jerseys.append((x, y_adj, player.jersey))

    return x_pos, y_pos, colors, sizes, edge_widths, jerseys


def _clock_strings(moment):
    """Format shot clock and game clock for display."""
    sc = moment.shot_clock
    if sc is None or (isinstance(sc, float) and np.isnan(sc)):
        sc = 24.0
    shot_clock_str = str(int(sc))

    mins, secs = divmod(moment.game_clock, 60)
    game_clock_str = f"{int(mins):02d}:{int(secs):02d}"
    quarter_str = f"Q{moment.period}"
    return shot_clock_str, game_clock_str, quarter_str


def render_frame(moment, game, pbp_events, team_colors=None,
                 commentary=True, highlight_player_id=None):
    """Render a single frame and return the matplotlib Figure.

    Useful for generating a static image of a moment.
    """
    home_id = game.home_team["teamid"]
    away_id = game.visitor_team["teamid"]
    home_abbr = game.home_team["abbreviation"]
    away_abbr = game.visitor_team["abbreviation"]
    if team_colors is None:
        team_colors = {-1: "#ff8c00", home_id: "#e74c3c", away_id: "#3498db"}

    x_pos, y_pos, colors, sizes, edges, jerseys = _extract_frame_data(
        moment, game, team_colors, highlight_player_id)
    shot_clock_str, game_clock_str, quarter_str = _clock_strings(moment)
    commentary_script, score_str = get_commentary(
        pbp_events, moment.period, moment.game_clock)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("white")

    # Court axes — upper portion
    ax = fig.add_axes([0.02, 0.25, 0.96, 0.65])
    ax.set_facecolor("#e8e8e8")
    draw_court(ax, color="black")
    ax.set_xlim(-5, 100)
    ax.set_ylim(-55, 5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Players & ball
    ax.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=0.9,
               linewidths=edges, edgecolors="black", zorder=5)

    # Jersey numbers
    for jx, jy, jersey in jerseys:
        ax.text(jx, jy, jersey, color="white", fontsize=6,
                ha="center", va="center", fontweight="bold", zorder=6)

    # Team colour indicators on court
    ax.scatter([30], [2.5], s=80, c=[team_colors[away_id]],
               edgecolors="black", linewidths=0.5, zorder=5)
    ax.scatter([67], [2.5], s=80, c=[team_colors[home_id]],
               edgecolors="black", linewidths=0.5, zorder=5)

    # Score & team names (top)
    fig.text(0.5, 0.94,
             f"{away_abbr}   {score_str}   {home_abbr}",
             ha="center", va="center", color="black",
             fontsize=16, fontweight="bold")

    # Clocks (below score)
    fig.text(0.5, 0.91,
             f"{quarter_str}   {game_clock_str}      Shot: {shot_clock_str}",
             ha="center", va="center", color="black", fontsize=13)

    # Commentary (below court)
    if commentary and commentary_script:
        fig.text(0.5, 0.20, commentary_script,
                 ha="center", va="top", color="black",
                 fontsize=10, linespacing=1.4,
                 fontstyle="italic")

    return fig


def _build_handler_lookup(game, period, gc_start, gc_end):
    """Run ball handler detection and return a function that maps game_clock → player_id or None."""
    from src.tracking_tools import get_ball_handler

    data = get_ball_handler(game, period, gc_start, gc_end)
    segments = data.get("handlers", [])

    # Resolve handler names to player_ids
    resolved = []  # (start_gc, end_gc, player_id_or_None)
    for seg in segments:
        name = seg.get("handler")
        pid = None
        if name:
            p = resolve_player_by_name(game, name)
            if p:
                pid = p.player_id
        resolved.append((seg["start_gc"], seg["end_gc"], pid))

    def lookup(gc):
        for s_gc, e_gc, pid in resolved:
            # gc counts down, so s_gc >= e_gc
            if e_gc <= gc <= s_gc:
                return pid
        return None

    return lookup


def render_video(game, pbp_events, period, gc_start, gc_end,
                 output_path, commentary=True, highlight_player_id=None,
                 show_handler=True):
    """Render an MP4 video for a time segment.

    Args:
        game: GameData instance
        pbp_events: normalised ESPN PBP list
        period: quarter number (1-4)
        gc_start: game clock at start (higher value, clock counts down)
        gc_end: game clock at end (lower value)
        output_path: file path for MP4 output
        commentary: include PBP commentary text below court
        highlight_player_id: optional static player_id to highlight
        show_handler: highlight the detected ball handler each frame

    Returns:
        Path to the created MP4 file, or None on failure.
    """
    moments = get_moments(game, period, gc_start, gc_end)
    if not moments:
        print(f"No moments found for period {period}, {gc_start} -> {gc_end}")
        return None

    home_id = game.home_team["teamid"]
    away_id = game.visitor_team["teamid"]
    home_abbr = game.home_team["abbreviation"]
    away_abbr = game.visitor_team["abbreviation"]
    team_colors = {-1: "#ff8c00", home_id: "#e74c3c", away_id: "#3498db"}

    # Build per-frame handler lookup
    handler_lookup = None
    if show_handler:
        print("Detecting ball handlers ...")
        handler_lookup = _build_handler_lookup(game, period, gc_start, gc_end)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Set up persistent figure (reused every frame) ---
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("white")

    # Court axes
    ax = fig.add_axes([0.02, 0.25, 0.96, 0.65])

    # Persistent text objects (updated each frame)
    score_text = fig.text(0.5, 0.94, "", ha="center", va="center",
                          color="black", fontsize=16, fontweight="bold")
    clock_text = fig.text(0.5, 0.91, "", ha="center", va="center",
                          color="black", fontsize=13)
    commentary_text = fig.text(0.5, 0.20, "", ha="center", va="top",
                               color="black", fontsize=10,
                               linespacing=1.4, fontstyle="italic")

    # --- ffmpeg pipe (RGBA rawvideo) ---
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        str(output_path),
    ]
    pipe = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    n = len(moments)
    print(f"Rendering {n} frames ({n/FPS:.1f}s) ...")

    for i, moment in enumerate(moments):
        # -- Clear court axes, redraw static court --
        ax.clear()
        ax.set_facecolor("#e8e8e8")
        draw_court(ax, color="black")
        ax.set_xlim(-5, 100)
        ax.set_ylim(-55, 5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # -- Determine which player to highlight --
        frame_highlight = highlight_player_id
        if handler_lookup is not None:
            detected = handler_lookup(moment.game_clock)
            if detected is not None:
                frame_highlight = detected

        # -- Frame data --
        x_pos, y_pos, colors, sizes, edges, jerseys = _extract_frame_data(
            moment, game, team_colors, frame_highlight)
        shot_clock_str, game_clock_str, quarter_str = _clock_strings(moment)
        commentary_script, score_str = get_commentary(
            pbp_events, moment.period, moment.game_clock)

        # -- Draw players & ball --
        ax.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=0.9,
                   linewidths=edges, edgecolors="black", zorder=5)

        for jx, jy, jersey in jerseys:
            ax.text(jx, jy, jersey, color="white", fontsize=6,
                    ha="center", va="center", fontweight="bold", zorder=6)

        # Team colour dots near top of court
        ax.scatter([30], [2.5], s=80, c=[team_colors[away_id]],
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.scatter([67], [2.5], s=80, c=[team_colors[home_id]],
                   edgecolors="black", linewidths=0.5, zorder=5)

        # -- Update text overlays --
        score_text.set_text(f"{away_abbr}   {score_str}   {home_abbr}")
        clock_text.set_text(
            f"{quarter_str}   {game_clock_str}      Shot: {shot_clock_str}")
        if commentary:
            commentary_text.set_text(commentary_script)
        else:
            commentary_text.set_text("")

        # -- Render to RGBA buffer and write to pipe --
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        pipe.stdin.write(bytes(buf))

        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  {i + 1}/{n} frames")

    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)

    if output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Video saved to {output_path} ({size_mb:.1f} MB)")
        return str(output_path)

    stderr = pipe.stderr.read().decode(errors="replace")
    print(f"Video rendering failed (ffmpeg exit {pipe.returncode})")
    if stderr:
        print(stderr[-500:])
    return None
