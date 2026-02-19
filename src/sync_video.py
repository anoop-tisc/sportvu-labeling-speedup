"""Sync broadcast video with SportVU court visualization.

OCRs the game clock from broadcast frames using Gemini Flash,
computes a linear alignment between video time and game clock,
then renders a stacked output (broadcast on top, court viz on bottom).
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google import genai

from src.sportvu_loader import load_game, find_moment
from src.pbp import fetch_pbp
from src.visualize import (
    draw_court, draw_court_3d, _extract_frame_data, _clock_strings,
    get_commentary, _build_handler_lookup, WIDTH, HEIGHT, DPI, FIG_W, FIG_H,
)

# Output layout
OUT_WIDTH = 1280
OUT_HEIGHT = 1440  # 720 broadcast + 720 court viz
OUT_FPS = 25


# ---------------------------------------------------------------------------
# Frame extraction helpers
# ---------------------------------------------------------------------------

def extract_frame_at(video_path: str, frame_idx: int) -> Image.Image:
    """Decode a single frame by index using ffmpeg, return as PIL Image."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select=eq(n\\,{frame_idx})",
        "-vframes", "1",
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-"
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for frame {frame_idx}: {proc.stderr[-300:]}")

    # Probe video dimensions
    w, h = _probe_dimensions(video_path)
    raw = proc.stdout
    expected = w * h * 3
    if len(raw) < expected:
        raise RuntimeError(f"Frame {frame_idx}: got {len(raw)} bytes, expected {expected}")
    return Image.frombytes("RGB", (w, h), raw[:expected])


def _probe_dimensions(video_path: str) -> tuple[int, int]:
    """Get video width and height via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def _probe_frame_count(video_path: str) -> int:
    """Get total number of video frames via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return int(out)


def _probe_fps(video_path: str) -> float:
    """Get video FPS via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    num, den = out.split("/")
    return int(num) / int(den)


def extract_all_frames(video_path: str) -> np.ndarray:
    """Decode all frames into a numpy array (N, H, W, 3) uint8."""
    w, h = _probe_dimensions(video_path)
    n_frames = _probe_frame_count(video_path)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "error",
        "-"
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr[-300:]}")
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_bytes = w * h * 3
    actual_frames = len(raw) // frame_bytes
    return raw[:actual_frames * frame_bytes].reshape(actual_frames, h, w, 3)


# ---------------------------------------------------------------------------
# Gemini Flash OCR
# ---------------------------------------------------------------------------

@dataclass
class ClockMarker:
    frame_idx: int
    clock_str: str
    game_clock_seconds: float


def _ocr_single_frame(img: Image.Image, client) -> str | None:
    """Send one frame to Gemini Flash, return clock string like '3:53' or None."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            img,
            "Read the NBA game clock from this broadcast frame. "
            "Return ONLY the game clock time in M:SS or MM:SS format (e.g. '3:53' or '11:02'). "
            "If you cannot read the game clock, return 'NONE'.",
        ],
    )
    text = response.text.strip()
    # Extract clock pattern from response
    match = re.search(r"(\d{1,2}:\d{2})", text)
    if match:
        return match.group(1)
    return None


def _clock_to_seconds(clock_str: str) -> float:
    """Convert '3:53' -> 233.0 seconds."""
    parts = clock_str.split(":")
    return int(parts[0]) * 60.0 + float(parts[1])


def find_clock_transition(video_path: str, client) -> ClockMarker:
    """Find the exact frame where the game clock first ticks to a new second.

    OCR frame 0 to get the starting clock value, then binary-search
    within the next fps frames (guaranteed to contain a 1-second transition).
    ~7 Gemini calls total (1 + log2(60)).
    """
    fps = _probe_fps(video_path)
    hi = int(fps)  # transition must occur within one second of frames

    # OCR frame 0 to get starting displayed second
    img0 = extract_frame_at(video_path, 0)
    start_clock = _ocr_single_frame(img0, client)
    if not start_clock:
        raise RuntimeError("Could not OCR frame 0")
    start_secs = _clock_to_seconds(start_clock)
    print(f"Frame 0: {start_clock} ({start_secs:.0f}s)")

    # The next displayed second (clock counts down)
    target_secs = start_secs - 1.0

    # Binary search [0, fps] for first frame showing target_secs
    lo = 0
    print(f"Binary-searching transition in frames [0, {hi}] ...")

    while hi - lo > 1:
        mid = (lo + hi) // 2
        img = extract_frame_at(video_path, mid)
        clock_str = _ocr_single_frame(img, client)
        if clock_str:
            secs = _clock_to_seconds(clock_str)
            if secs <= target_secs:
                hi = mid
            else:
                lo = mid
        else:
            lo = mid

    # hi is the first frame showing the new (lower) displayed second.
    # The clock displays "M:SS" for the entire interval (SS+1, SS].
    # So at the transition frame, the actual game clock just crossed
    # displayed_seconds + 1.0 (e.g., display changes to "4:49" means
    # game clock just crossed 290.0, not 289.0).
    actual_gc = target_secs + 1.0
    display_str = f"{int(target_secs // 60)}:{int(target_secs % 60):02d}"
    print(f"  Transition at frame {hi}: display={display_str}, game_clock={actual_gc:.0f}s")
    return ClockMarker(hi, display_str, actual_gc)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def compute_alignment(
    marker: ClockMarker, video_fps: float
) -> tuple[float, float]:
    """Compute alignment from a single transition marker.

    Slope is fixed at -1.0 (real-time playback).
    Returns (slope, intercept) for: game_clock = -1.0 * video_time + intercept
    """
    video_time = marker.frame_idx / video_fps
    # game_clock = -1.0 * video_time + intercept  =>  intercept = game_clock + video_time
    intercept = marker.game_clock_seconds + video_time

    print(f"\nAlignment: game_clock = -1.0 * video_time + {intercept:.2f}")
    print(f"  marker: frame {marker.frame_idx} (t={video_time:.2f}s), "
          f"display={marker.clock_str}, game_clock={marker.game_clock_seconds:.0f}s")

    return -1.0, float(intercept)


def save_alignment(path: str, slope: float, intercept: float,
                   marker: ClockMarker, video_fps: float):
    """Cache alignment to JSON."""
    data = {
        "slope": slope,
        "intercept": intercept,
        "video_fps": video_fps,
        "marker": {
            "frame_idx": marker.frame_idx,
            "clock_str": marker.clock_str,
            "game_clock_seconds": marker.game_clock_seconds,
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Alignment cached to {path}")


def load_alignment(path: str) -> tuple[float, float, float]:
    """Load cached alignment. Returns (slope, intercept, video_fps)."""
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded cached alignment from {path}")
    print(f"  slope={data['slope']:.4f}, intercept={data['intercept']:.2f}")
    return data["slope"], data["intercept"], data["video_fps"]


# ---------------------------------------------------------------------------
# Stacked video rendering
# ---------------------------------------------------------------------------

def _extract_3d_frame_data(moment, game, team_colors, highlight_player_id=None):
    """Extract 3D scatter data from a Moment. Separates ball (with z) from players (z=0)."""
    home_id = game.home_team["teamid"]
    # Players
    px, py, pz, pcolors, psizes, pedges = [], [], [], [], [], []
    jerseys = []
    # Ball
    bx, by, bz, bcolor, bsize = None, None, None, None, None

    for entry in moment.players:
        tid, pid, x, y = entry[0], entry[1], entry[2], entry[3]
        z = entry[4] if len(entry) > 4 else 0.0
        y_adj = -y  # Court drawn y=-50 to y=0

        if tid == -1:  # Ball
            bx, by, bz = x, y_adj, z
            bcolor = team_colors.get(tid, "gray")
            bsize = max(150 - 2 * (z - 5) ** 2, 10)
        else:
            px.append(x)
            py.append(y_adj)
            pz.append(0.0)
            pcolors.append(team_colors.get(tid, "gray"))
            psizes.append(200)
            if highlight_player_id is not None and pid == highlight_player_id:
                pedges.append(4)
            else:
                pedges.append(0.5)
            player = game.roster.get(pid)
            if player:
                jerseys.append((x, y_adj, 0.0, player.jersey))

    return (px, py, pz, pcolors, psizes, pedges, jerseys,
            bx, by, bz, bcolor, bsize)


def _setup_3d_ax(ax):
    """Configure a 3D axes for court visualization."""
    ax.set_xlim(-2, 96)
    ax.set_ylim(-52, 2)
    ax.set_zlim(0, 18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([0, 5, 10, 15])
    ax.set_zticklabels(['0', '5', '10', '15'], fontsize=6, color='#666')
    ax.set_zlabel('ft', fontsize=7, color='#666')
    ax.view_init(elev=30, azim=-55)
    ax.set_box_aspect([94, 50, 18])
    ax.dist = 5.5  # zoom in (default is ~10)
    # Lighten the pane backgrounds
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.set_facecolor('#e8e8e8')
    ax.zaxis.pane.set_alpha(0.3)
    ax.grid(False)


def _draw_flat_circle(ax, cx, cy, r, color, edgecolor='black', edgewidth=0.5, alpha=0.9, n=16):
    """Draw a filled circle flat on the z=0 plane using Poly3DCollection."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    theta = np.linspace(0, 2 * np.pi, n)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z = np.zeros(n)
    verts = [list(zip(x, y, z))]
    poly = Poly3DCollection(verts, facecolors=[color], edgecolors=[edgecolor],
                            linewidths=[edgewidth], alpha=alpha)
    ax.add_collection3d(poly)


def render_synced_video(
    video_path: str,
    game,
    pbp_events: list[dict],
    period: int,
    slope: float,
    intercept: float,
    output_path: str,
    commentary: bool = True,
    show_handler: bool = True,
    use_3d: bool = False,
):
    """Render stacked video: broadcast on top, court viz on bottom."""
    video_fps = _probe_fps(video_path)
    bw, bh = _probe_dimensions(video_path)

    print(f"\nDecoding all broadcast frames ...")
    broadcast_frames = extract_all_frames(video_path)
    n_broadcast = len(broadcast_frames)
    duration = n_broadcast / video_fps
    print(f"  {n_broadcast} frames, {video_fps:.1f} FPS, {duration:.2f}s")

    n_out = int(duration * OUT_FPS)
    print(f"Output: {n_out} frames at {OUT_FPS} FPS, {OUT_WIDTH}x{OUT_HEIGHT}")
    if use_3d:
        print("  3D rendering enabled")

    # Determine game clock range for handler detection
    gc_start = slope * 0 + intercept
    gc_end = slope * duration + intercept
    if gc_start < gc_end:
        gc_start, gc_end = gc_end, gc_start

    # Team colors and identifiers
    home_id = game.home_team["teamid"]
    away_id = game.visitor_team["teamid"]
    home_abbr = game.home_team["abbreviation"]
    away_abbr = game.visitor_team["abbreviation"]
    team_colors = {-1: "#ff8c00", home_id: "#e74c3c", away_id: "#3498db"}

    # Handler lookup
    handler_lookup = None
    if show_handler:
        print("Detecting ball handlers ...")
        handler_lookup = _build_handler_lookup(game, period, gc_start, gc_end)

    # Set up persistent matplotlib figure for court viz
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("white")

    if use_3d:
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=-0.1, right=1.1, top=0.85, bottom=0.05)
    else:
        ax = fig.add_axes([0.02, 0.25, 0.96, 0.65])

    score_text = fig.text(0.5, 0.94, "", ha="center", va="center",
                          color="black", fontsize=16, fontweight="bold")
    clock_text = fig.text(0.5, 0.91, "", ha="center", va="center",
                          color="black", fontsize=13)
    commentary_text = fig.text(0.5, 0.08, "" if use_3d else "", ha="center",
                               va="top", color="black", fontsize=10,
                               linespacing=1.4, fontstyle="italic")

    # ffmpeg output pipe — takes raw RGBA, produces H.264
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{OUT_WIDTH}x{OUT_HEIGHT}",
        "-r", str(OUT_FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        str(out_path),
    ]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print(f"Rendering {n_out} frames ...")

    for i in range(n_out):
        t = i / OUT_FPS  # output time in seconds

        # --- Top half: broadcast frame ---
        bc_idx = min(int(round(t * video_fps)), n_broadcast - 1)
        bc_frame = broadcast_frames[bc_idx]  # (H, W, 3) uint8

        # Resize broadcast to OUT_WIDTH x 720 if needed
        if bc_frame.shape[1] != OUT_WIDTH or bc_frame.shape[0] != 720:
            bc_img = Image.fromarray(bc_frame).resize((OUT_WIDTH, 720), Image.LANCZOS)
            bc_rgba = np.array(bc_img.convert("RGBA"))
        else:
            # Add alpha channel
            bc_rgba = np.concatenate(
                [bc_frame, np.full((*bc_frame.shape[:2], 1), 255, dtype=np.uint8)],
                axis=2,
            )

        # --- Bottom half: court visualization ---
        game_clock = slope * t + intercept
        moment = find_moment(game, period, game_clock)

        if use_3d:
            # 3D rendering path
            ax.clear()
            ax.set_facecolor('white')
            draw_court_3d(ax, color="#555", lw=0.8)
            _setup_3d_ax(ax)

            if moment:
                frame_highlight = None
                if handler_lookup is not None:
                    frame_highlight = handler_lookup(moment.game_clock)

                (px, py, pz, pcolors, psizes, pedges, jerseys,
                 bx, by, bz, bcolor, bsize) = _extract_3d_frame_data(
                    moment, game, team_colors, frame_highlight)
                shot_clock_str, game_clock_str, quarter_str = _clock_strings(moment)
                commentary_script, score_str = get_commentary(
                    pbp_events, moment.period, moment.game_clock)

                # Players as flat circles on the floor
                for j in range(len(px)):
                    ew = pedges[j] if j < len(pedges) else 0.5
                    _draw_flat_circle(ax, px[j], py[j], r=1.5,
                                      color=pcolors[j], edgewidth=ew)
                for jx, jy, jz, jersey in jerseys:
                    ax.text(jx, jy, 0.1, jersey, color="white", fontsize=5,
                            ha="center", va="center", fontweight="bold")

                # Ball in 3D with shadow
                if bx is not None:
                    ax.scatter([bx], [by], [bz], c=[bcolor], s=[bsize],
                               alpha=0.95, edgecolors="black", linewidths=0.5,
                               depthshade=False)
                    # Shadow line from ball down to floor
                    ax.plot([bx, bx], [by, by], [0, bz],
                            color='#ff8c00', alpha=0.4, lw=1, linestyle='--')
                    # Shadow dot on floor
                    ax.scatter([bx], [by], [0], c=['#ff8c00'], s=[40],
                               alpha=0.3, edgecolors='none')

                score_text.set_text(f"{away_abbr}   {score_str}   {home_abbr}")
                clock_text.set_text(
                    f"{quarter_str}   {game_clock_str}      Shot: {shot_clock_str}")
                if commentary:
                    commentary_text.set_text(commentary_script)
                else:
                    commentary_text.set_text("")
            else:
                score_text.set_text("")
                clock_text.set_text("")
                commentary_text.set_text("")
        else:
            # 2D rendering path (original)
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

            if moment:
                frame_highlight = None
                if handler_lookup is not None:
                    frame_highlight = handler_lookup(moment.game_clock)

                x_pos, y_pos, colors, sizes, edges, jerseys = _extract_frame_data(
                    moment, game, team_colors, frame_highlight)
                shot_clock_str, game_clock_str, quarter_str = _clock_strings(moment)
                commentary_script, score_str = get_commentary(
                    pbp_events, moment.period, moment.game_clock)

                ax.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=0.9,
                           linewidths=edges, edgecolors="black", zorder=5)
                for jx, jy, jersey in jerseys:
                    ax.text(jx, jy, jersey, color="white", fontsize=6,
                            ha="center", va="center", fontweight="bold", zorder=6)
                ax.scatter([30], [2.5], s=80, c=[team_colors[away_id]],
                           edgecolors="black", linewidths=0.5, zorder=5)
                ax.scatter([67], [2.5], s=80, c=[team_colors[home_id]],
                           edgecolors="black", linewidths=0.5, zorder=5)

                score_text.set_text(f"{away_abbr}   {score_str}   {home_abbr}")
                clock_text.set_text(
                    f"{quarter_str}   {game_clock_str}      Shot: {shot_clock_str}")
                if commentary:
                    commentary_text.set_text(commentary_script)
                else:
                    commentary_text.set_text("")
            else:
                score_text.set_text("")
                clock_text.set_text("")
                commentary_text.set_text("")

        fig.canvas.draw()
        court_rgba = np.asarray(fig.canvas.buffer_rgba()).copy()  # (720, 1280, 4)

        # --- Stack top + bottom ---
        stacked = np.concatenate([bc_rgba, court_rgba], axis=0)  # (1440, 1280, 4)
        pipe.stdin.write(stacked.tobytes())

        if (i + 1) % 30 == 0 or i == n_out - 1:
            print(f"  {i + 1}/{n_out} frames")

    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)

    if out_path.exists() and out_path.stat().st_size > 0:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"\nSynced video saved to {out_path} ({size_mb:.1f} MB)")
        return str(out_path)

    stderr = pipe.stderr.read().decode(errors="replace")
    print(f"Video rendering failed (ffmpeg exit {pipe.returncode})")
    if stderr:
        print(stderr[-500:])
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sync broadcast video with SportVU court visualization")
    parser.add_argument("--video", default="data/0021500492_event64.mp4",
                        help="Path to broadcast video clip")
    parser.add_argument("--game", default="data/0021500492.json",
                        help="Path to SportVU game JSON")
    parser.add_argument("--period", type=int, default=1,
                        help="Game period (quarter)")
    parser.add_argument("--output", default="outputs/synced_event64.mp4",
                        help="Output video path")
    parser.add_argument("--cache-alignment",
                        help="JSON file to cache/load alignment (skip OCR on re-runs)")
    parser.add_argument("--no-handler", action="store_true",
                        help="Disable ball handler detection")
    parser.add_argument("--no-commentary", action="store_true",
                        help="Disable PBP commentary overlay")
    parser.add_argument("--3d", dest="use_3d", action="store_true",
                        help="Render court visualization in 3D (shows ball height)")
    args = parser.parse_args()

    # Load alignment from cache, or run OCR
    if args.cache_alignment and Path(args.cache_alignment).exists():
        slope, intercept, video_fps = load_alignment(args.cache_alignment)
        markers = []
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY not set. Add it to .env or environment.")
            sys.exit(1)

        client = genai.Client(api_key=api_key)
        video_fps = _probe_fps(args.video)

        # Binary-search for the first clock-second transition
        marker = find_clock_transition(args.video, client)
        slope, intercept = compute_alignment(marker, video_fps)

        if args.cache_alignment:
            save_alignment(args.cache_alignment, slope, intercept,
                           marker, video_fps)

    # Load game data
    print(f"\nLoading game data from {args.game} ...")
    game = load_game(args.game)
    pbp_events = fetch_pbp(game.game_id)
    print(f"  {len(pbp_events)} PBP events loaded")

    # Render synced video
    render_synced_video(
        video_path=args.video,
        game=game,
        pbp_events=pbp_events,
        period=args.period,
        slope=slope,
        intercept=intercept,
        output_path=args.output,
        commentary=not args.no_commentary,
        show_handler=not args.no_handler,
        use_3d=args.use_3d,
    )


if __name__ == "__main__":
    main()
