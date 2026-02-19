"""Run agent test questions and output results as markdown."""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.sportvu_loader import load_game, print_game_info
from src.pbp import fetch_pbp
from src.possession import identify_possessions, Possession
from src.agent import run_agent
from src.sync_video import load_alignment, _probe_fps, _probe_frame_count

GAME_PATH = "data/0021500492.json"
GAME_ID = "0021500492"
PERIOD = 1
TEAM = "TOR"
VIDEO_PATH = "data/clip_q4.mp4"
ALIGNMENT_PATH = "outputs/alignment.json"

QUESTIONS = [
    "At the start of this clip, who has the ball and where are all 10 players positioned?",
    "Are there any passes during this segment? If so, describe each pass.",
    "What are the defensive matchups and how tight is the coverage?",
]

OUTPUT_PATH = "outputs/agent_test_output_v4.md"


def main():
    # Derive time range from alignment + video duration
    slope, intercept, video_fps = load_alignment(ALIGNMENT_PATH)
    n_frames = _probe_frame_count(VIDEO_PATH)
    duration = n_frames / video_fps
    gc_start = intercept                       # video_time=0 → gc
    gc_end = slope * duration + intercept      # video_time=duration → gc
    print(f"Clip time range: gc {gc_start:.2f} → {gc_end:.2f} (duration {duration:.2f}s)")

    print(f"Loading game data from {GAME_PATH}...")
    game = load_game(GAME_PATH)
    print_game_info(game)

    print(f"\nFetching play-by-play for {GAME_ID}...")
    pbp_events = fetch_pbp(GAME_ID)
    print(f"Loaded {len(pbp_events)} PBP events")

    possessions = identify_possessions(pbp_events, game)
    print(f"Identified {len(possessions)} possessions")

    possession = Possession(
        team_id="custom",
        team_abbr=TEAM,
        period=PERIOD,
        start_gc=gc_start,
        end_gc=gc_end,
    )

    import math
    def _broadcast_clock(gc):
        gc_ceil = math.ceil(gc)
        return f"{gc_ceil // 60}:{gc_ceil % 60:02d}"

    lines = []
    lines.append(f"# Agent Test Output v4 — Q1 {_broadcast_clock(gc_start)} to {_broadcast_clock(gc_end)}")
    lines.append("")
    lines.append(f"**Game:** {GAME_ID} — CHA @ TOR, 2016-01-01")
    lines.append(f"**Segment:** Period {PERIOD}, game clock {_broadcast_clock(gc_start)} to {_broadcast_clock(gc_end)}")
    lines.append("**Changes since v3:** Time range derived from alignment + video duration for accurate broadcast-to-tracking sync.")
    lines.append("")

    for i, question in enumerate(QUESTIONS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(QUESTIONS)}] Q: {question}")
        print(f"{'='*60}")

        answer = run_agent(
            question, game, pbp_events,
            period=PERIOD,
            gc_start=gc_start,
            gc_end=gc_end,
            team_abbr=TEAM,
            verbose=True,
        )

        lines.append("---")
        lines.append("")
        lines.append(f"## Q{i+1}: {question}")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(answer)
        lines.append("")

    md = "\n".join(lines)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PATH).write_text(md)
    print(f"\nOutput saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
