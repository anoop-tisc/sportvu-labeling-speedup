"""Run agent test questions and output results as markdown."""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.sportvu_loader import load_game, print_game_info
from src.pbp import fetch_pbp
from src.possession import identify_possessions, Possession
from src.agent import run_agent

GAME_PATH = "data/0021500492.json"
GAME_ID = "0021500492"
PERIOD = 1
GC_START = 290.0
GC_END = 282.0
TEAM = "TOR"

QUESTIONS = [
    "At the start of this clip, who has the ball and where are all 10 players positioned?",
    "Are there any passes during this segment? If so, describe each pass.",
    "What are the defensive matchups and how tight is the coverage?",
]

OUTPUT_PATH = "outputs/agent_test_output_v3.md"


def main():
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
        start_gc=GC_START,
        end_gc=GC_END,
    )

    gc_start_mm = int(GC_START) // 60
    gc_start_ss = GC_START - gc_start_mm * 60
    gc_end_mm = int(GC_END) // 60
    gc_end_ss = GC_END - gc_end_mm * 60

    lines = []
    lines.append(f"# Agent Test Output v3 — Q1 {gc_start_mm}:{gc_start_ss:04.1f} to {gc_end_mm}:{gc_end_ss:04.1f}")
    lines.append("")
    lines.append(f"**Game:** {GAME_ID} — CHA @ TOR, 2016-01-01")
    lines.append(f"**Segment:** Period {PERIOD}, game clock {GC_START}s ({gc_start_mm}:{gc_start_ss:04.1f}) to {GC_END}s ({gc_end_mm}:{gc_end_ss:04.1f})")
    lines.append("**Changes since v2:** Replaced zone classification with landmark-based position descriptions. All positions described from offensive team's perspective. Fixed system prompt to include raw gc values.")
    lines.append("")

    for i, question in enumerate(QUESTIONS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(QUESTIONS)}] Q: {question}")
        print(f"{'='*60}")

        answer = run_agent(
            question, game, pbp_events,
            period=PERIOD,
            gc_start=GC_START,
            gc_end=GC_END,
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
