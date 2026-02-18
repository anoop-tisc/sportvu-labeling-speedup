"""CLI entry point: pick game/possession, run agent interactively."""

import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.sportvu_loader import load_game, print_game_info
from src.pbp import fetch_pbp
from src.possession import identify_possessions, list_possessions, Possession
from src.agent import run_agent


def pick_possession(possessions: list[Possession], period: int | None = None) -> Possession:
    """Interactive possession picker."""
    lines = list_possessions(possessions, period=period)
    if not lines:
        print("No possessions found for the specified period.")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print(f"Found {len(lines)} possessions" + (f" in period {period}" if period else ""))
    print(f"{'=' * 80}")
    for line in lines[:50]:  # Show first 50
        print(line)
    if len(lines) > 50:
        print(f"  ... and {len(lines) - 50} more. Use --period to filter.")

    while True:
        try:
            idx = int(input("\nEnter possession index: "))
            if 0 <= idx < len(possessions):
                p = possessions[idx]
                if period is not None and p.period != period:
                    print(f"Possession {idx} is in period {p.period}, not {period}. Use anyway? (y/n)")
                    if input().strip().lower() != "y":
                        continue
                return p
            print(f"Index must be 0-{len(possessions) - 1}")
        except (ValueError, EOFError):
            print("Enter a valid number.")


def run_interactive(game, pbp_events, possession, verbose=False):
    """Interactive Q&A loop for a possession."""
    print(f"\n{'=' * 80}")
    print(f"Possession: {possession.summary()}")
    print(f"Period {possession.period}, {possession.start_gc:.0f}s → {possession.end_gc:.0f}s")
    print(f"Team: {possession.team_abbr}")
    print(f"{'=' * 80}")
    print("Ask questions about this possession. Type 'quit' to exit, 'switch' to change possession.\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "switch":
            return "switch"

        print()
        answer = run_agent(
            question, game, pbp_events,
            period=possession.period,
            gc_start=possession.start_gc,
            gc_end=possession.end_gc,
            team_abbr=possession.team_abbr,
            verbose=verbose,
        )
        print(f"A: {answer}\n")

    return "quit"


def run_batch(game, pbp_events, possession, questions_file, verbose=False):
    """Run a batch of questions from a file."""
    with open(questions_file) as f:
        questions = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    results = []
    for i, question in enumerate(questions):
        print(f"\n[{i + 1}/{len(questions)}] Q: {question}")
        answer = run_agent(
            question, game, pbp_events,
            period=possession.period,
            gc_start=possession.start_gc,
            gc_end=possession.end_gc,
            team_abbr=possession.team_abbr,
            verbose=verbose,
        )
        print(f"A: {answer}")
        results.append({"question": question, "answer": answer})

    # Save results
    out_path = Path(questions_file).stem + "_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SportVU Basketball Analytics Agent")
    parser.add_argument("--game", default="data/0021500492.json",
                        help="Path to SportVU game JSON file")
    parser.add_argument("--game-id", default="0021500492",
                        help="NBA game ID (for PBP lookup)")
    parser.add_argument("--espn-id", default=None,
                        help="ESPN event ID (if not in the built-in map)")
    parser.add_argument("--period", type=int, default=None,
                        help="Filter possessions to this period")
    parser.add_argument("--possession", type=int, default=None,
                        help="Possession index (skip interactive picker)")
    parser.add_argument("--gc-start", type=float, default=None,
                        help="Custom start game clock (overrides possession)")
    parser.add_argument("--gc-end", type=float, default=None,
                        help="Custom end game clock (overrides possession)")
    parser.add_argument("--team", default=None,
                        help="Team abbreviation for custom range (e.g., TOR, CHA)")
    parser.add_argument("--questions", default=None,
                        help="Path to questions file for batch mode")
    parser.add_argument("--render", action="store_true",
                        help="Render MP4 video for the selected possession/time range")
    parser.add_argument("--output-dir", default="outputs/",
                        help="Directory for rendered videos (default: outputs/)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show agent tool calls")
    args = parser.parse_args()

    # Load game data
    print(f"Loading game data from {args.game}...")
    game = load_game(args.game)
    print_game_info(game)

    # Fetch PBP
    print(f"\nFetching play-by-play for {args.game_id}...")
    pbp_events = fetch_pbp(args.game_id, espn_event_id=args.espn_id)
    print(f"Loaded {len(pbp_events)} PBP events")

    # Identify possessions
    possessions = identify_possessions(pbp_events, game)
    print(f"Identified {len(possessions)} possessions")

    # Custom time range
    if args.gc_start is not None and args.gc_end is not None:
        team = args.team or game.home_team["abbreviation"]
        possession = Possession(
            team_id="custom",
            team_abbr=team,
            period=args.period or 1,
            start_gc=args.gc_start,
            end_gc=args.gc_end,
        )
    elif args.possession is not None:
        if args.possession >= len(possessions):
            print(f"Possession index {args.possession} out of range (0-{len(possessions) - 1})")
            sys.exit(1)
        possession = possessions[args.possession]
    else:
        possession = pick_possession(possessions, period=args.period)

    # Render video if requested
    if args.render:
        from src.visualize import render_video
        out_dir = Path(args.output_dir)
        gc_s = possession.start_gc
        gc_e = possession.end_gc
        fname = f"P{possession.period}_{gc_s:.0f}_{gc_e:.0f}.mp4"
        video_path = render_video(
            game, pbp_events,
            period=possession.period,
            gc_start=gc_s,
            gc_end=gc_e,
            output_path=out_dir / fname,
        )
        if video_path:
            print(f"Video: {video_path}")

    # Run
    if args.questions:
        run_batch(game, pbp_events, possession, args.questions, verbose=args.verbose)
    else:
        while True:
            result = run_interactive(game, pbp_events, possession, verbose=args.verbose)
            if result == "switch":
                possession = pick_possession(possessions, period=args.period)
            else:
                break


if __name__ == "__main__":
    main()
