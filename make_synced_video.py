#!/usr/bin/env python3
"""
make_synced_video.py — Create a synced broadcast + court visualization video.

Pipeline overview:
  1. Determine the alignment offset: the game clock (in ms) at video frame 0.
     Then for any frame:  game_clock = offset - video_time
  2. Load SportVU tracking data (player/ball XY at 25 fps) and play-by-play events.
  3. Render synced video: broadcast on top, animated court on the bottom.

Alignment modes:

  Simple (default) — Gemini OCR reads the broadcast clock, ~300-400ms accuracy:
    python make_synced_video.py --game data/0021500492.json

  Precise with known offset — you already know the exact value:
    python make_synced_video.py --game data/0021500492.json --alignment precise --offset 705822

  Precise with auto-detection — OCR + SportVU hoop detection + SAM3 tracking:
    python make_synced_video.py --game data/0021500492.json --alignment precise --offset auto
"""

import argparse
import os
import shutil
import sys

from src.sportvu_loader import load_game
from src.pbp import fetch_pbp
from src.sync_video import render_synced_video

# Defaults for our checked-in game data (CHA @ TOR, 2016-01-01)
DEFAULT_VIDEO = "data/0021500492_event64.mp4"
DEFAULT_OUTPUT = "outputs/synced_clip.mp4"


def _ocr_alignment(video_path):
    """Gemini OCR: find a clock transition and compute alignment offset.

    Returns (offset_sec, video_fps).  offset_sec is the game clock (seconds)
    at video frame 0.
    """
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)
    from google import genai
    from src.sync_video import find_clock_transition, compute_alignment, _probe_fps
    client = genai.Client(api_key=api_key)
    video_fps = _probe_fps(video_path)
    print("Finding clock transition via Gemini OCR ...")
    marker = find_clock_transition(video_path, client)
    _slope, offset_sec = compute_alignment(marker, video_fps)
    return offset_sec, video_fps


def _refine_with_sam3(video_path, game, period, rough_offset_sec, video_fps):
    """Refine a rough offset using SportVU hoop detection + SAM3 tracking.

    Uses the rough offset to derive the game clock range of the video, finds
    a made basket in the SportVU data, then runs SAM3 on a ~1s window around
    the estimated hoop frame to pinpoint the exact crossing.

    Returns the refined offset in seconds.
    """
    from src.sync_video import _probe_frame_count
    from src.auto_align import (
        find_hoop_entry_gc, _extract_frame_range_to_dir,
        track_objects_sam3, find_crossing_frame,
    )

    total_frames = _probe_frame_count(video_path)
    duration = total_frames / video_fps

    # Derive game clock range from rough offset + video duration
    gc_start = rough_offset_sec              # game clock at frame 0
    gc_end = rough_offset_sec - duration     # game clock at last frame

    # Find hoop entry in SportVU tracking data
    print("\nFinding made basket in SportVU data ...")
    gc_hoop = find_hoop_entry_gc(game, period, gc_start, gc_end)

    # Estimate broadcast frame of hoop entry using rough offset
    video_time_est = rough_offset_sec - gc_hoop
    frame_est = int(round(video_time_est * video_fps))
    print(f"  Estimated hoop frame: {frame_est} (video_time={video_time_est:.2f}s)")

    # Extract a ~1s window around the estimate and run SAM3
    search_window_sec = 1.0
    window_frames = int(search_window_sec * video_fps)
    win_start = max(0, frame_est - window_frames)
    win_end = min(frame_est + int(video_fps * 0.2), total_frames)
    print(f"\nRunning SAM3 on frames {win_start}-{win_end} ({win_end - win_start} frames) ...")

    frames_dir, frame_count = _extract_frame_range_to_dir(video_path, win_start, win_end)
    tracking = track_objects_sam3(frames_dir, frame_count)
    crossing_in_window = find_crossing_frame(
        tracking["ball_centroids"], tracking["rim_centroids"], tracking["rim_x_extents"]
    )
    shutil.rmtree(frames_dir, ignore_errors=True)

    # Compute refined offset
    frame_hoop = win_start + crossing_in_window
    video_time_hoop = frame_hoop / video_fps
    offset_sec = gc_hoop + video_time_hoop

    print(f"\n  Refined offset: {offset_sec:.3f}s (was {rough_offset_sec:.3f}s, "
          f"diff={offset_sec - rough_offset_sec:.3f}s)")
    return offset_sec


def main():
    parser = argparse.ArgumentParser(
        description="Create a synced broadcast + court visualization video.")
    parser.add_argument("--video", default=DEFAULT_VIDEO,
                        help="Path to broadcast video clip")
    parser.add_argument("--game", required=True,
                        help="Path to SportVU game JSON (e.g. data/0021500492.json)")
    parser.add_argument("--period", type=int, default=1,
                        help="Game period / quarter (default: 1)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output video path")
    parser.add_argument("--3d", dest="use_3d", action="store_true",
                        help="Render court in 3D (shows ball height)")

    # Alignment
    parser.add_argument("--alignment", choices=["simple", "precise"], default="simple",
                        help="simple = Gemini OCR only (~300-400ms off); "
                             "precise = exact offset (default: simple)")
    parser.add_argument("--offset", type=str,
                        help="For --alignment precise: offset in ms (int) or 'auto'. "
                             "The offset is the game clock (ms) at video frame 0.")

    args = parser.parse_args()

    # --- Determine alignment offset ---
    if args.alignment == "simple":
        if args.offset is not None:
            parser.error("--offset is only used with --alignment precise")
        offset_sec, _fps = _ocr_alignment(args.video)

    else:  # precise
        if args.offset is None:
            parser.error("--alignment precise requires --offset <ms> or --offset auto")

        if args.offset == "auto":
            # OCR for rough offset, then refine with SportVU + SAM3
            rough_offset_sec, video_fps = _ocr_alignment(args.video)
            game = load_game(args.game)
            offset_sec = _refine_with_sam3(
                args.video, game, args.period, rough_offset_sec, video_fps)
        else:
            try:
                offset_ms = int(args.offset)
            except ValueError:
                parser.error(f"--offset must be an integer (ms) or 'auto', got '{args.offset}'")
            offset_sec = offset_ms / 1000.0
            print(f"Using offset: {offset_ms} ms ({offset_sec:.3f}s)")

    # --- Load tracking data and play-by-play ---
    print(f"\nLoading game data from {args.game} ...")
    game = load_game(args.game)
    pbp_events = fetch_pbp(game.game_id)
    print(f"  {len(pbp_events)} PBP events loaded")

    # --- Render ---
    render_synced_video(
        video_path=args.video,
        game=game,
        pbp_events=pbp_events,
        period=args.period,
        slope=-1.0,
        intercept=offset_sec,
        output_path=args.output,
        use_3d=args.use_3d,
    )


if __name__ == "__main__":
    main()
