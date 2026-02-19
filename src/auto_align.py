"""Automatic video-to-tracking alignment via ball-through-hoop detection.

Uses SportVU tracking data (ball z-crossing at rim height) and Gemini + SAM 2
video tracking (ball centroid crossing rim centroid) to find the hoop-entry
moment in both data sources, then computes alignment.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from src.config import LEFT_BASKET, RIGHT_BASKET
from src.sportvu_loader import load_game, get_moments
from src.sync_video import (
    _probe_fps,
    _probe_frame_count,
    _probe_dimensions,
    compute_alignment,
    save_alignment,
    ClockMarker,
    extract_frame_at,
    _ocr_single_frame,
    _clock_to_seconds,
)

# Rim height in feet
RIM_HEIGHT = 10.0
# Maximum horizontal distance from basket center for ball to be "at the rim"
RIM_RADIUS_FT = 0.75


# ---------------------------------------------------------------------------
# Part A: SportVU detection
# ---------------------------------------------------------------------------

def find_hoop_entry_gc(
    game, period: int, gc_start: float, gc_end: float
) -> float:
    """Find the game clock when the ball descends through the rim.

    Scans consecutive SportVU moments for ball z crossing 10.0 ft (rim height)
    while the ball (x, y) is within RIM_RADIUS_FT of either basket.

    Returns the interpolated game clock at the crossing (sub-frame precision).
    """
    moments = get_moments(game, period, gc_start, gc_end)
    if len(moments) < 2:
        raise RuntimeError(
            f"Not enough moments in period {period}, gc [{gc_end}, {gc_start}]"
        )

    baskets = [LEFT_BASKET, RIGHT_BASKET]

    for i in range(len(moments) - 1):
        m_before = moments[i]
        m_after = moments[i + 1]

        # Find ball entity (team_id == -1) in each moment
        ball_before = _find_ball(m_before)
        ball_after = _find_ball(m_after)
        if ball_before is None or ball_after is None:
            continue

        x_b, y_b, z_b = ball_before
        x_a, y_a, z_a = ball_after

        # Check: ball z descends through rim height
        if z_b <= RIM_HEIGHT or z_a >= RIM_HEIGHT:
            continue

        # Check: ball (x, y) is near a basket in both frames
        near_basket = False
        for bx, by in baskets:
            dist_b = ((x_b - bx) ** 2 + (y_b - by) ** 2) ** 0.5
            dist_a = ((x_a - bx) ** 2 + (y_a - by) ** 2) ** 0.5
            if dist_b < RIM_RADIUS_FT and dist_a < RIM_RADIUS_FT:
                near_basket = True
                break

        if not near_basket:
            continue

        # Interpolate game clock at z = RIM_HEIGHT
        frac = (z_b - RIM_HEIGHT) / (z_b - z_a)
        gc_hoop = m_before.game_clock + (m_after.game_clock - m_before.game_clock) * frac

        print(f"SportVU hoop entry detected:")
        print(f"  Frame before: gc={m_before.game_clock:.3f}, ball z={z_b:.2f}")
        print(f"  Frame after:  gc={m_after.game_clock:.3f}, ball z={z_a:.2f}")
        print(f"  Interpolated: gc={gc_hoop:.3f}")
        return gc_hoop

    raise RuntimeError(
        "No ball-through-hoop event found in SportVU data for the given range"
    )


def _find_ball(moment) -> tuple[float, float, float] | None:
    """Extract ball (x, y, z) from a Moment. Ball has team_id == -1."""
    for entry in moment.players:
        if entry[0] == -1:
            return entry[2], entry[3], entry[4] if len(entry) > 4 else 0.0
    return None


# ---------------------------------------------------------------------------
# Part B: Broadcast detection via Gemini + SAM 2
# ---------------------------------------------------------------------------

def _extract_tail_clip(video_path: str, duration_sec: float = 5.0) -> str:
    """Extract the last `duration_sec` seconds of a video to a temp file."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    total_dur = float(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
    start_time = max(0, total_dur - duration_sec)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        tmp.name,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg tail extract failed: {proc.stderr[-300:]}")
    print(f"Extracted last {duration_sec}s of video to {tmp.name}")
    return tmp.name


def _extract_frames_to_dir(video_path: str) -> tuple[str, int]:
    """Extract all frames from a video to a temp directory of JPEGs.

    Returns (dir_path, frame_count).
    """
    tmpdir = tempfile.mkdtemp(prefix="sam2_frames_")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-q:v", "2",
        "-start_number", "0",
        f"{tmpdir}/%06d.jpg",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {proc.stderr[-300:]}")

    frame_count = len(list(Path(tmpdir).glob("*.jpg")))
    print(f"Extracted {frame_count} frames to {tmpdir}")
    return tmpdir, frame_count


def _detect_objects_gemini(
    frame_path: str,
) -> dict:
    """Use Gemini Flash to detect basketball and rim bounding boxes in a frame.

    Returns dict with "ball_box" and "rim_box", each [x1, y1, x2, y2] in pixels,
    or None if not detected.
    """
    from google import genai
    from PIL import Image

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    img = Image.open(frame_path)
    w, h = img.size

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            img,
            "Find the basketball and the basketball rim/hoop in this NBA broadcast frame. "
            "Return bounding boxes as JSON with this exact format:\n"
            '{"ball": [x1, y1, x2, y2], "rim": [x1, y1, x2, y2]}\n'
            "Coordinates should be in pixels. The image is "
            f"{w}x{h} pixels. "
            "If an object is not visible, use null for its value. "
            "Return ONLY the JSON, no other text.",
        ],
    )

    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    data = json.loads(text)
    print(f"  Gemini detections: ball={data.get('ball')}, rim={data.get('rim')}")
    return {
        "ball_box": data.get("ball"),
        "rim_box": data.get("rim"),
    }


def track_objects_sam2(
    video_path: str, tail_seconds: float = 5.0
) -> dict:
    """Track basketball and rim through broadcast clip using Gemini + SAM 2.

    Uses the full clip (not just the tail) and initializes detection on a
    frame well before the expected scoring event for reliable tracking.

    1. Extract frames from the full clip
    2. Use Gemini Flash to detect ball and rim bounding boxes on an early frame
    3. Use SAM 2 video predictor to track both objects through all frames
    4. Return per-frame centroids

    Returns dict with:
        "rim_centroids": list of (cx, cy) or None per frame
        "ball_centroids": list of (cx, cy) or None per frame
        "frame_count": int
        "fps": float
    """
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    fps = _probe_fps(video_path)

    # Extract ALL frames from the clip (clips are short, typically <15s)
    frames_dir, frame_count = _extract_frames_to_dir(video_path)

    # Pick an early frame for Gemini detection (1 second in, or frame 0 if clip is very short)
    # This ensures the ball is visible and above the rim before the scoring event
    init_frame_idx = min(int(fps), frame_count - 1)  # ~1s in
    init_frame_path = f"{frames_dir}/{init_frame_idx:06d}.jpg"
    print(f"Detecting ball and rim with Gemini Flash on frame {init_frame_idx} ...")
    detections = _detect_objects_gemini(init_frame_path)

    if detections["ball_box"] is None:
        raise RuntimeError(f"Gemini could not detect the basketball in frame {init_frame_idx}")
    if detections["rim_box"] is None:
        raise RuntimeError(f"Gemini could not detect the basketball rim in frame {init_frame_idx}")

    # Build SAM 2 video predictor
    ckpt = "/home/gcpuser/sky_workdir/cv-pipeline/vendor/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    print("Building SAM 2 video predictor ...")
    predictor = build_sam2_video_predictor(cfg, ckpt)

    # Initialize state with the frames directory
    inference_state = predictor.init_state(video_path=frames_dir)

    # Add rim prompt (obj_id=1) on the init frame
    rim_box = np.array(detections["rim_box"], dtype=np.float32)
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=init_frame_idx,
        obj_id=1,
        box=rim_box,
    )
    print(f"  Rim prompt added (obj_id=1), box={detections['rim_box']}")

    # Add ball prompt (obj_id=2) on the init frame
    ball_box = np.array(detections["ball_box"], dtype=np.float32)
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=init_frame_idx,
        obj_id=2,
        box=ball_box,
    )
    print(f"  Ball prompt added (obj_id=2), box={detections['ball_box']}")

    # Propagate through all frames
    print(f"Propagating SAM 2 tracking through {frame_count} frames ...")
    rim_centroids = [None] * frame_count
    ball_centroids = [None] * frame_count

    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(inference_state):
        # video_res_masks: (num_objects, 1, H, W) tensor
        for i, obj_id in enumerate(obj_ids):
            mask = (video_res_masks[i, 0] > 0.0).cpu().numpy()
            centroid = _mask_centroid(mask)
            if obj_id == 1:
                rim_centroids[frame_idx] = centroid
            elif obj_id == 2:
                ball_centroids[frame_idx] = centroid

    # Clean up
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)

    # Log tracking stats
    rim_ok = sum(1 for c in rim_centroids if c is not None)
    ball_ok = sum(1 for c in ball_centroids if c is not None)
    print(f"  Tracking complete: rim={rim_ok}/{frame_count}, ball={ball_ok}/{frame_count}")

    return {
        "rim_centroids": rim_centroids,
        "ball_centroids": ball_centroids,
        "frame_count": frame_count,
        "fps": fps,
    }


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Compute the centroid (cx, cy) of a boolean mask. Returns None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _mask_x_extent(mask: np.ndarray) -> tuple[float, float] | None:
    """Get the horizontal extent (x_min, x_max) of a boolean mask. Returns None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(xs.max())


# ---------------------------------------------------------------------------
# Part B (SAM 3): Text-prompted video tracking
# ---------------------------------------------------------------------------

def _extract_frame_range_to_dir(
    video_path: str, start_frame: int, end_frame: int
) -> tuple[str, int]:
    """Extract a range of frames [start_frame, end_frame) from a video to JPEGs.

    Returns (dir_path, frame_count).
    """
    n_frames = end_frame - start_frame
    tmpdir = tempfile.mkdtemp(prefix="sam3_frames_")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='between(n\\,{start_frame}\\,{end_frame - 1})',setpts=N/FRAME_RATE/TB",
        "-q:v", "2",
        "-start_number", "0",
        "-vsync", "vfr",
        f"{tmpdir}/%06d.jpg",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame range extraction failed: {proc.stderr[-300:]}")

    actual = len(list(Path(tmpdir).glob("*.jpg")))
    print(f"  Extracted frames {start_frame}-{end_frame - 1} ({actual} frames) to {tmpdir}")
    return tmpdir, actual


def track_objects_sam3(frames_dir: str, frame_count: int) -> dict:
    """Track basketball and rim using SAM 3 on a pre-extracted frame directory.

    Returns dict with:
        "rim_centroids": list of (cx, cy) or None per frame
        "ball_centroids": list of (cx, cy) or None per frame
        "frame_count": int
    """
    from sam3.model_builder import build_sam3_video_predictor

    print("Building SAM 3 video predictor ...")
    predictor = build_sam3_video_predictor()

    # Prompt on first frame (frame 0 of the window — ball should be above rim here)
    init_frame_idx = 0

    # --- Track the rim ---
    print("Tracking 'orange basketball hoop rim' ...")
    response = predictor.handle_request(dict(
        type="start_session",
        resource_path=frames_dir,
    ))
    rim_session_id = response["session_id"]

    predictor.handle_request(dict(
        type="add_prompt",
        session_id=rim_session_id,
        frame_index=init_frame_idx,
        text="orange basketball hoop rim, not the net",
    ))

    rim_centroids = [None] * frame_count
    rim_x_extents = [None] * frame_count
    for frame_response in predictor.handle_stream_request(dict(
        type="propagate_in_video",
        session_id=rim_session_id,
        propagation_direction="forward",
    )):
        fidx = frame_response["frame_index"]
        outputs = frame_response["outputs"]
        masks = outputs["out_binary_masks"]
        if masks is not None and len(masks) > 0:
            rim_centroids[fidx] = _mask_centroid(masks[0])
            rim_x_extents[fidx] = _mask_x_extent(masks[0])

    predictor.handle_request(dict(type="close_session", session_id=rim_session_id))

    # --- Track the ball ---
    print("Tracking 'basketball' ...")
    response = predictor.handle_request(dict(
        type="start_session",
        resource_path=frames_dir,
    ))
    ball_session_id = response["session_id"]

    predictor.handle_request(dict(
        type="add_prompt",
        session_id=ball_session_id,
        frame_index=init_frame_idx,
        text="basketball",
    ))

    ball_centroids = [None] * frame_count
    for frame_response in predictor.handle_stream_request(dict(
        type="propagate_in_video",
        session_id=ball_session_id,
        propagation_direction="forward",
    )):
        fidx = frame_response["frame_index"]
        outputs = frame_response["outputs"]
        masks = outputs["out_binary_masks"]
        if masks is not None and len(masks) > 0:
            ball_centroids[fidx] = _mask_centroid(masks[0])

    predictor.handle_request(dict(type="close_session", session_id=ball_session_id))

    rim_ok = sum(1 for c in rim_centroids if c is not None)
    ball_ok = sum(1 for c in ball_centroids if c is not None)
    print(f"  Tracking complete: rim={rim_ok}/{frame_count}, ball={ball_ok}/{frame_count}")

    return {
        "rim_centroids": rim_centroids,
        "rim_x_extents": rim_x_extents,
        "ball_centroids": ball_centroids,
        "frame_count": frame_count,
    }


# ---------------------------------------------------------------------------
# Part B3: Find crossing frame
# ---------------------------------------------------------------------------

def find_crossing_frame(
    ball_centroids: list, rim_centroids: list, rim_x_extents: list = None
) -> float:
    """Find the last frame where ball centroid crosses through the rim segment.

    A crossing requires:
    1. Ball centroid y transitions from above to below rim centroid y
    2. Ball centroid x is within the horizontal extent of the rim mask

    If rim_x_extents is not provided, falls back to y-only crossing check.

    Returns interpolated frame index (float) for sub-frame precision.
    """
    crossings = []

    for i in range(len(ball_centroids) - 1):
        bc = ball_centroids[i]
        bc_next = ball_centroids[i + 1]
        rc = rim_centroids[i]
        rc_next = rim_centroids[i + 1]

        if bc is None or bc_next is None or rc is None or rc_next is None:
            continue

        ball_cy = bc[1]
        ball_cy_next = bc_next[1]
        rim_cy = rc[1]
        rim_cy_next = rc_next[1]

        # Ball above rim in frame i, below rim in frame i+1
        diff_before = ball_cy - rim_cy      # negative = above
        diff_after = ball_cy_next - rim_cy_next  # positive = below

        if diff_before < 0 and diff_after > 0:
            # Check if ball x is within the rim's horizontal span
            if rim_x_extents is not None:
                rx = rim_x_extents[i]
                rx_next = rim_x_extents[i + 1]
                if rx is None or rx_next is None:
                    continue
                # Use the union of both frames' rim extents for tolerance
                rim_xmin = min(rx[0], rx_next[0])
                rim_xmax = max(rx[1], rx_next[1])
                ball_cx = bc[0]
                ball_cx_next = bc_next[0]
                if not (rim_xmin <= ball_cx <= rim_xmax or rim_xmin <= ball_cx_next <= rim_xmax):
                    print(f"  Rejected crossing frame {i}->{i+1}: "
                          f"ball_cx={ball_cx:.0f},{ball_cx_next:.0f} outside rim [{rim_xmin:.0f},{rim_xmax:.0f}]")
                    continue

            frac = (-diff_before) / (diff_after - diff_before)
            frame_hoop = i + frac
            crossings.append((frame_hoop, i, ball_cy, rim_cy, ball_cy_next, rim_cy_next))

    if not crossings:
        raise RuntimeError("No ball-through-rim crossing found in broadcast video")

    # Log all crossings, use the last one
    for j, (fh, i, bcy, rcy, bcy_n, rcy_n) in enumerate(crossings):
        tag = " <<<" if j == len(crossings) - 1 else ""
        print(f"  Crossing {j}: frame {i}->{i+1}, "
              f"ball_cy={bcy:.0f}->{bcy_n:.0f}, rim_cy={rcy:.0f}->{rcy_n:.0f}, "
              f"interp={fh:.2f}{tag}")

    best = crossings[-1]
    print(f"Broadcast hoop entry detected (last of {len(crossings)} crossings):")
    print(f"  Frame {best[1]}: ball_cy={best[2]:.1f}, rim_cy={best[3]:.1f} (above)")
    print(f"  Frame {best[1]+1}: ball_cy={best[4]:.1f}, rim_cy={best[5]:.1f} (below)")
    print(f"  Interpolated frame: {best[0]:.2f}")
    return best[0]


# ---------------------------------------------------------------------------
# Part C: Alignment
# ---------------------------------------------------------------------------

def auto_align(
    video_path: str,
    game_json_path: str,
    period: int,
    gc_start: float,
    gc_end: float,
    search_window_sec: float = 1.0,
) -> dict:
    """Orchestrate automatic alignment.

    1. Find hoop entry game clock from SportVU data
    2. OCR one clock transition to get rough alignment
    3. Use rough alignment to find the ~frame of the hoop entry
    4. Extract a small window (~1s) of frames around that point
    5. Run SAM 3 on just that window to find the exact crossing frame
    6. Compute final alignment

    Returns dict with alignment parameters.
    """
    from google import genai

    broadcast_fps = _probe_fps(video_path)

    # Part A: SportVU
    print("=" * 60)
    print("Part A: SportVU ball-through-hoop detection")
    print("=" * 60)
    game = load_game(game_json_path)
    gc_hoop = find_hoop_entry_gc(game, period, gc_start, gc_end)

    # Part B1: Rough alignment via one OCR clock transition
    print()
    print("=" * 60)
    print("Part B1: Rough alignment via OCR clock transition")
    print("=" * 60)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    from src.sync_video import find_clock_transition
    marker_rough = find_clock_transition(video_path, client)
    _, intercept_rough = compute_alignment(marker_rough, broadcast_fps)

    # Estimate the broadcast frame of the hoop entry using rough alignment
    # game_clock = -1.0 * video_time + intercept  =>  video_time = intercept - game_clock
    video_time_est = intercept_rough - gc_hoop
    frame_est = int(round(video_time_est * broadcast_fps))
    total_frames = _probe_frame_count(video_path)
    print(f"  Estimated hoop frame: {frame_est} (video_time={video_time_est:.2f}s)")

    # Part B2: Extract a small window and run SAM 3
    print()
    print("=" * 60)
    print("Part B2: SAM 3 tracking on search window")
    print("=" * 60)
    window_frames = int(search_window_sec * broadcast_fps)
    # Window: from 1s before estimated frame to the estimated frame
    win_start = max(0, frame_est - window_frames)
    win_end = min(frame_est + int(broadcast_fps * 0.2), total_frames)  # small buffer after
    print(f"  Search window: frames {win_start}-{win_end} ({win_end - win_start} frames)")

    frames_dir, frame_count = _extract_frame_range_to_dir(video_path, win_start, win_end)
    tracking = track_objects_sam3(frames_dir, frame_count)

    # Find crossing within the window
    crossing_in_window = find_crossing_frame(
        tracking["ball_centroids"], tracking["rim_centroids"], tracking["rim_x_extents"]
    )

    # Convert window-relative frame to original video frame
    frame_hoop = win_start + crossing_in_window

    # Clean up
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)

    # Part C: Compute final alignment
    print()
    print("=" * 60)
    print("Part C: Compute final alignment")
    print("=" * 60)
    video_time_hoop = frame_hoop / broadcast_fps
    intercept = gc_hoop + video_time_hoop
    slope = -1.0

    print(f"  gc_hoop (SportVU) = {gc_hoop:.3f}")
    print(f"  frame_hoop (broadcast) = {frame_hoop:.2f} (window frame {crossing_in_window:.2f})")
    print(f"  video_time_hoop = {video_time_hoop:.3f}s")
    print(f"  intercept = {intercept:.3f}")
    print(f"  rough intercept was = {intercept_rough:.3f} (diff = {intercept - intercept_rough:.3f}s)")
    print(f"  Alignment: game_clock = -1.0 * video_time + {intercept:.3f}")

    # Save
    marker = ClockMarker(
        frame_idx=int(round(frame_hoop)),
        clock_str=f"{int(gc_hoop // 60)}:{int(gc_hoop % 60):02d}",
        game_clock_seconds=gc_hoop,
    )
    alignment_path = "outputs/alignment.json"
    save_alignment(alignment_path, slope, intercept, marker, broadcast_fps)

    return {
        "slope": slope,
        "intercept": intercept,
        "gc_hoop": gc_hoop,
        "frame_hoop": frame_hoop,
        "video_fps": broadcast_fps,
        "marker": marker,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automatic video-to-tracking alignment via ball-through-hoop detection"
    )
    parser.add_argument(
        "--video", required=True, help="Path to broadcast video clip"
    )
    parser.add_argument(
        "--game", required=True, help="Path to SportVU game JSON (or game ID)"
    )
    parser.add_argument(
        "--period", type=int, required=True, help="Game period (quarter)"
    )
    parser.add_argument(
        "--gc-start", type=float, required=True,
        help="Game clock start (higher value, clock counts down)"
    )
    parser.add_argument(
        "--gc-end", type=float, required=True,
        help="Game clock end (lower value)"
    )
    parser.add_argument(
        "--search-window", type=float, default=1.0,
        help="Seconds of video to search around estimated hoop frame (default: 1)"
    )
    args = parser.parse_args()

    # Resolve game path
    game_path = args.game
    if not Path(game_path).exists():
        # Try data directory
        game_path = f"data/{args.game}.json"
        if not Path(game_path).exists():
            print(f"Error: game file not found: {args.game}")
            raise SystemExit(1)

    result = auto_align(
        video_path=args.video,
        game_json_path=game_path,
        period=args.period,
        gc_start=args.gc_start,
        gc_end=args.gc_end,
        search_window_sec=args.search_window,
    )

    print(f"\nDone. Alignment: game_clock = {result['slope']:.1f} * video_time + {result['intercept']:.3f}")


if __name__ == "__main__":
    main()
