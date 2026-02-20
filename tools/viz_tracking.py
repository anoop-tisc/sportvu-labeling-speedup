"""Visualize SAM 3 tracking results overlaid on the broadcast video.

Renders an annotated video with mask overlays, centroids, rim reference
line, and crossing frame markers. Also outputs a y-position chart.

Can run on the full clip or a narrow frame range.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from PIL import Image, ImageDraw

from src.auto_align import (
    _extract_frames_to_dir,
    _extract_frame_range_to_dir,
    _mask_centroid,
    _mask_x_extent,
    find_crossing_frame,
    find_hoop_entry_gc,
    track_objects_sam3,
)
from src.sync_video import (
    _probe_fps,
    _probe_frame_count,
    extract_frame_at,
    _ocr_single_frame,
    _clock_to_seconds,
    find_clock_transition,
    compute_alignment,
)
from src.sportvu_loader import load_game


def run_tracking_viz(
    video_path: str,
    game_path: str = "data/0021500492.json",
    period: int = 1,
    gc_start: float = 710,
    gc_end: float = 697,
    output_path: str = "outputs/tracking_viz.mp4",
):
    """Run SAM 3 tracking on a narrow window and render annotated video."""
    from google import genai
    from sam3.model_builder import build_sam3_video_predictor

    broadcast_fps = _probe_fps(video_path)
    total_frames = _probe_frame_count(video_path)

    # Part A: SportVU hoop time
    game = load_game(game_path)
    gc_hoop = find_hoop_entry_gc(game, period, gc_start, gc_end)

    # Part B1: Rough alignment via OCR
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    marker_rough = find_clock_transition(video_path, client)
    _, intercept_rough = compute_alignment(marker_rough, broadcast_fps)

    # Estimate hoop frame
    video_time_est = intercept_rough - gc_hoop
    frame_est = int(round(video_time_est * broadcast_fps))
    print(f"Estimated hoop frame: {frame_est}")

    # Extract narrow window (1s before to 0.2s after)
    win_start = max(0, frame_est - int(broadcast_fps))
    win_end = min(frame_est + int(broadcast_fps * 0.2), total_frames)
    print(f"Search window: frames {win_start}-{win_end} ({win_end - win_start} frames)")

    frames_dir, frame_count = _extract_frame_range_to_dir(video_path, win_start, win_end)
    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    w, h = Image.open(frame_files[0]).size

    # SAM 3 tracking
    print("Building SAM 3 video predictor ...")
    predictor = build_sam3_video_predictor()

    # Track rim
    print("Tracking 'orange basketball hoop rim' ...")
    resp = predictor.handle_request(dict(type="start_session", resource_path=frames_dir))
    sid = resp["session_id"]
    predictor.handle_request(dict(
        type="add_prompt", session_id=sid, frame_index=0,
        text="orange basketball hoop rim, not the net"))

    rim_centroids = [None] * frame_count
    rim_masks = [None] * frame_count
    rim_x_extents = [None] * frame_count
    for fr in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=sid, propagation_direction="forward")):
        fidx = fr["frame_index"]
        masks = fr["outputs"]["out_binary_masks"]
        if masks is not None and len(masks) > 0:
            rim_masks[fidx] = masks[0]
            rim_centroids[fidx] = _mask_centroid(masks[0])
            rim_x_extents[fidx] = _mask_x_extent(masks[0])
    predictor.handle_request(dict(type="close_session", session_id=sid))

    # Track ball
    print("Tracking 'basketball' ...")
    resp = predictor.handle_request(dict(type="start_session", resource_path=frames_dir))
    sid = resp["session_id"]
    predictor.handle_request(dict(
        type="add_prompt", session_id=sid, frame_index=0, text="basketball"))

    ball_centroids = [None] * frame_count
    ball_masks = [None] * frame_count
    for fr in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=sid, propagation_direction="forward")):
        fidx = fr["frame_index"]
        masks = fr["outputs"]["out_binary_masks"]
        if masks is not None and len(masks) > 0:
            ball_masks[fidx] = masks[0]
            ball_centroids[fidx] = _mask_centroid(masks[0])
    predictor.handle_request(dict(type="close_session", session_id=sid))

    rim_ok = sum(1 for c in rim_centroids if c is not None)
    ball_ok = sum(1 for c in ball_centroids if c is not None)
    print(f"  Tracking: rim={rim_ok}/{frame_count}, ball={ball_ok}/{frame_count}")

    # Find crossings
    crossings = _find_all_crossings(ball_centroids, rim_centroids)
    crossing_frames = set()
    for fh, i, *_ in crossings:
        crossing_frames.add(i)
        crossing_frames.add(i + 1)
    if crossings:
        best = crossings[-1]
        print(f"Best crossing: window frame {best[1]} (original frame {win_start + best[1]})")

    # Render annotated video
    print(f"Rendering {frame_count} annotated frames to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(broadcast_fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20",
        output_path,
    ]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for fidx in range(frame_count):
        img = Image.open(frame_files[fidx]).convert("RGB")
        frame = _annotate_frame(
            img, fidx, frame_count, w, h, win_start, broadcast_fps,
            rim_masks[fidx], ball_masks[fidx],
            rim_centroids[fidx], ball_centroids[fidx],
            rim_x_extents[fidx],
            crossing_frames, crossings,
        )
        pipe.stdin.write(np.array(frame).tobytes())

    pipe.stdin.close()
    pipe.wait()
    print(f"Saved: {output_path}")

    # Save chart
    _save_chart(ball_centroids, rim_centroids, crossings, frame_count, broadcast_fps, win_start)

    shutil.rmtree(frames_dir, ignore_errors=True)
    print("Done.")


def _annotate_frame(img, fidx, frame_count, w, h, win_start, fps,
                    rim_mask, ball_mask, rim_c, ball_c, rim_x_ext,
                    crossing_frames, crossings):
    """Draw mask overlays, centroids, rim line segment, and status bar on a frame."""
    overlay = np.array(img)

    if rim_mask is not None:
        overlay[rim_mask] = (overlay[rim_mask] * 0.5 + np.array([0, 150, 255]) * 0.5).astype(np.uint8)
    if ball_mask is not None:
        overlay[ball_mask] = (overlay[ball_mask] * 0.5 + np.array([255, 140, 0]) * 0.5).astype(np.uint8)

    img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(img)

    # Rim centroid + horizontal line spanning only the rim mask width
    if rim_c is not None:
        rcx, rcy = int(rim_c[0]), int(rim_c[1])
        if rim_x_ext is not None:
            x_min, x_max = int(rim_x_ext[0]), int(rim_x_ext[1])
        else:
            x_min, x_max = 0, w
        draw.line([(x_min, rcy), (x_max, rcy)], fill=(0, 150, 255), width=3)
        draw.ellipse([rcx - 7, rcy - 7, rcx + 7, rcy + 7], fill=(0, 100, 255), outline="white", width=2)
        draw.text((rcx + 12, rcy - 8), "HOOP", fill=(0, 150, 255))

    # Ball centroid
    if ball_c is not None:
        bcx, bcy = int(ball_c[0]), int(ball_c[1])
        draw.ellipse([bcx - 7, bcy - 7, bcx + 7, bcy + 7], fill=(255, 140, 0), outline="white", width=2)
        draw.text((bcx + 12, bcy - 8), "BALL", fill=(255, 140, 0))

    # Crossing border
    is_crossing = fidx in crossing_frames
    if is_crossing:
        draw.rectangle([0, 0, w - 1, h - 1], outline="red", width=6)

    # Status bar
    orig_frame = win_start + fidx
    t = orig_frame / fps
    parts = [f"Win {fidx}/{frame_count} (orig {orig_frame})", f"t={t:.2f}s"]
    if ball_c is not None:
        parts.append(f"ball_y={ball_c[1]:.0f}")
    else:
        parts.append("ball=LOST")
    if rim_c is not None:
        parts.append(f"rim_y={rim_c[1]:.0f}")
    else:
        parts.append("rim=LOST")
    if ball_c is not None and rim_c is not None:
        diff = ball_c[1] - rim_c[1]
        parts.append(f"{'ABOVE' if diff < 0 else 'BELOW'}({abs(diff):.0f}px)")

    for ci, (fh, i, *_) in enumerate(crossings):
        if fidx in (i, i + 1):
            parts.append(f"CROSSING #{ci}")

    draw.rectangle([0, 0, w, 32], fill=(0, 0, 0, 200))
    draw.text((10, 7), "  |  ".join(parts), fill="white")

    return img


def _find_all_crossings(ball_centroids, rim_centroids):
    crossings = []
    for i in range(len(ball_centroids) - 1):
        bc, bc_next = ball_centroids[i], ball_centroids[i + 1]
        rc, rc_next = rim_centroids[i], rim_centroids[i + 1]
        if bc is None or bc_next is None or rc is None or rc_next is None:
            continue
        diff_before = bc[1] - rc[1]
        diff_after = bc_next[1] - rc_next[1]
        if diff_before < 0 and diff_after > 0:
            frac = (-diff_before) / (diff_after - diff_before)
            crossings.append((i + frac, i, bc[1], rc[1], bc_next[1], rc_next[1]))
    return crossings


def _save_chart(ball_centroids, rim_centroids, crossings, frame_count, fps, win_start):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = list(range(frame_count))
    times = [(win_start + f) / fps for f in frames]
    ball_ys = [bc[1] if bc else np.nan for bc in ball_centroids]
    rim_ys = [rc[1] if rc else np.nan for rc in rim_centroids]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, ball_ys, color="orange", linewidth=1.5, label="Ball y", alpha=0.9)
    ax.plot(times, rim_ys, color="blue", linewidth=1.5, label="Rim y", alpha=0.9)

    for ci, (fh, i, *_) in enumerate(crossings):
        t = (win_start + fh) / fps
        ax.axvline(t, color="red", linestyle="--", linewidth=2, alpha=0.8, label=f"Crossing #{ci}")

    ax.set_xlabel("Video time (seconds)")
    ax.set_ylabel("Y position (pixels, 0=top)")
    ax.set_title("Ball & Rim Y-Position — SAM 3 tracking (narrow window)")
    ax.legend(loc="upper right", fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/tracking_chart.png", dpi=150)
    plt.close(fig)
    print("  Saved outputs/tracking_chart.png")


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "tools/broadcast_clip2.mp4"
    run_tracking_viz(video)
