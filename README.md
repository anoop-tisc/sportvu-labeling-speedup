# sportvu-labeling-speedup

Sync NBA broadcast video with SportVU tracking data to produce a combined visualization: broadcast footage on top, animated court diagram on the bottom, with play-by-play commentary overlaid.

## Quick start

```bash
# Install dependencies
uv pip install -r requirements.txt

# Simplest case — Gemini OCR alignment (~300-400ms accuracy):
python make_synced_video.py --game data/0021500492.json

# With a known offset:
python make_synced_video.py --game data/0021500492.json --alignment precise --offset 705822

# With auto-detection (OCR + SportVU hoop detection + SAM3):
python make_synced_video.py --game data/0021500492.json --alignment precise --offset auto
```

Output goes to `outputs/synced_clip.mp4` by default.

## How alignment works

The key problem is syncing broadcast video time with the SportVU game clock. We need one number: the **offset** — the game clock value (in ms) at video frame 0. Once you have it, mapping any video frame to tracking data is just:

```
game_clock = offset - video_time
```

There are three ways to get the offset:

### Simple alignment (default)

Sends frames to **Gemini Flash** to OCR the broadcast clock overlay. Finds a frame where the clock ticks over (e.g. 11:41 to 11:40), which gives us the game clock at a known frame. One API call.

This is ~300-400ms off because the TV graphics pipeline doesn't update the clock overlay in perfect sync with the arena clock that SportVU uses.

```bash
python make_synced_video.py --game data/0021500492.json
```

### Precise alignment with known offset

If you already know the offset (e.g. from the manual alignment tool in `tools/align_tool.html`), pass it directly in milliseconds:

```bash
python make_synced_video.py --game data/0021500492.json --alignment precise --offset 705822
```

### Precise alignment with auto-detection

Automatically computes the exact offset by finding the same made-basket moment in both data sources:

1. **Gemini OCR** — same as simple alignment, gives a rough offset (one API call)
2. **SportVU hoop detection** — scans the tracking data for the ball's z-coordinate crossing rim height (10 ft) near a basket. Finds the exact game clock of the made shot. Pure math, instant.
3. **Frame estimation** — uses the rough offset to estimate which broadcast frame the hoop entry is in
4. **SAM3 refinement** — extracts a ~1-second window of frames around that estimate, sends to SAM3 to track ball and rim centroids frame-by-frame, finds the exact frame where the ball crosses the rim
5. **Final offset** — combines the SportVU game clock with the SAM3 video frame to get sub-frame alignment

```bash
python make_synced_video.py --game data/0021500492.json --alignment precise --offset auto
```

## CLI reference

```
python make_synced_video.py --help

  --game GAME           Path to SportVU game JSON (required)
  --video VIDEO         Path to broadcast video clip (default: data/0021500492_event64.mp4)
  --period PERIOD       Game period / quarter (default: 1)
  --output OUTPUT       Output video path (default: outputs/synced_clip.mp4)
  --3d                  Render court in 3D (shows ball height)
  --alignment {simple,precise}
                        simple = Gemini OCR only; precise = exact offset (default: simple)
  --offset OFFSET       For --alignment precise: offset in ms (int) or 'auto'
```

## Project structure

```
make_synced_video.py          # Main entry point — run this to make a synced video
src/
  sync_video.py               # Core rendering + OCR alignment logic
  auto_align.py               # Automatic alignment via hoop detection (SportVU + SAM3)
  sportvu_loader.py           # Load and query SportVU tracking JSON
  visualize.py                # Court drawing (2D and 3D)
  pbp.py                      # Play-by-play event fetching
  court.py                    # Court geometry, zones, landmarks
  config.py                   # Constants (court dimensions, basket positions)
  possession.py               # Possession / ball-handler detection
  agent.py                    # LLM agent for play analysis
  chat_server.py              # Chat test UI server
  tracking_tools.py           # Agent tool definitions for tracking queries
  cli.py                      # CLI utilities
tools/
  align_tool.html             # Manual alignment web UI
  court_landmarks_viz.py      # Court landmarks visualization
  court_zones_viz.py          # Court zones visualization
  viz_tracking.py             # Tracking data visualization
  run_agent_test.py           # Agent test runner
data/
  0021500492.json             # SportVU game data (CHA @ TOR, 2016-01-01)
  0021500492_event64.mp4      # Broadcast clip
  pbp_0021500492.json         # Cached play-by-play
```

## Environment

Requires a `.env` file with:

```
GEMINI_API_KEY=...    # For OCR alignment (simple + precise auto)
ANTHROPIC_API_KEY=... # For the LLM agent (optional)
```
