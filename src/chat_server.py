"""Flask chat UI for testing the basketball agent interactively."""

import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from flask import Flask, request, jsonify, send_from_directory
from src.sportvu_loader import load_game
from src.pbp import fetch_pbp
from src.agent import chat_agent


def _load_alignment(path: str) -> tuple:
    """Load cached alignment. Returns (slope, intercept, video_fps)."""
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded cached alignment from {path}")
    print(f"  slope={data['slope']:.4f}, intercept={data['intercept']:.2f}")
    return data["slope"], data["intercept"], data["video_fps"]


def _probe_frame_count(video_path: str) -> int:
    """Get total number of video frames via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return int(out)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GAME_PATH = "data/0021500492.json"
GAME_ID = "0021500492"
PERIOD = 1
TEAM = "TOR"
VIDEO_PATH = "data/clip_q4.mp4"
ALIGNMENT_PATH = "outputs/alignment.json"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

# ---------------------------------------------------------------------------
# One-time initialization
# ---------------------------------------------------------------------------
print("Loading alignment...")
slope, intercept, video_fps = _load_alignment(ALIGNMENT_PATH)
n_frames = _probe_frame_count(VIDEO_PATH)
duration = n_frames / video_fps
gc_start = intercept
gc_end = slope * duration + intercept
print(f"Clip time range: gc {gc_start:.2f} → {gc_end:.2f}")

print(f"Loading game data from {GAME_PATH}...")
game = load_game(GAME_PATH)

print(f"Fetching play-by-play for {GAME_ID}...")
pbp_events = fetch_pbp(GAME_ID)
print(f"Loaded {len(pbp_events)} PBP events")

# In-memory conversation state (single-session test tool)
conversation_messages: list = []

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Basketball Agent Chat</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       height: 100vh; display: flex; background: #1a1a2e; color: #eee; }
#left { width: 50%; display: flex; flex-direction: column; padding: 16px;
        border-right: 1px solid #333; }
#right { width: 50%; display: flex; flex-direction: column; padding: 16px; }
h2 { margin-bottom: 12px; font-size: 14px; color: #888; text-transform: uppercase;
     letter-spacing: 1px; }
video { width: 100%; max-height: 60vh; border-radius: 8px; background: #000; object-fit: contain; }
#info { margin-top: 12px; font-size: 13px; color: #888; }
#messages { flex: 1; overflow-y: auto; padding: 8px 0; display: flex;
            flex-direction: column; gap: 12px; }
.msg { max-width: 90%; padding: 10px 14px; border-radius: 12px; font-size: 14px;
       line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
.msg.user { align-self: flex-end; background: #0d47a1; color: #fff; }
.msg.agent { align-self: flex-start; background: #2d2d44; color: #e0e0e0; }
.msg.error { align-self: flex-start; background: #5a1a1a; color: #faa; }
#spinner { display: none; align-self: flex-start; padding: 10px 14px; font-size: 13px;
           color: #888; }
#spinner.active { display: block; }
#input-row { display: flex; gap: 8px; margin-top: 12px; }
#input-row input { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #444;
                   background: #2d2d44; color: #eee; font-size: 14px; outline: none; }
#input-row input:focus { border-color: #0d47a1; }
#input-row button { padding: 10px 20px; border-radius: 8px; border: none;
                    background: #0d47a1; color: #fff; font-size: 14px; cursor: pointer; }
#input-row button:hover { background: #1565c0; }
#input-row button:disabled { opacity: 0.5; cursor: not-allowed; }
#reset-btn { margin-top: 8px; padding: 6px 12px; border-radius: 6px; border: 1px solid #555;
             background: transparent; color: #888; font-size: 12px; cursor: pointer;
             align-self: flex-end; }
#reset-btn:hover { border-color: #c00; color: #c00; }
</style>
</head>
<body>

<div id="left">
  <h2>Synced Video</h2>
  <video controls preload="metadata">
    <source src="/video/synced_clip2.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <div id="info">
    Game: CHA @ TOR &mdash; 2016-01-01<br>
    Period 1 &mdash; Tracking data synced to broadcast video
  </div>
</div>

<div id="right">
  <h2>Agent Chat</h2>
  <div id="messages"></div>
  <div id="spinner">Thinking&hellip;</div>
  <div id="input-row">
    <input id="user-input" type="text" placeholder="Ask about the play…"
           autofocus autocomplete="off">
    <button id="send-btn" onclick="sendMessage()">Send</button>
  </div>
  <button id="reset-btn" onclick="resetChat()">Reset conversation</button>
</div>

<script>
const messagesDiv = document.getElementById('messages');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const spinner = document.getElementById('spinner');

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !sendBtn.disabled) sendMessage();
});

function addMessage(text, cls) {
  const el = document.createElement('div');
  el.className = 'msg ' + cls;
  el.textContent = text;
  messagesDiv.appendChild(el);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMessage(text, 'user');
  sendBtn.disabled = true;
  spinner.classList.add('active');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });
    const data = await res.json();
    if (data.error) {
      addMessage('Error: ' + data.error, 'error');
    } else {
      addMessage(data.reply, 'agent');
    }
  } catch (err) {
    addMessage('Network error: ' + err.message, 'error');
  } finally {
    sendBtn.disabled = false;
    spinner.classList.remove('active');
    input.focus();
  }
}

async function resetChat() {
  await fetch('/api/reset', {method: 'POST'});
  messagesDiv.innerHTML = '';
  input.focus();
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/video/<path:filename>")
def serve_video(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    global conversation_messages
    data = request.get_json(force=True)
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    try:
        reply, conversation_messages = chat_agent(
            user_msg, conversation_messages,
            game=game, pbp_events=pbp_events,
            period=PERIOD, gc_start=gc_start, gc_end=gc_end,
            team_abbr=TEAM, verbose=True,
        )
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global conversation_messages
    conversation_messages = []
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
