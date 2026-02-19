"""Claude tool_use agent for answering basketball questions from tracking data."""

import json
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from src.config import CLAUDE_MODEL, MAX_AGENT_TURNS

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from src.sportvu_loader import GameData
from src.court import detect_attacking_basket
from src.tracking_tools import (
    get_player_positions,
    get_ball_handler,
    detect_passes,
    get_defensive_matchups,
    get_player_trajectory,
    get_ball_trajectory,
    get_play_by_play,
    get_possession_summary,
)


TOOL_SCHEMAS = [
    {
        "name": "get_player_positions",
        "description": (
            "Get the positions of all 10 players and the ball at a specific game clock time. "
            "Returns each player's position relative to court landmarks (e.g., 'at the left elbow', "
            "'on the three-point line', 'near the right wing'), distance to basket, team, jersey number, "
            "and raw (x,y) coordinates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "game_clock": {
                    "type": "number",
                    "description": "Game clock in seconds (e.g., 660.0 for 11:00). Clock counts down from 720 (12:00).",
                },
                "period": {
                    "type": "integer",
                    "description": "Period/quarter number (1-4).",
                },
            },
            "required": ["game_clock", "period"],
        },
    },
    {
        "name": "get_ball_handler",
        "description": (
            "Detect who is handling the ball over a time range. Uses proximity-based detection "
            "with hysteresis (2.5ft acquire, 4.0ft release). Returns a timeline of handlers "
            "with start/end times and gaps labeled as 'ball in flight' or 'loose ball'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gc_start": {
                    "type": "number",
                    "description": "Start of range in game clock seconds (higher value, e.g., 660.0 for 11:00).",
                },
                "gc_end": {
                    "type": "number",
                    "description": "End of range in game clock seconds (lower value, e.g., 650.0 for 10:50).",
                },
                "period": {
                    "type": "integer",
                    "description": "Period/quarter number (1-4).",
                },
            },
            "required": ["gc_start", "gc_end", "period"],
        },
    },
    {
        "name": "detect_passes",
        "description": (
            "Detect passes in a time range. A pass is handler A losing the ball, "
            "a gap, then handler B gaining it. Returns passer, receiver, positions (described "
            "by court landmarks), distance, and flight time. Also detects turnovers (ball going to other team)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gc_start": {
                    "type": "number",
                    "description": "Start of range in game clock seconds (higher value).",
                },
                "gc_end": {
                    "type": "number",
                    "description": "End of range in game clock seconds (lower value).",
                },
                "period": {
                    "type": "integer",
                    "description": "Period/quarter number (1-4).",
                },
            },
            "required": ["gc_start", "gc_end", "period"],
        },
    },
    {
        "name": "get_defensive_matchups",
        "description": (
            "Get defensive matchups at a specific time. For each offensive player, finds the "
            "closest defender. Classifies coverage as tight (<4ft), moderate (4-8ft), "
            "loose/help (8-15ft), or unguarded (>15ft). Use window_seconds > 0 for "
            "majority-vote smoothing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "game_clock": {
                    "type": "number",
                    "description": "Game clock in seconds.",
                },
                "period": {
                    "type": "integer",
                    "description": "Period/quarter number (1-4).",
                },
                "window_seconds": {
                    "type": "number",
                    "description": "Window in seconds for majority-vote smoothing (0 for single frame). Default 0.",
                },
            },
            "required": ["game_clock", "period"],
        },
    },
    {
        "name": "get_player_trajectory",
        "description": (
            "Track a specific player's movement over a time range. Returns sampled positions "
            "at ~0.5s intervals with speed, court landmark, total distance, and position transitions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {
                    "type": "string",
                    "description": "Player name (full or partial, case-insensitive). E.g., 'DeRozan', 'Kyle Lowry'.",
                },
                "gc_start": {
                    "type": "number",
                    "description": "Start of range in game clock seconds (higher value).",
                },
                "gc_end": {
                    "type": "number",
                    "description": "End of range in game clock seconds (lower value).",
                },
                "period": {
                    "type": "integer",
                    "description": "Period/quarter number (1-4).",
                },
            },
            "required": ["player_name", "gc_start", "gc_end", "period"],
        },
    },
    {
        "name": "get_ball_trajectory",
        "description": (
            "Track the ball's movement over a time range. Returns positions with height (z), "
            "speed, and state (held/dribbled, in flight, shot arc)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gc_start": {
                    "type": "number",
                    "description": "Start of range in game clock seconds (higher value).",
                },
                "gc_end": {
                    "type": "number",
                    "description": "End of range in game clock seconds (lower value).",
                },
                "period": {
                    "type": "integer",
                    "description": "Period/quarter number (1-4).",
                },
            },
            "required": ["gc_start", "gc_end", "period"],
        },
    },
    {
        "name": "get_play_by_play",
        "description": (
            "Get official play-by-play events (from ESPN) for a time window. "
            "Returns event descriptions with timestamps. Useful for context about "
            "made/missed shots, fouls, turnovers, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gc_start": {
                    "type": "number",
                    "description": "Start of range in game clock seconds (higher value).",
                },
                "gc_end": {
                    "type": "number",
                    "description": "End of range in game clock seconds (lower value).",
                },
                "period": {
                    "type": "integer",
                    "description": "Period/quarter number (1-4).",
                },
            },
            "required": ["gc_start", "gc_end", "period"],
        },
    },
    {
        "name": "get_possession_summary",
        "description": (
            "Get a comprehensive summary of a possession. Returns ball handler timeline, "
            "passes, turnovers, PBP events, defensive matchups, and player positions "
            "at start/middle/end. Use this for a broad overview before drilling into specifics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gc_start": {
                    "type": "number",
                    "description": "Start of possession in game clock seconds (higher value).",
                },
                "gc_end": {
                    "type": "number",
                    "description": "End of possession in game clock seconds (lower value).",
                },
                "period": {
                    "type": "integer",
                    "description": "Period/quarter number (1-4).",
                },
            },
            "required": ["gc_start", "gc_end", "period"],
        },
    },
]


def build_system_prompt(game: GameData, period: int, gc_start: float, gc_end: float,
                        team_abbr: str) -> str:
    """Build the system prompt for the agent."""
    return f"""You are a basketball analytics agent with access to NBA SportVU player tracking data \
(25 FPS x,y,z coordinates for the ball and all 10 players on the court).

Game: {game.game_id} — {game.visitor_team['abbreviation']} @ {game.home_team['abbreviation']}, {game.game_date}
Home: {game.home_team['name']} ({game.home_team['abbreviation']})
Away: {game.visitor_team['name']} ({game.visitor_team['abbreviation']})

Current possession: Period {period}, {_fmt_clock(gc_start)} to {_fmt_clock(gc_end)}, {team_abbr} has the ball.

Use your tools to query exact tracking data. Do NOT guess positions or handlers — always call a tool first.

Report positions using court landmark descriptions (e.g., "at the left elbow", "on the three-point line", \
"near the right wing", "at the basket", "on the free throw line", etc.).

Left/right convention: from the OFFENSIVE player's perspective facing the basket.
- "Left wing" = to the player's left when facing the basket.
- "Right corner" = to the player's right when facing the basket.

Game clock: counts DOWN from 720.0 (12:00) to 0.0 per quarter. gc_start > gc_end.
To convert: 11:00 = 660.0s, 5:30 = 330.0s, 1:15 = 75.0s.

You CAN answer from tracking data:
- Player positions relative to court landmarks at any moment
- Ball handler identity and timeline
- Ball movement path and passes (passer, receiver, positions)
- Defensive matchups and coverage tightness
- Player trajectories, speed, and distance covered
- Shot location (from ball trajectory arc)
- Timing of events

You CANNOT determine from tracking data alone:
- Pass type (bounce pass vs chest pass vs lob)
- Player body orientation or facing direction
- Screen contact or screen quality
- Pump fakes or shot fakes
- Verbal communication
For those, say "Cannot determine from tracking data alone."

When answering:
1. Start with get_possession_summary for broad context, or use specific tools for targeted questions.
2. Be precise: cite game clock times, landmark names, and distances.
3. If data seems inconsistent, query multiple time points to confirm.
4. Keep answers concise and factual."""


def _fmt_clock(gc: float) -> str:
    mins = int(gc) // 60
    secs = gc - mins * 60
    return f"{mins}:{secs:04.1f}"


def execute_tool(tool_name: str, tool_input: dict, game: GameData,
                 pbp_events: list[dict], attacking_basket: dict) -> str:
    """Execute a tool call and return the result as a JSON string."""
    try:
        if tool_name == "get_player_positions":
            result = get_player_positions(
                game, tool_input["period"], tool_input["game_clock"],
                attacking_basket=attacking_basket,
            )
        elif tool_name == "get_ball_handler":
            result = get_ball_handler(
                game, tool_input["period"], tool_input["gc_start"], tool_input["gc_end"],
            )
        elif tool_name == "detect_passes":
            result = detect_passes(
                game, tool_input["period"], tool_input["gc_start"], tool_input["gc_end"],
                attacking_basket=attacking_basket,
            )
        elif tool_name == "get_defensive_matchups":
            result = get_defensive_matchups(
                game, tool_input["period"], tool_input["game_clock"],
                window_seconds=tool_input.get("window_seconds", 0),
                attacking_basket=attacking_basket,
            )
        elif tool_name == "get_player_trajectory":
            result = get_player_trajectory(
                game, tool_input["player_name"], tool_input["period"],
                tool_input["gc_start"], tool_input["gc_end"],
                attacking_basket=attacking_basket,
            )
        elif tool_name == "get_ball_trajectory":
            result = get_ball_trajectory(
                game, tool_input["period"], tool_input["gc_start"], tool_input["gc_end"],
            )
        elif tool_name == "get_play_by_play":
            result = get_play_by_play(
                game, pbp_events, tool_input["period"],
                tool_input["gc_start"], tool_input["gc_end"],
            )
        elif tool_name == "get_possession_summary":
            result = get_possession_summary(
                game, pbp_events, tool_input["period"],
                tool_input["gc_start"], tool_input["gc_end"],
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        result = {"error": f"Tool execution error: {str(e)}"}

    return json.dumps(result, indent=2)


def run_agent(question: str, game: GameData, pbp_events: list[dict],
              period: int, gc_start: float, gc_end: float, team_abbr: str,
              verbose: bool = False) -> str:
    """Run the agent loop to answer a question about a possession.

    Returns the agent's final text answer.
    """
    client = Anthropic()
    attacking_basket = detect_attacking_basket(game, period)

    system_prompt = build_system_prompt(game, period, gc_start, gc_end, team_abbr)
    messages = [{"role": "user", "content": question}]

    for turn in range(MAX_AGENT_TURNS):
        if verbose:
            print(f"  [Agent turn {turn + 1}]")

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # Check for tool use
        if response.stop_reason == "tool_use":
            # Process all tool calls
            assistant_content = response.content
            tool_results = []

            for block in assistant_content:
                if block.type == "tool_use":
                    if verbose:
                        print(f"    → {block.name}({json.dumps(block.input)})")
                    result = execute_tool(
                        block.name, block.input, game, pbp_events, attacking_basket,
                    )
                    if verbose:
                        # Print truncated result
                        preview = result[:200] + "..." if len(result) > 200 else result
                        print(f"    ← {preview}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Final text response
            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            return "\n".join(text_parts)

    return "[Agent reached maximum turns without a final answer]"
