"""Constants for court geometry, detection thresholds, and API settings."""

# Court dimensions (feet)
COURT_LENGTH = 94
COURT_WIDTH = 50

# Basket centers
LEFT_BASKET = (5.25, 25.0)
RIGHT_BASKET = (88.75, 25.0)

# Three-point line
THREE_POINT_ARC_RADIUS = 23.75  # ft from basket center
THREE_POINT_CORNER_DIST = 22.0  # ft from basket in corners
THREE_POINT_CORNER_Y_MAX = 11.0  # corner 3 zone boundary from sideline

# Paint / key
PAINT_WIDTH = 16  # ft (y: 17-33)
PAINT_Y_MIN = 17.0
PAINT_Y_MAX = 33.0
PAINT_DEPTH = 19.0  # ft from baseline to free-throw line

# Restricted area
RESTRICTED_AREA_RADIUS = 4.0  # ft from basket center

# Ball handler detection thresholds
BALL_CONTROL_ENTRY_RADIUS = 2.5  # ft — acquire control
BALL_CONTROL_EXIT_RADIUS = 4.0   # ft — lose control (hysteresis)
MIN_HANDLER_FRAMES = 5           # 0.2s at 25 FPS
DRIBBLE_HEIGHT_MAX = 7.0         # ft — ball above this is airborne/shot

# Pass detection
PASS_SPEED_THRESHOLD = 15.0  # ft/s
LOOSE_BALL_RADIUS = 3.5      # ft from all players
REGAIN_WINDOW = 0.3          # seconds — same player regain = dribble, not pass

# Matchup classification distances (ft)
MATCHUP_TIGHT = 4.0
MATCHUP_MODERATE = 8.0
MATCHUP_LOOSE = 15.0

# Tracking data FPS
FPS = 25

# Default data directory
DATA_DIR = "data"

# Claude model for agent
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_AGENT_TURNS = 20
