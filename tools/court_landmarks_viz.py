"""Generate a labeled court landmark map showing all landmark points and lines."""

import math

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
from pathlib import Path

from src.court import (
    LANDMARK_POINTS, LANDMARK_LINES, LANDMARK_POINT_RADIUS,
    THREE_POINT_ARC_RADIUS,
)
from src.visualize import draw_court


def _to_plot(x: float, y: float) -> tuple[float, float]:
    """Convert SportVU half-court coords (x 0..47, y 0..50) to draw_court coords (x 0..94, y -50..0)."""
    return x, -y


def generate_landmark_map(output_path: str = "outputs/court_landmarks.png"):
    """Render a court image with landmark points (red circles) and lines (red highlights)."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
    fig.patch.set_facecolor("#f5f5f5")
    ax.set_facecolor("#e8e8e8")

    draw_court(ax, color="black", lw=1.5)

    # --- Draw landmark lines ---
    def _draw_line(line, zorder=5):
        if line.kind == "segment":
            px1, py1 = _to_plot(line.x1, line.y1)
            px2, py2 = _to_plot(line.x2, line.y2)
            ax.plot([px1, px2], [py1, py2], color="red", lw=3, alpha=0.6, zorder=zorder)
        elif line.kind == "arc":
            # matplotlib Arc expects center, width, height, angle range in degrees
            pcx, pcy = _to_plot(line.cx, line.cy)
            # In plot coords, y is negated, so angles are mirrored
            # angle_start and angle_end are in math convention (CCW from +x)
            # With y negated, angles become CW, so swap and negate
            deg_start = -math.degrees(line.angle_end)
            deg_end = -math.degrees(line.angle_start)
            arc = Arc((pcx, pcy), 2 * line.radius, 2 * line.radius,
                      angle=0, theta1=deg_start, theta2=deg_end,
                      color="red", lw=3, alpha=0.6, zorder=zorder)
            ax.add_patch(arc)
        elif line.kind == "composite":
            for part in line.parts:
                _draw_line(part, zorder=zorder)

    for ll in LANDMARK_LINES:
        _draw_line(ll)

    # --- Draw landmark points ---
    for lp in LANDMARK_POINTS:
        px, py = _to_plot(lp.x, lp.y)
        circle = Circle((px, py), LANDMARK_POINT_RADIUS, color="red", alpha=0.35, zorder=6)
        ax.add_patch(circle)
        ax.plot(px, py, "o", color="red", markersize=4, zorder=7)

        # Label
        # Offset label slightly to avoid overlap with the point
        offset_y = LANDMARK_POINT_RADIUS + 1.5
        label_y = py - offset_y  # go below (more negative in plot coords)
        # For points near the bottom of the court (low py), put label above
        if py < -40:
            label_y = py + offset_y + 1

        ax.text(px, label_y, lp.name, ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="white", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#c0392b",
                          edgecolor="white", alpha=0.85, lw=0.5))

    # --- Line labels ---
    line_label_positions = {
        "baseline": (0, -25),
        "left sideline": (23.5, 0),
        "right sideline": (23.5, -50),
        "left lane line": (9.5, -17),
        "right lane line": (9.5, -33),
        "free throw line": (19, -25),
        "three-point line": (29, -12),
        "half-court line": (47, -25),
        "restricted area": (5.25, -21),
    }
    for ll in LANDMARK_LINES:
        if ll.name in line_label_positions:
            lx, ly = line_label_positions[ll.name]
            ax.text(lx, ly, ll.name, ha="center", va="center",
                    fontsize=6, fontweight="bold", color="white", zorder=10,
                    fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#2980b9",
                              edgecolor="white", alpha=0.85, lw=0.5))

    ax.set_xlim(-3, 97)
    ax.set_ylim(-53, 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("NBA Court Landmarks Map", fontsize=16,
                 fontweight="bold", pad=15, color="#333")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="red", alpha=0.35, edgecolor="red", label="Landmark point"),
        Line2D([0], [0], color="red", lw=3, alpha=0.6, label="Landmark line"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=9, frameon=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Landmark map saved to {output_path}")


if __name__ == "__main__":
    generate_landmark_map()
