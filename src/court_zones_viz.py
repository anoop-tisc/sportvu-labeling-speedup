"""Generate a labeled court zone map showing all classification zones."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, to_rgba
from pathlib import Path

from src.court import classify_zone
from src.visualize import draw_court


# Zone name -> display color
ZONE_COLORS = {
    "restricted area":     "#e74c3c",
    "left low post":       "#c0392b",
    "right low post":      "#c0392b",
    "left high post":      "#e67e22",
    "right high post":     "#e67e22",
    "left elbow":          "#f39c12",
    "right elbow":         "#f39c12",
    "left short corner":   "#e6b422",
    "right short corner":  "#e6b422",
    "left mid-range":      "#2ecc71",
    "right mid-range":     "#2ecc71",
    "top of the key":      "#27ae60",
    "left corner":         "#3498db",
    "right corner":        "#3498db",
    "left wing":           "#2980b9",
    "right wing":          "#2980b9",
    "top of the arc":      "#8e44ad",
    "backcourt":           "#95a5a6",
}

# Short labels for the map
ZONE_LABELS = {
    "restricted area":     "Restricted\nArea",
    "left low post":       "Left\nLow Post",
    "right low post":      "Right\nLow Post",
    "left high post":      "Left\nHigh Post",
    "right high post":     "Right\nHigh Post",
    "left elbow":          "Left\nElbow",
    "right elbow":         "Right\nElbow",
    "left short corner":   "Left Short\nCorner",
    "right short corner":  "Right Short\nCorner",
    "left mid-range":      "Left\nMid-Range",
    "right mid-range":     "Right\nMid-Range",
    "top of the key":      "Top of\nthe Key",
    "left corner":         "Left\nCorner",
    "right corner":        "Right\nCorner",
    "left wing":           "Left\nWing",
    "right wing":          "Right\nWing",
    "top of the arc":      "Top of\nthe Arc",
    "backcourt":           "Backcourt",
}


def generate_zone_map(output_path: str = "outputs/court_zones.png",
                      resolution: float = 0.5):
    """Render a court image with colored zones and labels.

    draw_court uses: x 0..94, y -50..0  (y is negated from SportVU's 0..50).
    classify_zone uses raw SportVU coords: x 0..47, y 0..50.
    """
    # Sample the half-court in classify_zone's coordinate system (x 0..47, y 0..50)
    xs = np.arange(0, 47.5, resolution)
    ys = np.arange(0, 50.5, resolution)

    zone_names = sorted(ZONE_COLORS.keys())
    zone_to_idx = {z: i for i, z in enumerate(zone_names)}

    grid = np.full((len(ys), len(xs)), -1, dtype=int)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            result = classify_zone(x, y)
            zname = result["zone"]
            if zname in zone_to_idx:
                grid[iy, ix] = zone_to_idx[zname]

    # Build colormap: index 0 = transparent, 1..N = zone colors
    cmap_colors = [(0, 0, 0, 0)]  # transparent for -1
    cmap_colors += [to_rgba(ZONE_COLORS[z], alpha=0.45) for z in zone_names]
    cmap = ListedColormap(cmap_colors)
    grid_shifted = grid + 1  # shift so -1 -> 0

    # --- Figure using draw_court's coordinate system ---
    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
    fig.patch.set_facecolor('#f5f5f5')
    ax.set_facecolor('#e8e8e8')

    draw_court(ax, color="black", lw=1.5)

    # imshow needs extent in plot coords.
    # draw_court: x 0..94, y -50..0.
    # classify_zone y=0 corresponds to draw_court y=0 (top sideline),
    # classify_zone y=50 corresponds to draw_court y=-50 (bottom sideline).
    # Grid row 0 = y=0 (top), row -1 = y=50 (bottom).
    # imshow displays row 0 at top, so extent bottom=-50, top=0.

    # Left half-court (x 0..47)
    ax.imshow(grid_shifted, extent=[0, 47, -50, 0], aspect='auto',
              cmap=cmap, vmin=0, vmax=len(cmap_colors) - 1,
              interpolation='nearest', zorder=1)

    # Right half-court: mirror x and y
    ax.imshow(grid_shifted[::-1, ::-1], extent=[47, 94, -50, 0], aspect='auto',
              cmap=cmap, vmin=0, vmax=len(cmap_colors) - 1,
              interpolation='nearest', zorder=1)

    ax.set_xlim(-3, 97)
    ax.set_ylim(-53, 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Labels at representative positions ---
    # Positions given in classify_zone coords (x, y with y 0..50).
    # Convert to plot coords: x stays, y_plot = -y.
    label_positions = {
        "restricted area":     (5.25, 25.0),
        "left low post":       (6.0, 20.0),
        "right low post":      (6.0, 30.0),
        "left high post":      (14.0, 20.0),
        "right high post":     (14.0, 30.0),
        "left elbow":          (19.5, 16.5),
        "right elbow":         (19.5, 33.5),
        "left short corner":   (8.0, 13.0),
        "right short corner":  (8.0, 37.0),
        "left mid-range":      (17.0, 10.0),
        "right mid-range":     (17.0, 40.0),
        "top of the key":      (25.0, 25.0),
        "left corner":         (5.0, 4.0),
        "right corner":        (5.0, 46.0),
        "left wing":           (30.0, 8.0),
        "right wing":          (30.0, 42.0),
        "top of the arc":      (35.0, 25.0),
        "backcourt":           (70.0, 25.0),
    }

    for zone_name, (lx, ly) in label_positions.items():
        label = ZONE_LABELS[zone_name]
        color = ZONE_COLORS[zone_name]
        y_plot = -ly

        # Left half: swap Left<->Right labels
        swapped_label = label.replace("Left", "TEMP").replace("Right", "Left").replace("TEMP", "Right")
        ax.text(lx, y_plot, swapped_label, ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="white", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                          edgecolor="white", alpha=0.85, lw=0.5))

        # Right half: keep original labels
        if zone_name != "backcourt":
            rx = 94 - lx
            ax.text(rx, y_plot, label, ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="white", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                              edgecolor="white", alpha=0.85, lw=0.5))

    ax.set_title("NBA Court Zone Classification Map", fontsize=16,
                 fontweight="bold", pad=15, color="#333")

    # Legend (deduplicated)
    seen = set()
    legend_patches = []
    for zone_name in zone_names:
        base = zone_name.replace(" (left block)", "").replace(" (right block)", "") \
                        .replace(" (left)", "").replace(" (right)", "") \
                        .replace(" (center)", "").replace("left ", "").replace("right ", "") \
                        .replace("top ", "").replace("top of the ", "")
        if base in seen:
            continue
        seen.add(base)
        legend_patches.append(Patch(facecolor=ZONE_COLORS[zone_name],
                                    edgecolor='white', alpha=0.7,
                                    label=base.title()))

    ax.legend(handles=legend_patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.06), ncol=len(legend_patches),
              fontsize=7, frameon=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"Zone map saved to {output_path}")


if __name__ == "__main__":
    generate_zone_map()
