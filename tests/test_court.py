"""Tests for court landmark-based position description."""

import pytest
from src.court import describe_position, normalize_coords, distance_to_basket


class TestDistanceToBasket:
    def test_at_basket(self):
        assert distance_to_basket(5.25, 25.0) == 0.0

    def test_known_distance(self):
        # 3 feet directly in front of basket
        d = distance_to_basket(8.25, 25.0)
        assert abs(d - 3.0) < 0.01


class TestNormalizeCoords:
    def test_attacking_left_mirrors_y(self):
        # When attacking left, y is mirrored: y = 50 - y
        x, y = normalize_coords(20.0, 30.0, attacking_right=False)
        assert x == 20.0
        assert y == 20.0  # 50 - 30

    def test_attacking_right_mirrors_x(self):
        x, y = normalize_coords(80.0, 30.0, attacking_right=True)
        assert x == 14.0  # 94 - 80
        assert y == 30.0  # y unchanged


class TestDescribePosition:
    # --- "at" relation (landmark points) ---
    def test_at_basket(self):
        result = describe_position(5.25, 25.0)
        assert result["relation"] == "at"
        assert result["landmark"] == "basket"
        assert result["distance"] == 0.0

    def test_at_left_elbow(self):
        result = describe_position(19.0, 17.0)
        assert result["relation"] == "at"
        assert result["landmark"] == "left elbow"

    def test_at_right_elbow(self):
        result = describe_position(19.0, 33.0)
        assert result["relation"] == "at"
        assert result["landmark"] == "right elbow"

    def test_at_left_corner(self):
        result = describe_position(5.0, 3.0)
        assert result["relation"] == "at"
        assert result["landmark"] == "left corner"

    def test_at_right_corner(self):
        result = describe_position(5.0, 47.0)
        assert result["relation"] == "at"
        assert result["landmark"] == "right corner"

    def test_at_the_logo(self):
        result = describe_position(47.0, 25.0)
        assert result["relation"] == "at"
        assert result["landmark"] == "the logo"

    def test_at_left_wing(self):
        result = describe_position(28.0, 8.0)
        assert result["relation"] == "at"
        assert result["landmark"] == "left wing"

    def test_at_right_wing(self):
        result = describe_position(28.0, 42.0)
        assert result["relation"] == "at"
        assert result["landmark"] == "right wing"

    def test_at_top_of_the_arc(self):
        result = describe_position(29.0, 25.0)
        assert result["relation"] == "at"
        assert result["landmark"] == "top of the arc"

    # --- "on" relation (landmark lines) ---
    def test_on_baseline(self):
        # (0, 25) is on the baseline, not near any point
        result = describe_position(0.0, 25.0)
        assert result["relation"] == "on"
        assert result["landmark"] == "baseline"

    def test_on_left_lane_line(self):
        result = describe_position(10.0, 17.0)
        assert result["relation"] == "on"
        assert result["landmark"] == "left lane line"

    # --- "near" relation (fallback) ---
    def test_near_fallback(self):
        # A point in the mid-range, not close to any point or line
        result = describe_position(15.0, 10.0)
        assert result["relation"] == "near"
        assert "near" in result["description"]

    # --- Side detection ---
    def test_side_left(self):
        result = describe_position(19.0, 17.0)
        assert result["side"] == "left"

    def test_side_right(self):
        result = describe_position(19.0, 33.0)
        assert result["side"] == "right"

    def test_side_center(self):
        result = describe_position(29.0, 25.0)
        assert result["side"] == "center"

    # --- Return schema ---
    def test_return_keys(self):
        result = describe_position(19.0, 17.0)
        assert "description" in result
        assert "landmark" in result
        assert "relation" in result
        assert "distance" in result
        assert "side" in result
        assert "distance_to_basket" in result

    def test_distance_to_basket_in_result(self):
        result = describe_position(5.25, 25.0)
        assert result["distance_to_basket"] == 0.0

    # --- Points take priority over lines ---
    def test_point_priority_over_line(self):
        # Left elbow is at (19, 17) — exactly at the corner of the FT line and lane line.
        # The point should take priority over both lines.
        result = describe_position(19.0, 17.0)
        assert result["relation"] == "at"
        assert result["landmark"] == "left elbow"
