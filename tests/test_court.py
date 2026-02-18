"""Tests for court zone classification."""

import pytest
from src.court import classify_zone, normalize_coords, distance_to_basket


class TestDistanceToBasket:
    def test_at_basket(self):
        assert distance_to_basket(5.25, 25.0) == 0.0

    def test_known_distance(self):
        # 3 feet directly in front of basket
        d = distance_to_basket(8.25, 25.0)
        assert abs(d - 3.0) < 0.01


class TestNormalizeCoords:
    def test_attacking_left_no_change(self):
        x, y = normalize_coords(20.0, 30.0, attacking_right=False)
        assert x == 20.0
        assert y == 30.0

    def test_attacking_right_mirrors_x(self):
        x, y = normalize_coords(80.0, 30.0, attacking_right=True)
        assert x == 14.0  # 94 - 80
        assert y == 30.0  # y unchanged


class TestClassifyZone:
    def test_restricted_area(self):
        zone = classify_zone(5.25, 25.0)
        assert zone["zone"] == "restricted area"
        assert zone["distance_to_basket"] == 0.0

    def test_restricted_area_near_basket(self):
        zone = classify_zone(3.0, 25.0)
        assert zone["zone"] == "restricted area"

    def test_paint_left(self):
        zone = classify_zone(10.0, 20.0)
        assert "paint" in zone["zone"]
        assert zone["side"] == "left"

    def test_paint_right(self):
        zone = classify_zone(10.0, 30.0)
        assert "paint" in zone["zone"]
        assert zone["side"] == "right"

    def test_left_corner_three(self):
        zone = classify_zone(5.0, 3.0)
        assert zone["zone"] == "left corner three"
        assert zone["beyond_arc"] is True

    def test_right_corner_three(self):
        zone = classify_zone(5.0, 47.0)
        assert zone["zone"] == "right corner three"
        assert zone["beyond_arc"] is True

    def test_left_wing_three(self):
        zone = classify_zone(30.0, 10.0)
        assert zone["zone"] == "left wing three"
        assert zone["beyond_arc"] is True

    def test_right_wing_three(self):
        zone = classify_zone(30.0, 40.0)
        assert zone["zone"] == "right wing three"
        assert zone["beyond_arc"] is True

    def test_top_of_arc(self):
        zone = classify_zone(30.0, 25.0)
        assert zone["zone"] == "top of the arc"
        assert zone["beyond_arc"] is True

    def test_backcourt(self):
        zone = classify_zone(60.0, 25.0)
        assert zone["zone"] == "backcourt"

    def test_left_midrange(self):
        zone = classify_zone(15.0, 15.0)
        assert zone["zone"] == "left mid-range"
        assert zone["beyond_arc"] is False

    def test_right_midrange(self):
        zone = classify_zone(15.0, 35.0)
        assert zone["zone"] == "right mid-range"
        assert zone["beyond_arc"] is False

    def test_top_midrange(self):
        zone = classify_zone(20.0, 25.0)
        assert zone["zone"] == "top mid-range"
        assert zone["beyond_arc"] is False
