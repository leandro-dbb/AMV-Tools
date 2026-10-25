"""Mask Mode request bounds (brush radius / points, keyframe range)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.api.segment import BrushStroke


@pytest.mark.parametrize("radius", [0, 0.5, 513, 1e6])
def test_brush_radius_is_bounded(radius):
    with pytest.raises(ValidationError):
        BrushStroke(points=[(1, 1)], radius=radius)


def test_brush_points_are_bounded():
    BrushStroke(points=[(1, 1)] * 100, radius=12)
    with pytest.raises(ValidationError):
        BrushStroke(points=[(1, 1)] * 20_001, radius=12)
