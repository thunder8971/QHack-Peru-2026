"""Classical image processing for OrionQ."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


Zone = dict[str, Any]

MIN_AREA = 1_000.0
OVERLAY_ALPHA = 0.35

RED_RANGES = (
    (np.array([0, 150, 150]), np.array([12, 255, 255])),
    (np.array([168, 150, 150]), np.array([179, 255, 255])),
)
YELLOW_RANGE = (
    np.array([8, 150, 150]),
    np.array([42, 255, 255]),
)

ZONE_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
}
ZONE_WEIGHTS = {
    "red": 1.0,
    "yellow": 0.5,
}


def _read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return image


def _build_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, *RED_RANGES[0])
    red_mask |= cv2.inRange(hsv, *RED_RANGES[1])
    yellow_mask = cv2.inRange(hsv, *YELLOW_RANGE)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    masks = {"red": red_mask, "yellow": yellow_mask}
    for color, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        masks[color] = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )
    return masks


def _centroid(contour: np.ndarray) -> list[int]:
    moments = cv2.moments(contour)
    if moments["m00"]:
        return [
            round(moments["m10"] / moments["m00"]),
            round(moments["m01"] / moments["m00"]),
        ]

    x, y, width, height = cv2.boundingRect(contour)
    return [x + width // 2, y + height // 2]


def get_zones(image: np.ndarray) -> list[Zone]:
    """Extract external red and yellow regions as simple dictionaries."""
    zones: list[Zone] = []

    for color, mask in _build_masks(image).items():
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            hull = cv2.convexHull(contour)
            area = float(cv2.contourArea(hull))
            if area < MIN_AREA:
                continue

            zones.append(
                {
                    "color": color,
                    "area": area,
                    "weight": ZONE_WEIGHTS[color],
                    "raw_priority": area * ZONE_WEIGHTS[color],
                    "centroid": _centroid(hull),
                    "convex_hull": hull.reshape(-1, 2).astype(int).tolist(),
                }
            )

    zones.sort(key=lambda zone: (zone["centroid"][1], zone["centroid"][0]))
    maximum_priority = max(
        (zone["raw_priority"] for zone in zones),
        default=0.0,
    )

    for zone_id, zone in enumerate(zones):
        zone["id"] = zone_id
        prefix = "R" if zone["color"] == "red" else "Y"
        zone["label"] = f"{prefix}{zone_id}"
        zone["priority"] = (
            zone["raw_priority"] / maximum_priority
            if maximum_priority
            else 0.0
        )
    return zones


def assign_priority_order(zones: list[Zone]) -> list[Zone]:
    """Order zones for attention without replacing their decimal scores."""
    score_key = (
        "quantum_score"
        if zones and all("quantum_score" in zone for zone in zones)
        else "priority"
    )
    ordered_zones = sorted(
        zones,
        key=lambda zone: (
            -float(zone[score_key]),
            -float(zone["area"]),
            int(zone["id"]),
        ),
    )
    for order, zone in enumerate(ordered_zones, start=1):
        zone["priority_order"] = order
    return ordered_zones


def paint_zones(image: np.ndarray, zones: list[Zone]) -> np.ndarray:
    """Paint translucent hulls plus clear attention-order labels."""
    result = image.copy()
    overlay = image.copy()
    zones = assign_priority_order(zones)

    for zone in zones:
        hull = np.asarray(zone["convex_hull"], dtype=np.int32)
        cv2.fillConvexPoly(overlay, hull, ZONE_COLORS[zone["color"]])

    cv2.addWeighted(
        overlay,
        OVERLAY_ALPHA,
        result,
        1.0 - OVERLAY_ALPHA,
        0,
        result,
    )

    height, width = result.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_font_scale = max(0.7, min(height, width) / 800)
    thickness = 2
    padding = 7

    for zone in zones:
        hull = np.asarray(zone["convex_hull"], dtype=np.int32)
        color = ZONE_COLORS[zone["color"]]
        cv2.polylines(result, [hull], True, color, 2, cv2.LINE_AA)

        text = f"P{zone['priority_order']}"
        x, y = zone["centroid"]
        font_scale = base_font_scale
        text_size, baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        maximum_text_width = max(1, width - 2 * padding)
        if text_size[0] > maximum_text_width:
            font_scale *= maximum_text_width / text_size[0]
            text_size, baseline = cv2.getTextSize(
                text,
                font,
                font_scale,
                thickness,
            )

        box_width = text_size[0] + 2 * padding
        box_height = text_size[1] + baseline + 2 * padding
        left = min(max(0, x - box_width // 2), max(0, width - box_width))
        top = min(max(0, y - box_height // 2), max(0, height - box_height))
        right = min(width - 1, left + box_width)
        bottom = min(height - 1, top + box_height)
        cv2.rectangle(result, (left, top), (right, bottom), (0, 0, 0), -1)

        cv2.putText(
            result,
            text,
            (left + padding, top + padding + text_size[1]),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return result


def process_image(
    input_path: str | Path,
    output_path: str | Path,
) -> tuple[list[Zone], np.ndarray]:
    """Detect marked zones, save the visualization, and return both results."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    annotated_image = _read_image(input_path)
    zones = assign_priority_order(get_zones(annotated_image))

    clean_path = input_path.with_name("original.jpeg")
    canvas = _read_image(clean_path) if clean_path.is_file() else annotated_image
    if canvas.shape != annotated_image.shape:
        raise ValueError("The clean and annotated images must have equal sizes.")

    result = paint_zones(canvas, zones)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Could not save result image: {output_path}")
    return zones, result

