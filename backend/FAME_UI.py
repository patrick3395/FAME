"""FAME_UI: Polygon-aware analysis pipeline for the FP1 interactive UI."""
from __future__ import annotations

import base64
import dataclasses
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from PIL import Image
from scipy.interpolate import griddata

# Use non-interactive backend so the service can render off-screen
plt.ioff()

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float
    label: str | None = None


@dataclass(frozen=True)
class BoundaryPoint:
    x: float
    y: float


@dataclass
class AnalysisPayload:
    boundary: List[BoundaryPoint]
    points: List[Point3D]
    spacing: float
    unit: str
    floorplan_image: Optional[str] = None

    @classmethod
    def from_request(cls, payload: Dict) -> "AnalysisPayload":
        boundary_raw = payload.get("boundary")
        points_raw = payload.get("points")
        spacing = float(payload.get("spacing", 1.0))
        unit = str(payload.get("unit", "ft"))
        floorplan_image = payload.get("floorplanImage") or payload.get("floorplan_image")

        if not boundary_raw or len(boundary_raw) < 3:
            raise ValueError("Boundary must contain at least three points")
        if not points_raw or len(points_raw) < 3:
            raise ValueError("At least three measurement points are required")

        boundary = [BoundaryPoint(float(p["x"]), float(p["y"])) for p in boundary_raw]
        points = [
            Point3D(float(p["x"]), float(p["y"]), float(p["z"]), p.get("label"))
            for p in points_raw
        ]

        return cls(
            boundary=boundary,
            points=points,
            spacing=spacing,
            unit=unit,
            floorplan_image=floorplan_image,
        )


@dataclass
class AnalysisResult:
    images: Dict[str, str]
    profile_lines: List[Dict[str, Tuple[float, float]]]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def ensure_closed_polygon(vertices: Sequence[BoundaryPoint]) -> np.ndarray:
    """Return an array of polygon vertices ensuring the last point equals the first."""
    arr = np.array([[p.x, p.y] for p in vertices], dtype=float)
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack([arr, arr[0]])
    return arr


def polygon_bounds(polygon: np.ndarray) -> Tuple[float, float, float, float]:
    min_x = float(np.min(polygon[:, 0]))
    max_x = float(np.max(polygon[:, 0]))
    min_y = float(np.min(polygon[:, 1]))
    max_y = float(np.max(polygon[:, 1]))
    return min_x, max_x, min_y, max_y


def horizontal_intersections(polygon: np.ndarray, y_value: float) -> List[float]:
    """Return sorted x-intersections between a horizontal line and the polygon."""
    intersections: List[float] = []
    for (x1, y1), (x2, y2) in zip(polygon[:-1], polygon[1:]):
        if (y1 <= y_value < y2) or (y2 <= y_value < y1):
            if math.isclose(y1, y2):
                continue
            ratio = (y_value - y1) / (y2 - y1)
            x_cross = x1 + ratio * (x2 - x1)
            intersections.append(x_cross)
    intersections.sort()
    return intersections


def vertical_intersections(polygon: np.ndarray, x_value: float) -> List[float]:
    """Return sorted y-intersections between a vertical line and the polygon."""
    intersections: List[float] = []
    for (x1, y1), (x2, y2) in zip(polygon[:-1], polygon[1:]):
        if (x1 <= x_value < x2) or (x2 <= x_value < x1):
            if math.isclose(x1, x2):
                continue
            ratio = (x_value - x1) / (x2 - x1)
            y_cross = y1 + ratio * (y2 - y1)
            intersections.append(y_cross)
    intersections.sort()
    return intersections


def generate_profile_lines(
    polygon: np.ndarray,
    num_horizontal: int = 4,
    num_vertical: int = 4,
) -> List[Dict[str, Tuple[float, float]]]:
    """Generate profile lines that start and end on the polygon boundary."""
    min_x, max_x, min_y, max_y = polygon_bounds(polygon)
    results: List[Dict[str, Tuple[float, float]]] = []

    horizontal_levels = np.linspace(min_y, max_y, num_horizontal + 2)[1:-1]
    for y in horizontal_levels:
        xs = horizontal_intersections(polygon, y)
        for start_x, end_x in zip(xs[0::2], xs[1::2]):
            results.append({
                "orientation": "horizontal",
                "start": (start_x, y),
                "end": (end_x, y),
            })

    vertical_levels = np.linspace(min_x, max_x, num_vertical + 2)[1:-1]
    for x in vertical_levels:
        ys = vertical_intersections(polygon, x)
        for start_y, end_y in zip(ys[0::2], ys[1::2]):
            results.append({
                "orientation": "vertical",
                "start": (x, start_y),
                "end": (x, end_y),
            })

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def to_base64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _decode_floorplan(image_data: Optional[str]) -> Optional[np.ndarray]:
    if not image_data:
        return None

    try:
        encoded = image_data.split(",", 1)[1] if image_data.startswith("data:") else image_data
        image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGBA")
        return np.array(image)
    except Exception:
        return None


def _draw_floorplan(ax: plt.Axes, polygon: np.ndarray, floorplan_array: Optional[np.ndarray], alpha: float = 0.85):
    if floorplan_array is None:
        return

    min_x, max_x, min_y, max_y = polygon_bounds(polygon)
    ax.imshow(
        floorplan_array,
        extent=(min_x, max_x, min_y, max_y),
        origin="lower",
        alpha=alpha,
        zorder=0,
    )


def plot_heatmap(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    polygon: np.ndarray,
    points: Sequence[Point3D],
    title: str,
    floorplan_array: Optional[np.ndarray] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_floorplan(ax, polygon, floorplan_array)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=32, cmap="coolwarm", alpha=0.65, zorder=1)
    fig.colorbar(contour, ax=ax, shrink=0.8, pad=0.02, label="Elevation")
    ax.add_collection(
        PatchCollection([MplPolygon(polygon)], facecolor="none", edgecolor="black", linewidth=1.5, zorder=2)
    )
    ax.scatter([p.x for p in points], [p.y for p in points], c="black", s=18, alpha=0.8, zorder=3)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return to_base64(fig)


def plot_repair_plan(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    polygon: np.ndarray,
    profile_lines: Sequence[Dict[str, Tuple[float, float]]],
    title: str,
    floorplan_array: Optional[np.ndarray] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_floorplan(ax, polygon, floorplan_array)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=24, cmap="viridis", alpha=0.6, zorder=1)
    fig.colorbar(contour, ax=ax, shrink=0.8, pad=0.02, label="Elevation")
    ax.add_collection(
        PatchCollection([MplPolygon(polygon)], facecolor="none", edgecolor="#0f172a", linewidth=2, zorder=2)
    )

    for line in profile_lines:
        (x1, y1) = line["start"]
        (x2, y2) = line["end"]
        ax.plot([x1, x2], [y1, y2], "w--", linewidth=1.2, alpha=0.6, zorder=3)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return to_base64(fig)


def plot_profiles(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    polygon: np.ndarray,
    profile_lines: Sequence[Dict[str, Tuple[float, float]]],
    points: Sequence[Point3D],
    title: str,
    floorplan_array: Optional[np.ndarray] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_floorplan(ax, polygon, floorplan_array)
    ax.contourf(grid_x, grid_y, grid_z, levels=32, cmap="coolwarm", alpha=0.35, zorder=1)
    ax.add_collection(
        PatchCollection([MplPolygon(polygon)], facecolor="none", edgecolor="#2563eb", linewidth=2, zorder=2)
    )
    for line in profile_lines:
        (x1, y1) = line["start"]
        (x2, y2) = line["end"]
        ax.plot([x1, x2], [y1, y2], color="#f97316", linewidth=1.4, alpha=0.8, zorder=3)
    ax.scatter([p.x for p in points], [p.y for p in points], c="#0f172a", s=18, zorder=4)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return to_base64(fig)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


class FameUIPipeline:
    def __init__(self, grid_resolution: int = 200):
        self.grid_resolution = grid_resolution

    def run(self, payload: AnalysisPayload) -> AnalysisResult:
        polygon = ensure_closed_polygon(payload.boundary)
        path = Path(polygon)

        points_array = np.array([[p.x, p.y] for p in payload.points])
        z_array = np.array([p.z for p in payload.points])

        min_x, max_x, min_y, max_y = polygon_bounds(polygon)
        grid_x, grid_y = np.meshgrid(
            np.linspace(min_x, max_x, self.grid_resolution),
            np.linspace(min_y, max_y, self.grid_resolution),
        )

        interpolated = griddata(points_array, z_array, (grid_x, grid_y), method="cubic")

        # Mask everything outside the polygon
        mask = ~path.contains_points(np.vstack((grid_x.flatten(), grid_y.flatten())).T)
        masked_grid = np.ma.array(interpolated, mask=mask.reshape(grid_x.shape))

        profile_lines = generate_profile_lines(polygon)
        floorplan_array = _decode_floorplan(payload.floorplan_image)

        images = {
            "heatmap": plot_heatmap(
                grid_x, grid_y, masked_grid, polygon, payload.points, "FP1 Elevation Heatmap", floorplan_array
            ),
            "repair_plan": plot_repair_plan(
                grid_x, grid_y, masked_grid, polygon, profile_lines, "Contour Map", floorplan_array
            ),
            "profiles": plot_profiles(
                grid_x, grid_y, masked_grid, polygon, profile_lines, payload.points, "Profile Layout", floorplan_array
            ),
        }

        return AnalysisResult(images=images, profile_lines=profile_lines)


# ---------------------------------------------------------------------------
# Flask service facade
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/api/fame/run": {"origins": "*"}}, supports_credentials=False)
pipeline = FameUIPipeline()


@app.route("/api/fame/run", methods=["POST"])
def run_analysis():
    try:
        payload = AnalysisPayload.from_request(request.get_json(force=True))
        result = pipeline.run(payload)
        return jsonify({
            "images": result.images,
            "profileLines": result.profile_lines,
            "unit": payload.unit,
        })
    except Exception as exc:  # pragma: no cover - top-level guard
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)
