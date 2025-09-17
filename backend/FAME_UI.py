"""FAME_UI: Polygon-aware analysis pipeline for the FP1 interactive UI."""
from __future__ import annotations

import base64
import dataclasses
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image
from scipy.interpolate import griddata

# Use non-interactive backend so the service can render off-screen
plt.ioff()

logger = logging.getLogger('FAME_UI')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

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

@dataclass(frozen=True)
class ColorScale:
    cmap: LinearSegmentedColormap
    norm: TwoSlopeNorm
    levels: np.ndarray
    vmin: float
    vmax: float


def _summarize_points(points: Sequence[Point3D]) -> str:
    if not points:
        return '0 pts'
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    return (
        f"{len(points)} pts | x:[{min(xs):.2f},{max(xs):.2f}] "
        f"y:[{min(ys):.2f},{max(ys):.2f}] z:[{min(zs):.2f},{max(zs):.2f}]"
    )


def _summarize_boundary(boundary: Sequence[BoundaryPoint]) -> str:
    if not boundary:
        return '0 vertices'
    xs = [p.x for p in boundary]
    ys = [p.y for p in boundary]
    return (
        f"{len(boundary)} vertices | x:[{min(xs):.2f},{max(xs):.2f}] "
        f"y:[{min(ys):.2f},{max(ys):.2f}]"
    )


def _summarize_grid(label: str, grid: np.ndarray) -> str:
    finite = np.asarray(grid[~np.isnan(grid)]) if isinstance(grid, np.ndarray) else np.asarray(grid)
    if finite.size == 0:
        return f"{label}: empty"
    return f"{label}: shape={grid.shape} range=[{finite.min():.3f},{finite.max():.3f}]"


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


def _interpolate_surface(points: np.ndarray, values: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Interpolate with cubic preference and fallbacks to cover the polygon interior."""
    filled = np.full(grid_x.shape, np.nan, dtype=float)
    for method in ("cubic", "linear", "nearest"):
        try:
            candidate = griddata(points, values, (grid_x, grid_y), method=method)
        except Exception:
            candidate = None
        if candidate is None:
            continue
        mask = np.isnan(filled)
        filled[mask] = candidate[mask]
        if not np.isnan(filled).any():
            break
    if np.isnan(filled).any():
        fallback = float(np.nanmean(values))
        filled = np.where(np.isnan(filled), fallback, filled)
    return np.clip(filled, float(np.nanmin(values)), float(np.nanmax(values)))

def _compute_color_scale(grid_z: np.ndarray) -> ColorScale:
    """Generate a red-to-green diverging color scale centered at zero."""
    data = grid_z.compressed() if np.ma.isMaskedArray(grid_z) else grid_z[np.isfinite(grid_z)]
    if data.size == 0:
        data = np.array([0.0])
    data_min = min(float(np.min(data)), 0.0)
    data_max = max(float(np.max(data)), 0.0)
    lo = math.floor(data_min * 10.0) / 10.0 - 0.3
    hi = math.ceil(data_max * 10.0) / 10.0 + 0.3
    if math.isclose(lo, hi):
        spread = max(abs(lo), abs(hi), 0.5)
        lo = -spread
        hi = spread
    zero_ratio = float(np.clip((0.0 - lo) / (hi - lo), 0.0, 1.0))
    cmap = LinearSegmentedColormap.from_list(
        "fame_red_green",
        [
            (0.0, "#991b1b"),
            (zero_ratio, "#f8fafc"),
            (1.0, "#166534"),
        ],
    )
    levels = np.linspace(lo, hi, 32)
    if not np.isclose(levels, 0.0).any():
        levels = np.sort(np.append(levels, 0.0))
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    return ColorScale(cmap=cmap, norm=norm, levels=levels, vmin=lo, vmax=hi)

def plot_heatmap(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    polygon: np.ndarray,
    points: Sequence[Point3D],
    title: str,
    color_scale: ColorScale,
    floorplan_array: Optional[np.ndarray] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_floorplan(ax, polygon, floorplan_array)
    contour = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=color_scale.levels,
        cmap=color_scale.cmap,
        norm=color_scale.norm,
        zorder=1,
    )
    fig.colorbar(
        contour,
        ax=ax,
        shrink=0.82,
        pad=0.02,
        label="Elevation",
        extend="both",
    )
    ax.add_collection(
        PatchCollection([MplPolygon(polygon)], facecolor="none", edgecolor="black", linewidth=1.5, zorder=2)
    )
    ax.scatter(
        [p.x for p in points],
        [p.y for p in points],
        c="#111827",
        edgecolors="white",
        linewidths=0.6,
        s=36,
        zorder=3,
    )
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
    color_scale: ColorScale,
    floorplan_array: Optional[np.ndarray] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_floorplan(ax, polygon, floorplan_array)
    contour = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=color_scale.levels,
        cmap=color_scale.cmap,
        norm=color_scale.norm,
        alpha=0.85,
        zorder=1,
    )
    contour_lines = ax.contour(
        grid_x,
        grid_y,
        grid_z,
        levels=color_scale.levels,
        colors="#1f2937",
        linewidths=0.8,
        zorder=2.5,
    )
    ax.clabel(contour_lines, fmt="%.1f", fontsize=7, inline=True)
    fig.colorbar(
        contour,
        ax=ax,
        shrink=0.82,
        pad=0.02,
        label="Elevation",
        extend="both",
    )
    ax.add_collection(
        PatchCollection([MplPolygon(polygon)], facecolor="none", edgecolor="#0f172a", linewidth=2, zorder=3)
    )
    for line in profile_lines:
        (x1, y1) = line["start"]
        (x2, y2) = line["end"]
        ax.plot([x1, x2], [y1, y2], color="white", linestyle="--", linewidth=1.1, alpha=0.7, zorder=4)
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
    color_scale: ColorScale,
    floorplan_array: Optional[np.ndarray] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_floorplan(ax, polygon, floorplan_array)
    contour = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=color_scale.levels,
        cmap=color_scale.cmap,
        norm=color_scale.norm,
        alpha=0.4,
        zorder=1,
    )
    ax.add_collection(
        PatchCollection([MplPolygon(polygon)], facecolor="none", edgecolor="#2563eb", linewidth=2, zorder=2)
    )
    for line in profile_lines:
        (x1, y1) = line["start"]
        (x2, y2) = line["end"]
        ax.plot([x1, x2], [y1, y2], color="#f97316", linewidth=1.2, alpha=0.85, zorder=3)
    ax.scatter(
        [p.x for p in points],
        [p.y for p in points],
        c="#0f172a",
        edgecolors="white",
        linewidths=0.5,
        s=32,
        zorder=4,
    )
    fig.colorbar(
        contour,
        ax=ax,
        shrink=0.82,
        pad=0.02,
        label="Elevation",
        extend="both",
    )
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
        logger.info(
            'FP1 payload: %s; %s; spacing=%.3f %s; floorplan=%s',
            _summarize_boundary(payload.boundary),
            _summarize_points(payload.points),
            payload.spacing,
            payload.unit,
            bool(payload.floorplan_image),
        )
        print(
            "[FAME_UI] Processing payload -> boundary=%s points=%s spacing=%.3f %s floorplan=%s"
            % (
                _summarize_boundary(payload.boundary),
                _summarize_points(payload.points),
                payload.spacing,
                payload.unit,
                bool(payload.floorplan_image),
            )
        )

        points_array = np.array([[p.x, p.y] for p in payload.points])
        z_array = np.array([p.z for p in payload.points])

        min_x, max_x, min_y, max_y = polygon_bounds(polygon)
        grid_x, grid_y = np.meshgrid(
            np.linspace(min_x, max_x, self.grid_resolution),
            np.linspace(min_y, max_y, self.grid_resolution),
        )

        interpolated = _interpolate_surface(points_array, z_array, grid_x, grid_y)
        logger.info(_summarize_grid('Interpolated grid', interpolated))
        print("[FAME_UI] %s" % _summarize_grid('Interpolated grid', interpolated))

        mask = ~path.contains_points(np.vstack((grid_x.flatten(), grid_y.flatten())).T)
        mask = mask.reshape(grid_x.shape)
        interpolated = np.where(mask, np.nan, interpolated)
        masked_grid = np.ma.array(interpolated, mask=mask)
        logger.info('Masked cells=%d', int(mask.sum()))
        logger.info(_summarize_grid('Masked grid', masked_grid.filled(np.nan)))
        print(f"[FAME_UI] Masked cells={int(mask.sum())}")
        print("[FAME_UI] %s" % _summarize_grid('Masked grid', masked_grid.filled(np.nan)))

        profile_lines = generate_profile_lines(polygon)
        logger.info('Generated %d profile lines', len(profile_lines))
        print(f"[FAME_UI] Generated {len(profile_lines)} profile lines")
        floorplan_array = _decode_floorplan(payload.floorplan_image)
        color_scale = _compute_color_scale(masked_grid)
        logger.info('Color scale: vmin=%.3f vmax=%.3f levels=%d', color_scale.vmin, color_scale.vmax, len(color_scale.levels))
        print(
            "[FAME_UI] Color scale -> vmin=%.3f vmax=%.3f levels=%d"
            % (color_scale.vmin, color_scale.vmax, len(color_scale.levels))
        )

        images = {
            "heatmap": plot_heatmap(
                grid_x,
                grid_y,
                masked_grid,
                polygon,
                payload.points,
                "FP1 Elevation Heatmap",
                color_scale,
                floorplan_array,
            ),
            "repair_plan": plot_repair_plan(
                grid_x,
                grid_y,
                masked_grid,
                polygon,
                profile_lines,
                "Contour Map",
                color_scale,
                floorplan_array,
            ),
            "profiles": plot_profiles(
                grid_x,
                grid_y,
                masked_grid,
                polygon,
                profile_lines,
                payload.points,
                "Profile Layout",
                color_scale,
                floorplan_array,
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
    logger.info("HELLO from new revision")
    try:
        logger.info('run_analysis invoked from %s', request.remote_addr)
        print(f"[FAME_UI] run_analysis invoked from {request.remote_addr}")
        payload = AnalysisPayload.from_request(request.get_json(force=True))
        result = pipeline.run(payload)
        logger.info('run_analysis completed successfully')
        print("[FAME_UI] run_analysis completed successfully")
        return jsonify({
            "images": result.images,
            "profileLines": result.profile_lines,
            "unit": payload.unit,
        })
    except Exception as exc:  # pragma: no cover - top-level guard
        logger.exception('run_analysis failed: %s', exc)
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)



