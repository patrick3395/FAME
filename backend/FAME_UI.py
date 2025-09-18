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
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import ticker
import matplotlib.patheffects as PathEffects
from PIL import Image
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, cKDTree

# Use non-interactive backend so the service can render off-screen
plt.ioff()

BACKEND_REVISION_TAG = "backend-v1"

FLOORPLAN_ALPHA = 1.0

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


def _corsify_response(response):
    """Attach the CORS headers expected by the FP1 frontend."""
    origin = request.headers.get('Origin', '*') if request else '*'
    response.headers['Access-Control-Allow-Origin'] = origin
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response


def _generate_boundary_samples(boundary_points: Sequence[Tuple[float, float]], spacing: float) -> List[Tuple[float, float]]:
    if spacing <= 0:
        return []

    samples: List[Tuple[float, float]] = []
    boundary_array = np.asarray(boundary_points, dtype=np.float32)
    if len(boundary_array) < 2:
        return samples

    for idx in range(len(boundary_array) - 1):
        start = boundary_array[idx]
        end = boundary_array[idx + 1]
        segment = end - start
        length = float(np.hypot(segment[0], segment[1]))
        if length == 0:
            continue

        num_points = max(int(np.floor(length / spacing)), 1)
        for t in np.linspace(0.0, 1.0, num_points, endpoint=False):
            sample = (float(start[0] + t * segment[0]), float(start[1] + t * segment[1]))
            samples.append(sample)

    return samples


def _annotate_points(ax: plt.Axes, points: Sequence[Point3D]) -> None:
    for point in points:
        color = '#166534' if point.z > 0 else ('#991b1b' if point.z < 0 else '#1d4ed8')
        marker_edge = '#111827'
        value_text = f"{point.z:.1f}"
        if point.z > 0:
            value_text = f"+{value_text}"

        ax.scatter(point.x, point.y, c=color, edgecolors=marker_edge, linewidths=0.6, s=32, zorder=7)
        ax.text(
            point.x,
            point.y + 1.2,
            value_text,
            color=color,
            fontsize=9,
            fontweight='medium',
            ha='center',
            va='top',
            zorder=8,
            path_effects=[PathEffects.withStroke(linewidth=1.0, foreground='white')],
        )



def _scan_cartesian_intersections(polygon: np.ndarray, value: float, horizontal: bool = True) -> List[float]:
    intersections: List[float] = []
    for (x1, y1), (x2, y2) in zip(polygon[:-1], polygon[1:]):
        if horizontal:
            y_min, y_max = sorted((y1, y2))
            if y_min <= value < y_max or (math.isclose(value, y_max) and value != min(y1, y2)):
                if math.isclose(y1, y2):
                    continue
                t = (value - y1) / (y2 - y1)
                x_cross = x1 + t * (x2 - x1)
                intersections.append(x_cross)
        else:
            x_min, x_max = sorted((x1, x2))
            if x_min <= value < x_max or (math.isclose(value, x_max) and value != min(x1, x2)):
                if math.isclose(x1, x2):
                    continue
                t = (value - x1) / (x2 - x1)
                y_cross = y1 + t * (y2 - y1)
                intersections.append(y_cross)
    intersections.sort()
    return intersections


def _generate_profile_lines_from_boundary(boundary: Sequence[BoundaryPoint], polygon_path: Path) -> List[Dict[str, Tuple[float, float]]]:
    coords = [(point.x, point.y) for point in boundary]
    if len(coords) < 3:
        return []

    polygon = np.array(coords + [coords[0]], dtype=float)
    min_x, max_x, min_y, max_y = polygon_bounds(polygon)

    lines: List[Dict[str, Tuple[float, float]]] = []

    horizontal_levels = np.linspace(min_y, max_y, 42)[1:-1]
    for y in horizontal_levels:
        xs = _scan_cartesian_intersections(polygon, y, horizontal=True)
        for i in range(0, len(xs) - 1, 2):
            start = (xs[i], y)
            end = (xs[i + 1], y)
            midpoint = ((start[0] + end[0]) / 2.0, y)
            if polygon_path.contains_point(midpoint):
                lines.append({'start': start, 'end': end})

    vertical_levels = np.linspace(min_x, max_x, 42)[1:-1]
    for x in vertical_levels:
        ys = _scan_cartesian_intersections(polygon, x, horizontal=False)
        for i in range(0, len(ys) - 1, 2):
            start = (x, ys[i])
            end = (x, ys[i + 1])
            midpoint = (x, (start[1] + end[1]) / 2.0)
            if polygon_path.contains_point(midpoint):
                lines.append({'start': start, 'end': end})

    num_coords = len(coords)
    for i in range(num_coords):
        start = coords[i]
        for j in range(i + 1, num_coords):
            end = coords[j]
            lines.append({'start': start, 'end': end})

    for idx in range(len(coords)):
        start = coords[idx]
        end = coords[(idx + 1) % len(coords)]
        lines.append({'start': start, 'end': end})

    deduped: List[Dict[str, Tuple[float, float]]] = []
    seen = set()
    for line in lines:
        start = (round(line['start'][0], 4), round(line['start'][1], 4))
        end = (round(line['end'][0], 4), round(line['end'][1], 4))
        key = tuple(sorted([start, end]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)

    return deduped


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
    except Exception as exc:
        logger.warning("Failed to decode floorplan image: %s", exc)
        return None


def _autocrop_floorplan(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        mask = alpha > 8
    else:
        grayscale = np.mean(image[:, :, :3], axis=2)
        mask = grayscale < 248

    coords = np.argwhere(mask)
    if coords.size == 0:
        return image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    crop = image[y0:y1, x0:x1]
    # Fallback if cropping removed everything
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return image
    return crop


def _draw_floorplan(ax: plt.Axes, polygon: np.ndarray, floorplan_array: Optional[np.ndarray], alpha: float = FLOORPLAN_ALPHA):
    if floorplan_array is None:
        return

    cropped = _autocrop_floorplan(floorplan_array)
    flipped = np.flipud(cropped)

    if flipped.shape[2] == 4:
        rgba = flipped.copy()
    else:
        alpha_channel = np.full(flipped.shape[:2], 255, dtype=flipped.dtype)
        rgba = np.dstack([flipped, alpha_channel])

    rgb = rgba[:, :, :3].astype(np.float32)
    grayscale = rgb.mean(axis=2)
    transparent_mask = grayscale >= 245
    if transparent_mask.any():
        rgba = rgba.copy()
        rgba[transparent_mask, 3] = 0

    min_x, max_x, min_y, max_y = polygon_bounds(polygon)
    ax.imshow(
        rgba,
        extent=(min_x, max_x, min_y, max_y),
        origin="upper",
        alpha=alpha,
        zorder=20,
        interpolation="bilinear",
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
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=color_scale.levels,
        cmap=color_scale.cmap,
        norm=color_scale.norm,
        zorder=1,
    )
    _draw_floorplan(ax, polygon, floorplan_array)
    cbar = fig.colorbar(
        contour,
        ax=ax,
        shrink=0.5,
        pad=0.08,
        label="Elevation",
        extend="both",
    )
    cbar.formatter = ticker.FormatStrFormatter('%.1f')
    cbar.update_ticks()
    cbar.formatter = ticker.FormatStrFormatter('%.1f')
    cbar.update_ticks()
    ax.add_collection(
        PatchCollection([MplPolygon(polygon)], facecolor="none", edgecolor="black", linewidth=1.5, zorder=6)
    )
    _annotate_points(ax, points)
    min_x, max_x, min_y, max_y = polygon_bounds(polygon)
    pad = max(max_x - min_x, max_y - min_y) * 0.1
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Left to Right')
    ax.set_ylabel('Bottom to Top')
    ax.set_aspect("equal", adjustable="box")
    return to_base64(fig)


def plot_repair_plan(
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
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=color_scale.levels,
        cmap=color_scale.cmap,
        norm=color_scale.norm,
        zorder=1,
    )
    _draw_floorplan(ax, polygon, floorplan_array)
    contour_lines = ax.contour(
        grid_x,
        grid_y,
        grid_z,
        levels=color_scale.levels,
        colors="#1f2937",
        linewidths=0.8,
        zorder=6.5,
    )
    ax.clabel(contour_lines, fmt="%.1f", fontsize=7, inline=True)
    cbar = fig.colorbar(
        contour,
        ax=ax,
        shrink=0.5,
        pad=0.08,
        label="Elevation",
        extend="both",
    )
    cbar.formatter = ticker.FormatStrFormatter('%.1f')
    cbar.update_ticks()
    cbar.formatter = ticker.FormatStrFormatter('%.1f')
    cbar.update_ticks()
    ax.add_collection(
        PatchCollection([MplPolygon(polygon)], facecolor="none", edgecolor="#0f172a", linewidth=2, zorder=6)
    )
    for line in profile_lines:
        (x1, y1) = line["start"]
        (x2, y2) = line["end"]
        ax.plot([x1, x2], [y1, y2], color="white", linestyle="--", linewidth=1.1, alpha=0.35, zorder=7)
    _annotate_points(ax, points)
    min_x, max_x, min_y, max_y = polygon_bounds(polygon)
    pad = max(max_x - min_x, max_y - min_y) * 0.1
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Left to Right')
    ax.set_ylabel('Bottom to Top')
    ax.set_aspect("equal", adjustable="box")
    return to_base64(fig)


def plot_profiles(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    polygon: np.ndarray,
    profile_lines: Sequence[Dict[str, Tuple[float, float]]],
    points: Sequence[Point3D],
    boundary: Sequence[BoundaryPoint],
    title: str,
    color_scale: ColorScale,
    floorplan_array: Optional[np.ndarray] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=color_scale.levels,
        cmap=color_scale.cmap,
        norm=color_scale.norm,
        alpha=0.45,
        zorder=1,
    )
    _draw_floorplan(ax, polygon, floorplan_array)
    ax.add_collection(
        PatchCollection([MplPolygon(polygon)], facecolor="none", edgecolor="#2563eb", linewidth=2, zorder=6)
    )
    for line in profile_lines:
        (x1, y1) = line["start"]
        (x2, y2) = line["end"]
        ax.plot([x1, x2], [y1, y2], color="#111827", linewidth=0.7, alpha=0.55, zorder=6.5)
    boundary_coords = [(p.x, p.y) for p in boundary]
    ax.plot(
        [pt[0] for pt in boundary_coords + boundary_coords[:1]],
        [pt[1] for pt in boundary_coords + boundary_coords[:1]],
        color='#0f172a',
        linewidth=2,
        zorder=7,
    )
    _annotate_points(ax, points)
    cbar = fig.colorbar(
        contour,
        ax=ax,
        shrink=0.5,
        pad=0.08,
        label="Elevation",
        extend="both",
    )
    cbar.formatter = ticker.FormatStrFormatter('%.1f')
    cbar.update_ticks()
    cbar.formatter = ticker.FormatStrFormatter('%.1f')
    cbar.update_ticks()
    min_x, max_x, min_y, max_y = polygon_bounds(polygon)
    pad = max(max_x - min_x, max_y - min_y) * 0.1
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Left to Right')
    ax.set_ylabel('Bottom to Top')
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

        boundary_coords: List[Tuple[float, float]] = [(p.x, p.y) for p in payload.boundary]
        if boundary_coords[0] != boundary_coords[-1]:
            boundary_coords.append(boundary_coords[0])

        boundary_points = np.array(boundary_coords, dtype=float)
        measurement_points = np.array([[p.x, p.y] for p in payload.points], dtype=float)
        z_values = np.array([p.z for p in payload.points], dtype=float)

        boundary_samples = _generate_boundary_samples(boundary_points, max(payload.spacing, 0.1))
        if boundary_samples:
            tree = cKDTree(measurement_points)
            sample_points = np.array(boundary_samples, dtype=float)
            _, nearest_idx = tree.query(sample_points)
            sample_z = z_values[nearest_idx]
            interp_points = np.vstack([measurement_points, sample_points])
            interp_values = np.concatenate([z_values, sample_z])
        else:
            interp_points = measurement_points
            interp_values = z_values

        min_x, max_x, min_y, max_y = polygon_bounds(polygon)
        grid_x, grid_y = np.meshgrid(
            np.linspace(min_x, max_x, self.grid_resolution),
            np.linspace(min_y, max_y, self.grid_resolution),
        )

        interpolated = _interpolate_surface(interp_points, interp_values, grid_x, grid_y)
        logger.info(_summarize_grid('Interpolated grid', interpolated))

        mask = ~path.contains_points(np.vstack((grid_x.flatten(), grid_y.flatten())).T)
        mask = mask.reshape(grid_x.shape)
        interpolated = np.where(mask, np.nan, interpolated)
        masked_grid = np.ma.array(interpolated, mask=mask)
        logger.info('Masked cells=%d', int(mask.sum()))
        logger.info(_summarize_grid('Masked grid', masked_grid.filled(np.nan)))
        print(f"[FAME_UI] Masked cells={int(mask.sum())}")
        print("[FAME_UI] %s" % _summarize_grid('Masked grid', masked_grid.filled(np.nan)))

        profile_lines = _generate_profile_lines_from_boundary(payload.boundary, path)
        logger.info('Generated %d profile lines', len(profile_lines))
        print(f"[FAME_UI] Generated {len(profile_lines)} profile lines")
        floorplan_array = _decode_floorplan(payload.floorplan_image)
        if floorplan_array is None:
            logger.info('No floorplan image supplied or decoding failed')
        else:
            logger.info('Decoded floorplan image with shape %s', floorplan_array.shape)
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
                payload.points,
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
                payload.boundary,
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


@app.route("/api/fame/run", methods=["POST", "OPTIONS"])
def run_analysis():
    if request.method == "OPTIONS":
        return _corsify_response(make_response("", 204))

    try:
        logger.info('run_analysis invoked from %s', request.remote_addr)
        payload = AnalysisPayload.from_request(request.get_json(force=True))
        result = pipeline.run(payload)

        response = jsonify({
            "images": result.images,
            "profileLines": result.profile_lines,
            "unit": payload.unit,
            "version": BACKEND_REVISION_TAG,
        })
        response.status_code = 200
        return _corsify_response(response)
    except Exception as exc:  # pragma: no cover - top-level guard
        logger.exception('run_analysis failed: %s', exc)
        response = jsonify({
            "error": str(exc),
            "version": BACKEND_REVISION_TAG,
        })
        response.status_code = 400
        return _corsify_response(response)


if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)
