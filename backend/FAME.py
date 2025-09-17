import base64
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from itertools import product
from time import sleep
import gspread
import re
import math
import matplotlib.patheffects as PathEffects
from googleapiclient.discovery import build
from google.auth import default
from io import BytesIO
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from google.oauth2 import service_account
from google.oauth2.service_account import Credentials
from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import json
import io
from PIL import Image
from flask import jsonify
import googleapiclient.discovery
from functools import lru_cache
import threading
import time

# Performance optimizations
plt.ioff()  # Turn off interactive mode
np.seterr(all='ignore')  # Suppress numpy warnings for performance

def to_native_types(data):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, list):
        return [to_native_types(item) for item in data]
    elif isinstance(data, dict):
        return {key: to_native_types(value) for key, value in data.items()}
    else:
        return data

class ExcelCalculator:
    """Replicate Excel 4-EFE calculations in Python"""
    
    def __init__(self, effective_length: float = 20.0):
        self.effective_length = effective_length  # P3 value (from FP0!K10)
        
    def calculate_line_profile(self, 
                              start_x: float, start_y: float,  # E10, F10
                              end_x: float, end_y: float,      # J10, K10
                              lengths: np.ndarray,              # D5:P5
                              z_values: np.ndarray) -> dict:
        """Calculate all profile analysis for a single line"""
        results = {}
        
        # Calculate basic parameters (cells E11, J11, O10, P11)
        e11_length = max(abs(end_x - start_x), abs(end_y - start_y))  # E11
        j11_width = min(abs(end_x - start_x), abs(end_y - start_y))   # J11
        o10_k_factor = math.sqrt(e11_length**2 + j11_width**2) / e11_length if e11_length > 0 else 1  # O10
        p11_limit = 360 / o10_k_factor if o10_k_factor > 0 else 0  # P11
        
        results['k_length'] = e11_length
        results['k_width'] = j11_width
        results['k_factor'] = o10_k_factor
        results['limit_360'] = p11_limit
        
        # Calculate straight line interpolation (row 6)
        # Excel formula for E6: $D$7-(E5/$J$3)*($D$7-(INDEX($D$7:$P$7, 1, MAX(($D$7:$P$7<>"")*COLUMN($D$7:$P$7)-COLUMN($D$7)+1))))
        # This means: first_z - (length/max_length) * (first_z - last_non_empty_z)
        if len(z_values) > 0 and len(lengths) > 0:
            first_z = z_values[0]  # D7
            
            # Find the last non-empty z value (equivalent to Excel's INDEX with MAX)
            last_z = first_z  # Default to first if no other values
            for z in reversed(z_values):
                if np.isfinite(z):
                    last_z = z
                    break
            
            max_length = np.max(lengths[np.isfinite(lengths)])  # J3
            
            # Calculate interpolated values (row 6)
            interpolated = []
            for i, length in enumerate(lengths):
                if i == 0:
                    # First point (D6 = D7)
                    interpolated.append(first_z)
                elif np.isfinite(length) and max_length > 0:
                    # Excel formula: first_z - (length/max_length) * (first_z - last_z)
                    interp_z = first_z - (length / max_length) * (first_z - last_z)
                    interpolated.append(interp_z)
                else:
                    interpolated.append(0)
            
            # Calculate deflections (actual - interpolated)
            deflections = []
            for i in range(len(z_values)):
                if i < len(interpolated) and np.isfinite(z_values[i]):
                    deflection = abs(z_values[i] - interpolated[i])
                    deflections.append(deflection)
                else:
                    deflections.append(0)
            
            # Calculate deflection ratios matching Excel's exact methodology
            # Excel calculates for ALL possible 3-point combinations (not just segments)
            all_deflection_calcs = []
            
            # Replicate Excel rows 30-315: test every 3-point combination
            # where point i is start, j is middle, k is end
            for i in range(len(lengths)):  # Start point (D column)
                for k in range(i + 2, len(lengths)):  # End point (must be at least 2 positions after start)
                    for j in range(i + 1, k):  # Middle point (between start and end)
                        # All indices must be valid
                        if i < len(z_values) and j < len(z_values) and k < len(z_values):
                            # Excel formula E30: =E$7-(D$7+(E$5/F$5)*(F$7-D$7))
                            # This is: MiddleZ - (StartZ + (MiddleLength/EndLength)*(EndZ-StartZ))
                            
                            # Get the actual z values
                            z_start = z_values[i]  # D7
                            z_middle = z_values[j]  # E7, F7, etc.
                            z_end = z_values[k]    # The end point's Z
                            
                            # Get lengths
                            length_start = lengths[i]  # D5
                            length_middle = lengths[j]  # E5, F5, etc.
                            length_end = lengths[k]    # The end point's length
                            
                            # Calculate segment length (H column: end - start)
                            segment_length = length_end - length_start
                            
                            # Only process if segment length >= effective_length (P3)
                            if segment_length >= self.effective_length:
                                # Calculate expected Z at middle point using linear interpolation
                                if (length_end - length_start) > 0:
                                    # Excel: D$7+(E$5/F$5)*(F$7-D$7) where F is the end column
                                    # This becomes: StartZ + ((MiddleLength-StartLength)/(EndLength-StartLength))*(EndZ-StartZ)
                                    fraction = (length_middle - length_start) / (length_end - length_start)
                                    expected_z = z_start + fraction * (z_end - z_start)
                                    
                                    # Calculate deflection (E column in Excel)
                                    deflection = z_middle - expected_z
                                    
                                    # Calculate L/deflection ratio (F column: ABS(segment_length*12/deflection))
                                    if abs(deflection) > 0.0001:  # Avoid division by zero
                                        ratio = abs(segment_length * 12 / deflection)
                                        
                                        all_deflection_calcs.append({
                                            'ratio': ratio,
                                            'pt1': i + 1,  # 1-indexed for display
                                            'pt2': j + 1,
                                            'pt3': k + 1,
                                            'deflection': abs(deflection),
                                            'length': segment_length
                                        })
            
            # Sort all calculations by ratio (ascending - smallest ratios are worst)
            deflection_details = sorted(all_deflection_calcs, key=lambda x: x['ratio']) if all_deflection_calcs else []
            
            # Get top 5 worst deflections (smallest L/deflection ratios)
            if deflection_details:
                # Extract the top 5 worst deflections (already sorted)
                for i in range(1, 6):
                    if i - 1 < len(deflection_details):
                        detail = deflection_details[i - 1]
                        results[f'd{i}a'] = detail['ratio']
                        
                        # Calculate percentage
                        # M63 formula: =ROUND(P11/M37*100,0)-100 (for exceed case - shows excess %)
                        # M64 formula: =ROUND(P11/M37*100,0) (for pass case - shows usage %)
                        # The sheet displays these divided by 100 (e.g., 194 becomes 1.94)
                        if detail['ratio'] > 0:
                            # Don't round until final display - keep full precision
                            raw_percentage = (p11_limit / detail['ratio']) * 100
                            if detail['ratio'] > p11_limit:
                                # PASS case - using X% of allowable
                                percent_value = raw_percentage
                            else:
                                # FAIL case - exceeds by X%
                                percent_value = raw_percentage - 100
                            # Store as decimal for display (divide by 100)
                            percent_of_allowable = percent_value / 100
                        else:
                            percent_of_allowable = 0
                        
                        results[f'd{i}per'] = percent_of_allowable
                        
                        # Determine pass/fail
                        if detail['ratio'] > p11_limit:
                            results[f'exceeds{i}'] = 'NO'  # PASS
                        else:
                            results[f'exceeds{i}'] = 'YES'  # FAIL
                        
                        # Store point indices
                        results[f'd{i}pt1'] = detail['pt1']
                        results[f'd{i}pt2'] = detail['pt2']
                        results[f'd{i}pt3'] = detail['pt3']
                    else:
                        results[f'd{i}a'] = 0
                        results[f'd{i}per'] = 0
                        results[f'exceeds{i}'] = 'NO'
                        results[f'd{i}pt1'] = 0
                        results[f'd{i}pt2'] = 0
                        results[f'd{i}pt3'] = 0
            else:
                # No deflections found
                for i in range(1, 6):
                    results[f'd{i}a'] = 0
                    results[f'd{i}per'] = 0
                    results[f'exceeds{i}'] = 'NO'
                    results[f'd{i}pt1'] = 0
                    results[f'd{i}pt2'] = 0
                    results[f'd{i}pt3'] = 0
            
            # Calculate tilt (E20)
            # Excel formulas in row 26: =iferror(100*ABS(E7-$D7)/(12*(E5-$D5)),0)
            # E20 formula selects the tilt value for the LAST non-empty data point
            # The IF chain goes from right to left checking for non-empty cells
            final_tilt = 0
            if len(z_values) > 1 and len(lengths) > 1:
                first_z = z_values[0]  # D7
                first_length = lengths[0]  # D5
                
                # Find the last valid data point and calculate its tilt
                for i in range(len(z_values) - 1, 0, -1):  # Start from the end
                    if i < len(lengths) and np.isfinite(z_values[i]) and np.isfinite(first_z):
                        length_diff = lengths[i] - first_length
                        if length_diff > 0:
                            z_diff = abs(z_values[i] - first_z)
                            final_tilt = (100 * z_diff) / (12 * length_diff)
                            break  # Found the last valid point, stop
            
            # Store tilt with full precision - no rounding yet
            results['tilt'] = final_tilt
            
            # Check tilt exceeds
            if final_tilt <= 1.0:
                results['exceeds6'] = 'NO'
                results['tilt_percent'] = final_tilt * 100  # No rounding yet
            else:
                results['exceeds6'] = 'YES'
                results['tilt_percent'] = (final_tilt - 1.0) * 100  # No rounding yet
            
            # Generate text messages (matching Excel formulas)
            for i in range(1, 6):
                if results[f'exceeds{i}'] == 'YES':  # FAIL case
                    # Round only for display in text message
                    exceed_percent = int(round(results[f'd{i}per'] * 100))
                    results[f'text{i}'] = f"EXCEEDS THE kL/360 (L/{int(round(p11_limit))}) LIMIT BY {exceed_percent}%"
                else:  # PASS case
                    # Round only for display in text message
                    using_percent = int(round(results[f'd{i}per'] * 100))
                    results[f'text{i}'] = f"USING {using_percent}% OF THE ALLOWABLE kL/360 (L/{int(round(p11_limit))}) LIMIT"
            
            if results['exceeds6'] == 'YES':
                results['tilt_text'] = f"EXCEEDS ALLOWABLE 1% LIMIT BY {int(round(results['tilt_percent']))}%"
            else:
                results['tilt_text'] = f"USING {int(round(results['tilt_percent']))}% OF THE ALLOWABLE 1% LIMIT"
        
        else:
            # No data case
            for i in range(1, 6):
                results[f'd{i}a'] = 0
                results[f'd{i}per'] = 0
                results[f'exceeds{i}'] = 'NO'
                results[f'd{i}pt1'] = 0
                results[f'd{i}pt2'] = 0
                results[f'd{i}pt3'] = 0
                results[f'text{i}'] = 'NO DATA'
            
            results['tilt'] = 0
            results['exceeds6'] = 'NO'
            results['tilt_text'] = 'NO DATA'
            results['tilt_percent'] = 0
        
        return results

    def batch_calculate_lines(self, lines_data: list) -> list:
        """Process multiple lines in batch"""
        results = []
        
        for line in lines_data:
            line_results = self.calculate_line_profile(
                start_x=line['start_x'],
                start_y=line['start_y'],
                end_x=line['end_x'],
                end_y=line['end_y'],
                lengths=np.array(line['lengths']),
                z_values=np.array(line['z_values'])
            )
            line_results['line_name'] = line['line_name']
            results.append(line_results)
        
        return results

class FAMEOptimizer:
    """Optimized FAME processing with caching and connection pooling"""
    
    def __init__(self):
        self.service_cache = {}
        self.credentials_cache = {}
        self.floorplan_cache = {}
        self.auth_lock = threading.Lock()
        
    @lru_cache(maxsize=32)
    def get_cached_credentials(self, bucket_name, key_file_name):
        """Cache credentials to avoid repeated downloads"""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(key_file_name)
        json_key_content = blob.download_as_string()
        return json.loads(json_key_content)
    
    def get_drive_service(self, json_key, user_to_impersonate='noble@noble-pi.com'):
        """Get or create cached Drive service"""
        service_key = f"drive_{user_to_impersonate}"
        
        with self.auth_lock:
            if service_key not in self.service_cache:
                SCOPES = ['https://www.googleapis.com/auth/drive']
                credentials = service_account.Credentials.from_service_account_info(
                    json_key, scopes=SCOPES)
                credentials = credentials.with_subject(user_to_impersonate)
                self.service_cache[service_key] = build('drive', 'v3', credentials=credentials)
            
        return self.service_cache[service_key]
    
    def get_sheets_service(self, json_key):
        """Get or create cached Sheets service"""
        service_key = "sheets"
        
        with self.auth_lock:
            if service_key not in self.service_cache:
                scope = ['https://www.googleapis.com/auth/spreadsheets']
                credentials = ServiceAccountCredentials.from_json_keyfile_dict(json_key, scope)
                self.service_cache[service_key] = gspread.authorize(credentials)
            
        return self.service_cache[service_key]
    
    def get_floorplan_image(self, floorplan_id, service, width, height):
        """Cache floorplan images to avoid repeated downloads"""
        if floorplan_id not in self.floorplan_cache:
            request = service.files().get_media(fileId=floorplan_id)
            floorplan_buf = io.BytesIO()
            downloader = MediaIoBaseDownload(floorplan_buf, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            floorplan_buf.seek(0)
            floorplan_image = Image.open(floorplan_buf)
            floorplan_image = floorplan_image.resize((width, height), Image.LANCZOS)
            self.floorplan_cache[floorplan_id] = np.array(floorplan_image)
        
        return self.floorplan_cache[floorplan_id]

# Global optimizer instance
optimizer = FAMEOptimizer()

def setup_2d_plot_optimized(ax, xi, yi, zi, custom_cmap, title, min_z_rounded, max_z_rounded, 
                           x, y, z, point_id, flag, global_vars):
    """Setup 2D plot with original styling"""
    
    if flag == 'R':
        custom_cmap = adjust_colormap_alpha(custom_cmap, alpha=0.3)
    
    min_zi = np.nanmin(zi)
    max_zi = np.nanmax(zi)
    num_steps = int((max_zi - min_zi) / 0.1) + 1
    
    if num_steps <= 20:
        levels = np.arange(min_z_rounded - 0.05, max_z_rounded + 0.1, 0.1)
        if 0 not in levels:
            levels = np.sort(np.append(levels, [0 - 0.05, 0, 0 + 0.05]))
    else:
        levels = np.linspace(min_z_rounded, max_z_rounded, 20)
    
    if 0 not in levels:
        levels = np.sort(np.append(levels, 0))
    
    contourf = ax.contourf(xi, yi, zi, levels=levels, cmap=custom_cmap)
    
    if flag != 'C':
        global_vars['contourf'] = contourf
    else:
        global_vars['contourf_comp'] = contourf
    
    if flag == 'B' or flag == 'R' or flag == 'C':
        contour = ax.contour(xi, yi, zi, levels=levels, colors='dimgray', linewidths=1.0, linestyles='solid')
        labels = ax.clabel(contour, inline=True, fontsize=10, fmt='%1.1f')
        for label in labels:
            label.set_weight('bold')
    
    if flag != 'O':
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.margins(0)
        ax.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        ticks = np.arange(min_z_rounded, max_z_rounded + 0.2, 0.2)
        global_vars['colorbar'] = ax.figure.colorbar(contourf, ax=ax, ticks=ticks, label='Elevation Plot', shrink=0.4)
    
    ax.set_aspect('equal')
    ax.invert_yaxis()

def adjust_colormap_alpha(cmap, alpha, N=256):
    """Adjust colormap alpha as in original"""
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    colors = cmap(np.linspace(0, 1, N))
    colors[:, 3] = alpha

    new_cmap = LinearSegmentedColormap.from_list("adjusted_cmap", colors, N=N)
    return new_cmap

def _create_boundary_path(boundary_points):
    boundary_array = np.asarray(boundary_points, dtype=np.float32)
    if len(boundary_array) == 0:
        return None, boundary_array

    if not np.array_equal(boundary_array[0], boundary_array[-1]):
        boundary_array = np.vstack([boundary_array, boundary_array[0]])

    path = Path(boundary_array)
    return path, boundary_array

def _apply_boundary_clip(ax, boundary_points):
    path, boundary_array = _create_boundary_path(boundary_points)
    if path is None:
        return boundary_array

    clip_patch = PathPatch(path, transform=ax.transData, facecolor='none')
    ax.add_patch(clip_patch)

    for artist in list(ax.collections) + ax.lines + ax.patches:
        try:
            artist.set_clip_path(clip_patch)
        except AttributeError:
            continue

    ax.set_xlim(boundary_array[:, 0].min(), boundary_array[:, 0].max())
    ax.set_ylim(boundary_array[:, 1].max(), boundary_array[:, 1].min())
    return boundary_array

def _generate_boundary_samples(boundary_points, spacing):
    if spacing <= 0:
        return []

    samples = []
    boundary_array = np.asarray(boundary_points, dtype=np.float32)
    if len(boundary_array) < 2:
        return samples

    for idx in range(len(boundary_array) - 1):
        start = boundary_array[idx]
        end = boundary_array[idx + 1]
        segment = end - start
        length = np.hypot(segment[0], segment[1])
        if length == 0:
            continue

        num_points = max(int(np.floor(length / spacing)), 1)
        for t in np.linspace(0, 1, num_points, endpoint=False):
            samples.append((float(start[0] + t * segment[0]),
                            float(start[1] + t * segment[1])))
    return samples

def _figure_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def _process_initial_graphics_payload(request_data):
    boundary_data = request_data.get('boundary', [])
    point_data = request_data.get('points', [])
    spacing = float(request_data.get('spacing', 1))
    unit = request_data.get('unit', 'ft')

    if len(boundary_data) < 3 or len(point_data) == 0:
        return jsonify({'error': 'Boundary and points data are required to generate graphics.'}), 400

    boundary_array = np.array([(float(p['x']), float(p['y'])) for p in boundary_data], dtype=np.float32)
    points_array = np.array([(float(p['x']), float(p['y']), float(p['z'])) for p in point_data], dtype=np.float32)

    min_x = min(boundary_array[:, 0].min(), points_array[:, 0].min())
    min_y = min(boundary_array[:, 1].min(), points_array[:, 1].min())

    boundary_array[:, 0] -= min_x
    boundary_array[:, 1] -= min_y

    points_array[:, 0] -= min_x
    points_array[:, 1] -= min_y

    x = points_array[:, 0].astype(np.float32)
    y = points_array[:, 1].astype(np.float32)
    z = points_array[:, 2].astype(np.float32)

    grid_size = max(int(max(boundary_array[:, 0].ptp(), boundary_array[:, 1].ptp()) * (2 / max(spacing, 0.1))), 50)
    xi, yi, zi = optimized_interpolate_grid(x, y, z, grid_size=grid_size, method='linear' if len(point_data) < 10 else 'cubic')

    min_z = float(np.nanmin(zi))
    max_z = float(np.nanmax(zi))
    min_z_rounded = np.floor(min_z * 10) / 10 - 0.3
    max_z_rounded = np.ceil(max_z * 10) / 10 + 0.3

    zero_norm = (0 - min_z_rounded) / (max_z_rounded - min_z_rounded) if max_z_rounded != min_z_rounded else 0.5
    cdict = {
        'red': [(0.0, 1.0, 1.0), (zero_norm, 1.0, 1.0), (1.0, 0.0, 0.0)],
        'green': [(0.0, 0.0, 0.0), (zero_norm, 1.0, 1.0), (1.0, 1.0, 1.0)],
        'blue': [(0.0, 0.0, 0.0), (zero_norm, 1.0, 1.0), (1.0, 0.0, 0.0)]
    }
    custom_cmap = LinearSegmentedColormap('payload_custom_cmap', cdict)

    boundary_path, boundary_closed = _create_boundary_path(boundary_array)
    boundary_samples = _generate_boundary_samples(boundary_closed, spacing)

    points_labels = [p.get('label', f'Point {idx + 1}') for idx, p in enumerate(point_data)]

    plots = {}
    for name in ['Mesh Contour A', 'Mesh Contour B', 'Mesh All Profiles']:
        fig, ax = plt.subplots(figsize=(8, 8))
        plots[name] = {'fig': fig, 'ax': ax}

    global_vars = {'min_z_rounded': min_z_rounded, 'max_z_rounded': max_z_rounded}

    setup_2d_plot_optimized(
        plots['Mesh Contour A']['ax'], xi, yi, zi, custom_cmap,
        'Mesh Contour A', min_z_rounded, max_z_rounded,
        x, y, z, np.array(points_labels), 'A', global_vars
    )
    contour_a_boundary = _apply_boundary_clip(plots['Mesh Contour A']['ax'], boundary_closed)
    plots['Mesh Contour A']['ax'].plot(contour_a_boundary[:, 0], contour_a_boundary[:, 1], color='black', linewidth=1.0)
    plots['Mesh Contour A']['ax'].scatter(x, y, c='white', edgecolors='black', s=50, zorder=5)

    setup_2d_plot_optimized(
        plots['Mesh Contour B']['ax'], xi, yi, zi, custom_cmap,
        'Mesh Contour B', min_z_rounded, max_z_rounded,
        x, y, z, np.array(points_labels), 'B', global_vars
    )
    contour_b_boundary = _apply_boundary_clip(plots['Mesh Contour B']['ax'], boundary_closed)
    plots['Mesh Contour B']['ax'].plot(contour_b_boundary[:, 0], contour_b_boundary[:, 1], color='black', linewidth=1.0)
    plots['Mesh Contour B']['ax'].scatter(x, y, c='white', edgecolors='black', s=50, zorder=5)

    setup_2d_plot_optimized(
        plots['Mesh All Profiles']['ax'], xi, yi, zi, custom_cmap,
        'Mesh All Profiles', min_z_rounded, max_z_rounded,
        x, y, z, np.array(points_labels), 'yes', global_vars
    )
    profiles_boundary = _apply_boundary_clip(plots['Mesh All Profiles']['ax'], boundary_closed)
    plots['Mesh All Profiles']['ax'].plot(profiles_boundary[:, 0], profiles_boundary[:, 1], color='black', linewidth=1.0)

    if boundary_samples:
        samples_x, samples_y = zip(*boundary_samples)
        plots['Mesh All Profiles']['ax'].scatter(samples_x, samples_y, c='yellow', edgecolors='black', s=30, zorder=6)

    plots['Mesh All Profiles']['ax'].scatter(x, y, c='white', edgecolors='black', s=60, zorder=7)

    encoded_graphics = []
    naming_map = {
        'Mesh Contour A': '1 - Elevation Plot',
        'Mesh Contour B': '2 - Contours Mesh',
        'Mesh All Profiles': '3 - All Profiles'
    }

    for name, meta in plots.items():
        image_b64 = _figure_to_base64(meta['fig'])
        encoded_graphics.append({
            'name': naming_map.get(name, name),
            'image': f'data:image/png;base64,{image_b64}'
        })

    response_payload = {
        'unit': unit,
        'spacing': spacing,
        'boundary': request_data.get('boundary'),
        'points': request_data.get('points'),
        'graphics': encoded_graphics
    }

    return jsonify(response_payload)

def add_profile_lines_optimized(ax, boundary_points, left_boundary, right_boundary, 
                               bottom_boundary, top_boundary):
    """Add profile lines with original logic"""
    count = 0
    
    print(f"\nDEBUG: Boundary Analysis:")
    print(f"  Total boundary points: {len(boundary_points)}")
    print(f"  Left boundary points: {len(left_boundary)}")
    print(f"  Right boundary points: {len(right_boundary)}")
    print(f"  Top boundary points: {len(top_boundary)}")
    print(f"  Bottom boundary points: {len(bottom_boundary)}")
    
    bottom_left_corner = bottom_boundary[0]
    bottom_right_corner = bottom_boundary[-1]
    top_left_corner = top_boundary[0]
    top_right_corner = top_boundary[-1]
    
    evaluated_lines2 = set()
    valid_pairs = []
    
    for index_pair in product(range(len(boundary_points)), repeat=2):
        index1, index2 = index_pair
        start, end = boundary_points[index1], boundary_points[index2]
        
        checker_start = ''
        checker_end = ''
        
        if start == top_right_corner:
            checker_start = 'TR'
        elif start == top_left_corner:
            checker_start = 'TL'
        elif start == bottom_right_corner:
            checker_start = 'BR'
        elif start == bottom_left_corner:
            checker_start = 'BL'
        
        if end == top_right_corner:
            checker_end = 'TR'
        elif end == top_left_corner:
            checker_end = 'TL'
        elif end == bottom_right_corner:
            checker_end = 'BR'
        elif end == bottom_left_corner:
            checker_end = 'BL'
        
        if checker_start == '':
            if start in top_boundary:
                checker_start = 'TOP'
            elif start in bottom_boundary:
                checker_start = 'BOTTOM'
            elif start in left_boundary:
                checker_start = 'LEFT'
            elif start in right_boundary:
                checker_start = 'RIGHT'
        
        if checker_end == '':
            if end in top_boundary:
                checker_end = 'TOP'
            elif end in bottom_boundary:
                checker_end = 'BOTTOM'
            elif end in left_boundary:
                checker_end = 'LEFT'
            elif end in right_boundary:
                checker_end = 'RIGHT'
        
        # Apply all original filtering conditions
        if start == end:
            continue
        if checker_start == checker_end:
            continue
        if checker_start == 'TR' and (checker_end == 'RIGHT' or checker_end == 'TOP'):
            continue
        if checker_start == 'TL' and (checker_end == 'LEFT' or checker_end == 'TOP'):
            continue
        if checker_start == 'BR' and (checker_end == 'RIGHT' or checker_end == 'BOTTOM'):
            continue
        if checker_start == 'BL' and (checker_end == 'LEFT' or checker_end == 'BOTTOM'):
            continue
        if checker_end == 'TR' and (checker_start == 'RIGHT' or checker_start == 'TOP'):
            continue
        if checker_end == 'TL' and (checker_start == 'LEFT' or checker_start == 'TOP'):
            continue
        if checker_end == 'BR' and (checker_start == 'RIGHT' or checker_start == 'BOTTOM'):
            continue
        if checker_end == 'BL' and (checker_start == 'LEFT' or checker_start == 'BOTTOM'):
            continue
        if checker_start == 'TL' and checker_end == 'TR':
            continue
        if checker_start == 'TL' and checker_end == 'BL':
            continue
        if checker_start == 'TR' and checker_end == 'TL':
            continue
        if checker_start == 'TR' and checker_end == 'BR':
            continue
        if checker_start == 'BL' and checker_end == 'BR':
            continue
        if checker_start == 'BR' and checker_end == 'BL':
            continue
        if checker_start == 'BL' and checker_end == 'TL':
            continue
        if checker_start == 'BR' and checker_end == 'TR':
            continue
        
        line_id = tuple(sorted([start, end], key=lambda x: (x[0], x[1])))
        if line_id not in evaluated_lines2:
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.5)
            line_name = "L" + str(count + 1)  # Generate name with count+1 like FAME.py
            valid_pairs.append((line_name, (start, end)))
            evaluated_lines2.add(line_id)
            count += 1  # Increment after, like FAME.py
            
            # Debug every 10th line and lines near 80
            if count % 10 == 0 or count >= 78:
                print(f"DEBUG: Added line #{count}: {line_name} from {checker_start} to {checker_end}")
    
    print(f"\nDEBUG: Line Generation Summary:")
    print(f"  Total lines generated: {count}")
    print(f"  valid_pairs entries: {len(valid_pairs)}")
    print(f"  evaluated_lines2 entries: {len(evaluated_lines2)}")
    
    if count != 80:
        print(f"\nWARNING: Generated {count} lines, expected exactly 80!")
        if count == 81:
            print("\nAnalyzing line 81:")
            name, (start, end) = valid_pairs[80]
            print(f"  Line name: {name}")
            print(f"  Start: {start}")
            print(f"  End: {end}")
            # Check if this is a duplicate that wasn't caught
            for i in range(80):
                other_name, (other_start, other_end) = valid_pairs[i]
                if (abs(start[0] - other_start[0]) < 0.001 and abs(start[1] - other_start[1]) < 0.001 and 
                    abs(end[0] - other_end[0]) < 0.001 and abs(end[1] - other_end[1]) < 0.001):
                    print(f"  POTENTIAL DUPLICATE of {other_name}!")
                elif (abs(start[0] - other_end[0]) < 0.001 and abs(start[1] - other_end[1]) < 0.001 and 
                      abs(end[0] - other_start[0]) < 0.001 and abs(end[1] - other_start[1]) < 0.001):
                    print(f"  POTENTIAL REVERSE DUPLICATE of {other_name}!")
    
    return valid_pairs

def optimized_interpolate_grid(x, y, z, grid_size=100, method='cubic'):
    """Optimized grid interpolation with vectorization and safe fallbacks"""
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    xi = np.linspace(x_min, x_max, grid_size, dtype=np.float32)
    yi = np.linspace(y_min, y_max, grid_size, dtype=np.float32)
    xi, yi = np.meshgrid(xi, yi, sparse=False, copy=False)

    methods = [method]
    if method != 'linear':
        methods.append('linear')
    if method != 'nearest':
        methods.append('nearest')

    zi = None
    for interp_method in methods:
        try:
            zi_candidate = griddata((x, y), z, (xi, yi), method=interp_method)
        except Exception:
            zi_candidate = None

        if zi_candidate is None:
            continue

        if np.all(np.isnan(zi_candidate)):
            continue

        zi = zi_candidate
        break

    if zi is None:
        zi = griddata((x, y), z, (xi, yi), method='nearest')

    if np.all(np.isnan(zi)):
        zi = np.full_like(xi, np.nanmean(z, dtype=np.float32))

    zi = np.clip(np.nan_to_num(zi, nan=np.nanmean(z, dtype=np.float32)), z.min(), z.max())

    return xi, yi, zi

def optimized_calculate_lengths_and_z(start, end, xi, yi, zi, num_points):
    """Optimized calculation using vectorized operations"""
    t = np.linspace(0, 1, num_points, dtype=np.float32)
    points = np.column_stack([
        start[0] + t * (end[0] - start[0]),
        start[1] + t * (end[1] - start[1])
    ])
    
    z_values = griddata(
        (xi.ravel(), yi.ravel()), 
        zi.ravel(), 
        points, 
        method='linear',
        fill_value=0
    )
    
    lengths = np.sqrt((points[:, 0] - start[0])**2 + (points[:, 1] - start[1])**2)
    
    return points, lengths, z_values, points[:, 0], points[:, 1]

def setup_singular_plot_with_floorplan(ax, xi, yi, zi, custom_cmap, min_z_rounded, max_z_rounded,
                                       x, y, z, point_id, global_vars, optimizer, request_data, scale_factor):
    """Setup singular plot with floorplan already embedded like in FAME.py"""
    # First setup the base heatmap
    min_zi = np.nanmin(zi)
    max_zi = np.nanmax(zi)
    num_steps = int((max_zi - min_zi) / 0.1) + 1
    
    if num_steps <= 20:
        levels = np.arange(min_z_rounded - 0.05, max_z_rounded + 0.1, 0.1)
        if 0 not in levels:
            levels = np.sort(np.append(levels, [0 - 0.05, 0, 0 + 0.05]))
    else:
        levels = np.linspace(min_z_rounded, max_z_rounded, 20)
    
    if 0 not in levels:
        levels = np.sort(np.append(levels, 0))
    
    # Draw the contour
    contourf = ax.contourf(xi, yi, zi, levels=levels, cmap=custom_cmap)
    
    # Now add the floorplan overlay
    json_key = optimizer.get_cached_credentials('engineering-support-doc', 
                                               'engineering-414319-561d04e6c03a.json')
    service = optimizer.get_drive_service(json_key)
    
    floorplan_array = optimizer.get_floorplan_image(
        request_data['imageId'],
        service,
        request_data['width'],
        request_data['height']
    )
    
    x_offset = request_data['centerXShape'] - request_data['centerXCell']
    y_offset = request_data['centerYShape'] - request_data['centerYCell']
    
    offset_plot = (x_offset / 25) * scale_factor - ((12.5 / 25) * scale_factor)
    offset_plot_right = ((x_offset / 25) + (request_data['width'] / 25)) * scale_factor
    offset_y_plot = ((y_offset / 25) * scale_factor) - ((12.5 / 25) * scale_factor)
    offset_y_plot_bottom = ((y_offset / 25) + (request_data['height'] / 25)) * scale_factor
    
    ax.set_xlim([global_vars['min_x'], global_vars['max_x']])
    ax.set_ylim([global_vars['min_y'], global_vars['max_y']])
    
    ax.imshow(floorplan_array, 
             extent=[offset_plot, offset_plot_right, offset_y_plot_bottom, offset_y_plot],
             aspect='equal', zorder=10, interpolation='bilinear')
    
    # Setup axis properties
    tick_values = np.arange(np.floor(global_vars['min_y'] / 10) * 10, 
                           np.ceil(global_vars['max_y'] / 10) * 10, 10)
    inverted_ticks_labels = {global_vars['max_y'] - value + global_vars['min_y']: str(int(value)) 
                            for value in tick_values}
    
    ax.set_yticks(list(inverted_ticks_labels.keys()))
    ax.set_yticklabels(list(inverted_ticks_labels.values()))
    ax.invert_yaxis()
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.margins(0)
    ax.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_aspect('equal')

def save_singular_plot_and_upload(optimizer_instance, figure, name, folder_id):
    """Simple save and upload for singular plots without modifying the axis"""
    try:
        json_key = optimizer_instance.get_cached_credentials('engineering-support-doc', 
                                                             'engineering-414319-561d04e6c03a.json')
        service = optimizer_instance.get_drive_service(json_key)
        
        # Save figure to buffer without modifying
        buf_final = BytesIO()
        figure.savefig(buf_final, format='jpeg', bbox_inches='tight', pad_inches=0.1, 
                      dpi=300, facecolor='white')
        buf_final.seek(0)
        
        # Upload to Google Drive
        file_metadata = {
            'name': name,
            'parents': [folder_id],
            'mimeType': 'image/jpeg'
        }
        
        media = MediaIoBaseUpload(buf_final, mimetype='image/jpeg', resumable=True)
        
        # Check if file exists and update or create
        query = f"name='{name}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        
        if items:
            file_id = items[0]['id']
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            file_id = file.get('id')
        
        # Make the file publicly accessible
        if file_id:
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            try:
                service.permissions().create(fileId=file_id, body=permission).execute()
                print(f"Made {name} publicly accessible")
            except Exception as perm_error:
                print(f"Warning: Could not set public permissions for {name}: {perm_error}")
        
        buf_final.close()
        return file_id
        
    except Exception as e:
        print(f"Error saving singular plot: {e}")
        return None

def optimized_save_plot_and_upload(optimizer_instance, x, y, z, point_id, ax, figure, name, 
                                  folder_id, scale_factor, request_data, global_vars):
    """Optimized save and upload with caching and compression"""
    
    json_key = optimizer_instance.get_cached_credentials('engineering-support-doc', 
                                                         'engineering-414319-561d04e6c03a.json')
    service = optimizer_instance.get_drive_service(json_key)
    
    floorplan_array = optimizer_instance.get_floorplan_image(
        request_data['imageId'],
        service,
        request_data['width'],
        request_data['height']
    )
    
    x_offset = request_data['centerXShape'] - request_data['centerXCell']
    y_offset = request_data['centerYShape'] - request_data['centerYCell']
    
    offset_plot = (x_offset / 25) * scale_factor - ((12.5 / 25) * scale_factor)
    offset_plot_right = ((x_offset / 25) + (request_data['width'] / 25)) * scale_factor
    offset_y_plot = ((y_offset / 25) * scale_factor) - ((12.5 / 25) * scale_factor)
    offset_y_plot_bottom = ((y_offset / 25) + (request_data['height'] / 25)) * scale_factor
    
    ax.set_xlim([global_vars['min_x'], global_vars['max_x']])
    ax.set_ylim([global_vars['min_y'], global_vars['max_y']])
    
    ax.imshow(floorplan_array, 
             extent=[offset_plot, offset_plot_right, offset_y_plot_bottom, offset_y_plot],
             aspect='equal', zorder=10, interpolation='bilinear')
    
    tick_values = np.arange(np.floor(global_vars['min_y'] / 10) * 10, 
                           np.ceil(global_vars['max_y'] / 10) * 10, 10)
    inverted_ticks_labels = {global_vars['max_y'] - value + global_vars['min_y']: str(int(value)) 
                            for value in tick_values}
    
    ax.set_yticks(list(inverted_ticks_labels.keys()))
    ax.set_yticklabels(list(inverted_ticks_labels.values()))
    ax.invert_yaxis()
    
    if "Comparison" in name:
        min_val = global_vars['min_z_rounded_comp']
        max_val = global_vars['max_z_rounded_comp']
        contourf = global_vars['contourf_comp']
    else:
        min_val = global_vars['min_z_rounded']
        max_val = global_vars['max_z_rounded']
        contourf = global_vars['contourf']
    
    min_tick = np.floor(min_val / 0.2) * 0.2
    max_tick = np.ceil(max_val / 0.2) * 0.2
    ticks = np.arange(min_tick, max_tick + 0.2, 0.2)
    
    cbar = figure.colorbar(contourf, ax=ax, ticks=ticks, shrink=.5)
    cbar.ax.set_yticklabels([f'{tick:.1f}' for tick in ticks])
    
    ax.set_xlabel('Left to Right', fontsize=12)
    ax.set_ylabel('Bottom to Top', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    if 'Elevation Plot' in name and point_id is not None:
        mask_valid = ~np.isin(point_id, ['O', 'S'])
        x_valid = x[mask_valid]
        y_valid = y[mask_valid]
        z_valid = z[mask_valid]
        id_valid = point_id[mask_valid]
        
        colors = np.where(z_valid > 0, 'green', 
                         np.where(z_valid < 0, '#8B0000', 'blue'))
        colors[np.isin(id_valid, ['G', 'C'])] = 'black'
        
        for xi, yi, zi, color, pid in zip(x_valid, y_valid, z_valid, colors, id_valid):
            ax.plot(xi, yi, 'o', color=color, markersize=2.5, zorder=11)
            
            if pid in ['G', 'C', 'B']:
                text_val = f'{zi:.1f} ({pid})'
            elif zi > 0:
                text_val = f'+{zi:.1f}'
            else:
                text_val = f'{zi:.1f}'
            
            ax.text(xi, yi + 1.5, text_val, ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color=color,
                   path_effects=[PathEffects.withStroke(linewidth=1, foreground="white")],
                   zorder=12)
    
    buf_final = io.BytesIO()
    figure.savefig(buf_final, format='jpeg', bbox_inches='tight', pad_inches=0.1, 
                  dpi=300, facecolor='white')
    buf_final.seek(0)
    
    file_metadata = {'name': name, 'parents': [folder_id]}
    media = MediaIoBaseUpload(buf_final, mimetype='image/jpeg')
    
    query = f"name = '{name}' and '{folder_id}' in parents and trashed = false"
    response = service.files().list(q=query, fields='files(id)').execute()
    for file in response.get('files', []):
        service.files().delete(fileId=file['id']).execute()
    
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    
    service.permissions().create(
        fileId=file_id,
        body={'type': 'anyone', 'role': 'reader'}
    ).execute()
    
    plt.close(figure)
    return file_id

def batch_process_all_lines(line_ids, xi, yi, zi, total_point_per_profile, 
                           effective_length, scale_factor, sa, spreadsheet_id,
                           plots, failed_profiles_id, x_all, y_all, z_all, point_id,
                           optimizer=None, request_data=None, global_vars=None):
    """Process all lines in memory and return batch results"""
    print(f"Starting batch processing of {len(line_ids)} lines...")
    start_time = time.time()
    
    calculator = ExcelCalculator(effective_length=effective_length)
    
    all_lines_data = []
    sheet = sa.open_by_key(spreadsheet_id)
    ws_config = sheet.worksheet('2-EFE')
    
    # Get all line configurations in one batch read
    all_cells = ws_config.get_all_values()
    config_data = {}
    
    for row_idx, row in enumerate(all_cells[6:88], start=7):  # Rows 7-88
        if row[1]:  # Column B has line name
            line_name = row[1]
            config_data[line_name] = {
                'row': row_idx,
                'pass_color': row[4] or '#000000',
                'pass_type': row[5] or 'Solid',
                'pass_weight': row[6] or '1',
                'fail_color': row[7] or '#FF0000',
                'fail_type': row[8] or 'Dashed',
                'fail_weight': row[9] or '1',
                'line_on_off': row[10] or 'Off',
                'tilt_fail_color': row[11] or '#FFA500',
                'tilt_fail_type': row[12] or 'Dashed',
                'tilt_fail_weight': row[13] or '1',
                'tilt_on_off': row[14] or 'Off',
            }
    
    # Process each line and collect data
    # Only process lines that were actually generated from boundary analysis
    print(f"\nProcessing {len(line_ids)} profile lines from boundary analysis")
    
    if len(line_ids) != 80:
        print(f"WARNING: Have {len(line_ids)} lines instead of expected 80")
    
    # Process only the lines that were actually generated
    for generated_name, (start, end) in line_ids:
        # Use the generated name directly - no remapping
        line_name = generated_name
        
        points, lengths, z_values, x_component, y_component = optimized_calculate_lengths_and_z(
            start, end, xi, yi, zi, total_point_per_profile
        )
        
        line_data = {
            'line_name': line_name,  # Using the generated name directly
            'start_x': start[0],
            'start_y': start[1],
            'end_x': end[0],
            'end_y': end[1],
            'lengths': to_native_types(lengths),
            'z_values': to_native_types(z_values),
            'x_component': to_native_types(x_component),
            'y_component': to_native_types(y_component),
            'config': config_data.get(line_name, {})
        }
        
        all_lines_data.append(line_data)
    
    # Batch calculate all lines in memory
    print(f"Calculating {len(all_lines_data)} lines in memory...")
    calculation_results = calculator.batch_calculate_lines(all_lines_data)
    
    # Prepare batch update data for Google Sheets
    batch_updates = []
    image_uploads_queue = []  # Collect image uploads to do after batch update
    
    for i, (line_data, calc_result) in enumerate(zip(all_lines_data, calculation_results)):
        line_name = line_data['line_name']
        config = line_data['config']
        
        if not config or (config.get('line_on_off') != 'On' and config.get('tilt_on_off') != 'On'):
            continue
        
        # Prepare row data for sheet update matching exact 2-EFE format
        # Correct column order based on headers:
        # 1. K-factor data (Length, Width, k-factor, Actual Length, Effective Length) - 5 columns
        # 2. Points 1-13 (X, Y, L, Z for each) - 52 columns
        # 3. Deflections 1-5 (Actual, %, Exceeds, P1, P2, P3 for each) - 30 columns  
        # 4. Tilt (%, Exceeds) - 2 columns
        # Total: 89 columns (matches P to DA range)
        
        row_data = []
        
        # 1. Add k-factor data (5 columns)
        row_data.extend([
            float(calc_result.get('k_length', 0)),
            float(calc_result.get('k_width', 0)),
            float(calc_result.get('k_factor', 0)),
            float(max(line_data['lengths']) if line_data['lengths'] else 0),
            float(effective_length)
        ])
        
        # 2. Add point data (13 points × 4 values = 52 columns)
        max_points = 13
        point_count = min(len(line_data['x_component']), max_points)
        
        for i in range(point_count):
            row_data.extend([
                float(line_data['x_component'][i]),
                float(line_data['y_component'][i]),
                float(line_data['lengths'][i]),
                float(line_data['z_values'][i])
            ])
        
        # Pad point data to exactly 52 columns
        points_added = point_count * 4
        while points_added < 52:
            row_data.append('')
            points_added += 1
        
        # 3. Add deflection results (30 values: 5 deflections × 6 values each)
        for j in range(1, 6):
            row_data.extend([
                int(round(float(calc_result.get(f'd{j}a', 0)))),  # Round to nearest integer only at final output
                round(float(calc_result.get(f'd{j}per', 0)), 2),  # Round percentage to 2 decimal places for display
                calc_result.get(f'exceeds{j}', 'NO'),
                int(calc_result.get(f'd{j}pt1', 0)),
                int(calc_result.get(f'd{j}pt2', 0)),
                int(calc_result.get(f'd{j}pt3', 0))
            ])
        
        # 4. Add tilt results (2 values)
        row_data.extend([
            round(float(calc_result.get('tilt', 0)), 2),  # Round tilt to 2 decimal places for display
            calc_result.get('exceeds6', 'NO')
        ])
        
        # Total should be exactly 89 columns (5 + 52 + 30 + 2)
        # But we need 90 for P-DA range, so add one empty column
        row_data.append('')
        
        # Check if Actual Length (column S, index 3) is less than 20
        actual_length = row_data[3]  # Column S (Actual Length) is at index 3
        
        # Add to batch update
        if config.get('row'):
            if actual_length < 20:
                # Only write column S (Actual Length) when Actual Length < 20
                # Column S is the 4th column (index 3) starting from P, so it's column S
                batch_updates.append({
                    'range': f'S{config["row"]}',
                    'values': [[actual_length]]  # Only the actual length value
                })
            else:
                # Write all columns P-DA when Actual Length >= 20
                batch_updates.append({
                    'range': f'P{config["row"]}:DA{config["row"]}',
                    'values': [row_data]
                })
        
        # Plot failed lines on graphics only if Actual Length >= 20
        if actual_length >= 20:
            if any(calc_result.get(f'exceeds{j}') == 'YES' for j in range(1, 6)):
                plot_line_on_heatmap(
                    plots.get('Mesh Deflection Exceeds', {}).get('ax'),
                    plots.get('Mesh Singular Plots', {}).get('ax'),
                    line_data, calc_result, config, 'DEFLECTION',
                    plots, failed_profiles_id, sa, spreadsheet_id, optimizer,
                    request_data, global_vars, x_all, y_all, z_all, point_id,
                    image_uploads_queue
                )
            
            if calc_result.get('exceeds6') == 'YES':
                plot_line_on_heatmap(
                    plots.get('Mesh Tilt Exceeds', {}).get('ax'),
                    plots.get('Mesh Singular Plots', {}).get('ax'),
                    line_data, calc_result, config, 'TILT',
                    plots, failed_profiles_id, sa, spreadsheet_id, optimizer,
                    request_data, global_vars, x_all, y_all, z_all, point_id,
                    image_uploads_queue
                )
    
    # Perform batch update to sheet
    print(f"Writing {len(batch_updates)} results to sheet in batch...")
    if batch_updates:
        ws_results = sheet.worksheet('2-EFE')
        ws_results.batch_update(batch_updates)
    
    # Now perform all image uploads AFTER the batch update
    if image_uploads_queue:
        print(f"Uploading {len(image_uploads_queue)} images to sheet...")
        for line_name, image_id in image_uploads_queue:
            upload_image_to_sheet(sa, spreadsheet_id, line_name, image_id)
    
    elapsed_time = time.time() - start_time
    print(f"Batch processing completed in {elapsed_time:.2f} seconds")
    
    return calculation_results

def upload_image_to_sheet(sa, sheet_id, line_name, image_id):
    """Upload image formula to the sheet for a specific line"""
    try:
        print(f"Uploading image for line '{line_name}' with image_id '{image_id}'")
        sheet = sa.open_by_key(sheet_id)
        ws = sheet.worksheet('2-EFE')
        
        # Use range like FAME.py does
        range_name = 'B:B'
        values = ws.range(range_name)
        values_list = [cell.value for cell in values]
        
        # Find the row number where line_name matches
        row_number = None
        for i, value in enumerate(values_list):
            if value == line_name:
                row_number = i + 1  # Google Sheets is 1-indexed
                break
        
        if row_number is None:
            print(f"Warning: Line name '{line_name}' not found in column B")
            return
        
        # Construct the public image URL
        image_url = f'https://drive.google.com/uc?export=view&id={image_id}'
        
        # Create the IMAGE formula for Google Sheets
        formula = f'=IMAGE("{image_url}")'
        
        # Update cell in column DA for this row
        cell_address = f'DA{row_number}'
        
        # Use update_acell like FAME.py
        ws.update_acell(cell_address, formula)
        print(f"Successfully updated {cell_address} with image formula for {line_name}")
        
    except Exception as e:
        print(f"Error uploading image to sheet for {line_name}: {e}")
        import traceback
        traceback.print_exc()

def plot_line_on_heatmap(ax, ax2, line_data, calc_result, config, plot_type, 
                        plots=None, failed_profiles_id=None, sa=None, sheet_id=None,
                        optimizer=None, request_data=None, global_vars=None,
                        x_all=None, y_all=None, z_all=None, point_id=None,
                        image_uploads_queue=None):
    """Plot a single line on the heatmap and generate individual failure images"""
    if not ax:
        return
    
    line_style_map = {
        "Solid": "-",
        "Dashed": "--",
        "Dotted": ":",
        "Dashdot": "-.",
    }
    
    # Lists to track plot elements for removal
    lines = []
    texts = []
    
    if plot_type == 'DEFLECTION':
        first_x = line_data['x_component'][0]
        first_y = line_data['y_component'][0]
        last_x = line_data['x_component'][-1]
        last_y = line_data['y_component'][-1]
        
        line_style = line_style_map.get(config.get('pass_type'), '-')
        ax.plot([first_x, last_x], [first_y, last_y], 
               linewidth=float(config.get('pass_weight', 1)),
               linestyle=line_style, 
               color=config.get('pass_color', 'black'))
        
        mid_x = (first_x + last_x) / 2
        mid_y = (first_y + last_y) / 2
        ax.text(mid_x, mid_y, line_data['line_name'], 
               fontsize=9, ha='center', fontweight='bold', va='center',
               path_effects=[PathEffects.withStroke(linewidth=1, foreground="white")])
        
        # Plot on singular plot for individual image
        if ax2:
            line = ax2.plot([first_x, last_x], [first_y, last_y], 
                   linewidth=float(config.get('pass_weight', 1)),
                   linestyle=line_style, 
                   color=config.get('pass_color', 'black'))
            text = ax2.text(mid_x, mid_y, line_data['line_name'], 
                   fontsize=9, ha='center', fontweight='bold', va='center',
                   path_effects=[PathEffects.withStroke(linewidth=1, foreground="white")])
            lines.append(line)
            texts.append(text)
        
        for i in range(1, 6):
            if calc_result.get(f'exceeds{i}') == 'YES':
                pt1 = int(calc_result.get(f'd{i}pt1', 1)) - 1
                pt3 = int(calc_result.get(f'd{i}pt3', 1)) - 1
                
                if 0 <= pt1 < len(line_data['x_component']) and 0 <= pt3 < len(line_data['x_component']):
                    fail_style = line_style_map.get(config.get('fail_type'), '--')
                    ax.plot([line_data['x_component'][pt1], line_data['x_component'][pt3]], 
                           [line_data['y_component'][pt1], line_data['y_component'][pt3]],
                           linewidth=float(config.get('fail_weight', 1)),
                           linestyle=fail_style,
                           color=config.get('fail_color', 'red'))
                    
                    # Plot on singular plot for individual image
                    if ax2:
                        line = ax2.plot([line_data['x_component'][pt1], line_data['x_component'][pt3]], 
                               [line_data['y_component'][pt1], line_data['y_component'][pt3]],
                               linewidth=float(config.get('fail_weight', 1)),
                               linestyle=fail_style,
                               color=config.get('fail_color', 'red'))
                        lines.append(line)
        
        # Generate and upload individual failure image
        if ax2 and plots and failed_profiles_id and optimizer and sa and sheet_id:
            singular_map_name = f"{line_data['line_name']} - Deflection.jpeg"
            image_id = save_singular_plot_and_upload(
                optimizer,
                plots['Mesh Singular Plots']['fig'],
                singular_map_name, 
                failed_profiles_id
            )
            if image_id and image_uploads_queue is not None:
                # Queue the upload instead of doing it immediately
                image_uploads_queue.append((line_data['line_name'], image_id))
            
            # Remove only the lines and text, not the background
            for line in lines:
                for l in line:  # Line may be a list of Line2D objects
                    l.remove()
            for text in texts:
                text.remove()
    
    elif plot_type == 'TILT':
        first_x = line_data['x_component'][0]
        first_y = line_data['y_component'][0]
        last_x = line_data['x_component'][-1]
        last_y = line_data['y_component'][-1]
        
        tilt_style = line_style_map.get(config.get('tilt_fail_type'), '--')
        ax.plot([first_x, last_x], [first_y, last_y],
               linewidth=float(config.get('tilt_fail_weight', 1)),
               linestyle=tilt_style,
               color=config.get('tilt_fail_color', 'orange'))
        
        mid_x = (first_x + last_x) / 2
        mid_y = (first_y + last_y) / 2
        ax.text(mid_x, mid_y, line_data['line_name'],
               fontsize=9, fontweight='bold', ha='center', va='center',
               path_effects=[PathEffects.withStroke(linewidth=1, foreground="white")])
        
        # Plot on singular plot for individual image
        if ax2:
            line = ax2.plot([first_x, last_x], [first_y, last_y],
                   linewidth=float(config.get('tilt_fail_weight', 1)),
                   linestyle=tilt_style,
                   color=config.get('tilt_fail_color', 'orange'))
            text = ax2.text(mid_x, mid_y, line_data['line_name'],
                   fontsize=9, fontweight='bold', ha='center', va='center',
                   path_effects=[PathEffects.withStroke(linewidth=1, foreground="white")])
            lines.append(line)
            texts.append(text)
        
        # Generate and upload individual tilt failure image
        if ax2 and plots and failed_profiles_id and optimizer and sa and sheet_id:
            singular_map_name = f"{line_data['line_name']} - Tilt.jpeg"
            image_id = save_singular_plot_and_upload(
                optimizer,
                plots['Mesh Singular Plots']['fig'],
                singular_map_name, 
                failed_profiles_id
            )
            if image_id and image_uploads_queue is not None:
                # Queue the upload instead of doing it immediately
                image_uploads_queue.append((line_data['line_name'], image_id))
            
            # Remove only the lines and text, not the background
            for line in lines:
                for l in line:  # Line may be a list of Line2D objects
                    l.remove()
            for text in texts:
                text.remove()

def script_run(input_data):
    """Main optimized script run function"""
    try:
        global_vars = {
            'min_x': None, 'max_x': None,
            'min_y': None, 'max_y': None,
            'min_z_rounded': None, 'max_z_rounded': None,
            'min_z_rounded_comp': None, 'max_z_rounded_comp': None,
            'contourf': None, 'contourf_comp': None,
            'color_check': None
        }
        
        request_data = input_data.get_json(silent=True)
        
        if request_data and 'boundary' in request_data and 'points' in request_data:
            return _process_initial_graphics_payload(request_data)

        json_key = optimizer.get_cached_credentials(
            'engineering-support-doc',
            'engineering-414319-561d04e6c03a.json'
        )
        
        sa = optimizer.get_sheets_service(json_key)
        
        spreadsheet_id = request_data['spreadsheetId']
        sheet = sa.open_by_key(spreadsheet_id)
        
        ws = sheet.worksheet('FP0')
        config = {
            'total_point_per_profile': int(ws.acell('K11').value),
            'effective_length': int(ws.acell('K10').value),
            'boundary_points': int(ws.acell('K12').value),
            'scale_factor': int(ws.acell('E10').value)
        }
        
        ws = sheet.worksheet('1-EFE')
        values = ws.get_all_values()
        headers = values[0]
        data = [dict(zip(headers, row)) for row in values[1:]]
        
        fp1_data = [d for d in data if d['FP1 ID'] not in ['G', 'C'] and d['FP1 Z (in)'] != '']
        x = np.array([float(d['FP1 X (ft)']) for d in fp1_data], dtype=np.float32)
        y = np.array([float(d['FP1 Y (ft)']) for d in fp1_data], dtype=np.float32)
        z = np.array([float(d['FP1 Z (in)']) for d in fp1_data], dtype=np.float32)
        
        x -= x.min()
        y -= y.min()
        
        xi, yi, zi = optimized_interpolate_grid(x, y, z, grid_size=100)
        
        all_data = [d for d in data if d['FP1 X (ft)'] != '']
        x_all = np.array([float(d['FP1 X (ft)']) for d in all_data], dtype=np.float32)
        y_all = np.array([float(d['FP1 Y (ft)']) for d in all_data], dtype=np.float32)
        z_all = np.array([float(d['FP1 Z (in)']) for d in all_data], dtype=np.float32)
        point_id = np.array([d['FP1 ID'] for d in all_data if d['FP1 Z (in)'] != ''])
        
        x_all -= x_all.min()
        y_all -= y_all.min()
        
        global_vars.update({
            'min_x': xi.min(),
            'max_x': xi.max(),
            'min_y': yi.min(),
            'max_y': yi.max(),
            'min_z_rounded': np.floor(zi.min() * 10) / 10 - 0.3,
            'max_z_rounded': np.ceil(zi.max() * 10) / 10 + 0.3,
            'xi': xi,
            'yi': yi,
            'zi': zi
        })
        
        fp2_data = [d for d in data if d['FP2 ID'] not in ['G', 'C'] 
                   and d['FP2 Z (in)'] != '' and d['FP2 Z (in)'] != 'NO DATA']
        
        comparison_flag = False
        custom_diff_cmap = None
        diff = None
        
        if fp2_data:
            x2 = np.array([float(d['FP2 X (ft)']) for d in fp2_data], dtype=np.float32)
            y2 = np.array([float(d['FP2 Y (ft)']) for d in fp2_data], dtype=np.float32)
            z2 = np.array([float(d['FP2 Z (in)']) for d in fp2_data], dtype=np.float32)
            
            x2 -= x2.min()
            y2 -= y2.min()
            
            xi2, yi2, zi2 = optimized_interpolate_grid(x2, y2, z2, grid_size=100)
            diff = zi - zi2
            
            global_vars['min_z_rounded_comp'] = np.min(diff)
            global_vars['max_z_rounded_comp'] = np.max(diff)
            comparison_flag = True
            
            min_diff = np.nanmin(diff)
            max_diff = np.nanmax(diff)
            
            if max_diff == min_diff:
                max_diff = min_diff + 1e-9
            
            norm_neg = np.clip((-0.1 - min_diff) / (max_diff - min_diff), 0, 1)
            norm_zero = np.clip((0.0 - min_diff) / (max_diff - min_diff), 0, 1)
            norm_pos = np.clip((0.1 - min_diff) / (max_diff - min_diff), 0, 1)
            
            cdict_diff = {
                'red':   [(0.0, 1.0, 1.0),
                        (norm_neg, 1.0, 1.0),
                        (norm_zero, 1.0, 1.0),
                        (norm_pos, 1.0, 1.0),
                        (1.0, 0.0, 0.0)],
                
                'green': [(0.0, 0.0, 0.0),
                        (norm_neg, 1.0, 1.0),
                        (norm_zero, 1.0, 1.0),
                        (norm_pos, 1.0, 1.0),
                        (1.0, 1.0, 1.0)],
                
                'blue':  [(0.0, 0.0, 0.0),
                        (norm_neg, 1.0, 1.0),
                        (norm_zero, 1.0, 1.0),
                        (norm_pos, 1.0, 1.0),
                        (1.0, 0.0, 0.0)]
            }
            
            custom_diff_cmap = LinearSegmentedColormap('custom_diff_cmap', cdict_diff)
        
        floorplan_data = request_data
        x_offset = floorplan_data['centerXShape'] - floorplan_data['centerXCell']
        y_offset = floorplan_data['centerYShape'] - floorplan_data['centerYCell']
        
        offset_plot = (x_offset / 25) * config['scale_factor'] - ((12.5 / 25) * config['scale_factor'])
        offset_plot_right = ((x_offset / 25) + (floorplan_data['width'] / 25)) * config['scale_factor']
        offset_y_plot = (y_offset / 25) * config['scale_factor'] - ((12.5 / 25) * config['scale_factor'] * 2)
        offset_y_plot_bottom = ((y_offset / 25) + (floorplan_data['height'] / 25)) * config['scale_factor']
        
        num_boundary = config['boundary_points']
        y_range = np.linspace(offset_y_plot, offset_y_plot_bottom, num_boundary)
        x_range = np.linspace(offset_plot, offset_plot_right, num_boundary)
        
        left_boundary = [(offset_plot, y) for y in y_range]
        right_boundary = [(offset_plot_right, y) for y in y_range]
        top_boundary = [(x, offset_y_plot_bottom) for x in x_range]
        bottom_boundary = [(x, offset_y_plot) for x in x_range]
        
        boundary_points = list(set(left_boundary + right_boundary + top_boundary + bottom_boundary))
        
        min_z = zi.min()
        max_z = zi.max()
        zero_norm = (0 - global_vars['min_z_rounded']) / (global_vars['max_z_rounded'] - global_vars['min_z_rounded'])
        
        cdict = {
            'red': [(0.0, 1.0, 1.0), (zero_norm, 1.0, 1.0), (1.0, 0.0, 0.0)],
            'green': [(0.0, 0.0, 0.0), (zero_norm, 1.0, 1.0), (1.0, 1.0, 1.0)],
            'blue': [(0.0, 0.0, 0.0), (zero_norm, 1.0, 1.0), (1.0, 0.0, 0.0)]
        }
        custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)
        global_vars['custom_cmap'] = custom_cmap
        
        plot_names = ['Mesh 3D', 'Mesh Contour A', 'Mesh Contour B', 'Mesh Repair Plan', 
                     'Mesh All Profiles', 'Mesh Deflection Exceeds', 'Mesh Tilt Exceeds', 
                     'Mesh Singular Plots', 'Overlay', 'Comparison']
        
        plots = {}
        for name in plot_names:
            if name == 'Mesh 3D':
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': '3d'})
            else:
                fig, ax = plt.subplots(figsize=(10, 10))
            plots[name] = {'fig': fig, 'ax': ax}
        
        for name in plot_names:
            if name != 'Mesh 3D':
                flag = 'no'
                if name == "Mesh Deflection Exceeds" or name == "Mesh All Profiles":
                    flag = 'yes'
                elif name == "Mesh Singular Plots":
                    # Special handling for singular plots - setup with floorplan already embedded
                    flag = 'singular'
                elif name == 'Mesh Contour A':
                    flag = 'A'
                elif name == 'Mesh Contour B':
                    flag = 'B'
                elif name == 'Mesh Repair Plan':
                    flag = 'R'
                elif name == 'Overlay':
                    flag = 'O'
                
                if name != 'Comparison':
                    ax = plots[name]['ax']
                    if flag == 'singular':
                        # Setup singular plot with floorplan embedded
                        setup_singular_plot_with_floorplan(ax, xi, yi, zi, custom_cmap, 
                                      global_vars['min_z_rounded'], global_vars['max_z_rounded'],
                                      x_all, y_all, z_all, point_id, global_vars, 
                                      optimizer, request_data, config['scale_factor'])
                    else:
                        setup_2d_plot_optimized(ax, xi, yi, zi, custom_cmap, name, 
                                      global_vars['min_z_rounded'], global_vars['max_z_rounded'], 
                                      x_all, y_all, z_all, point_id, flag, global_vars)
                elif name == 'Comparison' and comparison_flag:
                    ax = plots[name]['ax']
                    setup_2d_plot_optimized(ax, xi, yi, diff, custom_diff_cmap, 'Comparison', 
                                  np.min(diff), np.max(diff), x_all, y_all, diff, 
                                  point_id, 'C', global_vars)
        
        imageURLs = {}
        graphics_id = request_data['graphicsId']
        failed_profiles_id = request_data['failedProfilesId']
        one_id = request_data['oneID']
        
        if comparison_flag:
            mesh_map_name_contour = '6 - Mesh Comparison.jpeg'
            image_id = optimized_save_plot_and_upload(
                optimizer, x_all, y_all, z_all, point_id,
                plots['Comparison']['ax'], plots['Comparison']['fig'], 
                mesh_map_name_contour, graphics_id, config['scale_factor'], 
                request_data, global_vars
            )
            image_url = f'https://drive.google.com/uc?export=view&id={image_id}'
            formula = f'=IMAGE("{image_url}")'
            ws = sheet.worksheet('GRAPHICS')
            ws.update_acell('D8', formula)
            imageURLs['mesh_comparison'] = image_url
        
        mesh_map_name_contour = '1 - Elevation Plot.jpeg'
        image_id = optimized_save_plot_and_upload(
            optimizer, x_all, y_all, z_all, point_id,
            plots['Mesh Contour A']['ax'], plots['Mesh Contour A']['fig'],
            mesh_map_name_contour, graphics_id, config['scale_factor'],
            request_data, global_vars
        )
        image_url = f'https://drive.google.com/uc?export=view&id={image_id}'
        formula = f'=IMAGE("{image_url}")'
        ws = sheet.worksheet('GRAPHICS')
        ws.update_acell('H5', formula)
        imageURLs['elevation_plot'] = image_url
        
        credentials, project = default(scopes=['https://www.googleapis.com/auth/drive'])
        service = build('drive', 'v3', credentials=credentials)
        request = service.files().get_media(fileId=image_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        image = Image.open(fh)
        width, height = image.size
        
        inches = height / 300
        points = round(inches * 72)
        
        row_pixel_sizes = {
            229: points,
            242: points,
            257: points,
            262: points,
            269: points,
            282: points,
        }
        
        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        credentials = Credentials.from_service_account_info(json_key, scopes=scopes)
        sheets_service = googleapiclient.discovery.build('sheets', 'v4', credentials=credentials)
        
        requests = []
        for row_index, pixel_size in row_pixel_sizes.items():
            requests.append({
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": one_id,
                        "dimension": "ROWS",
                        "startIndex": row_index - 1,
                        "endIndex": row_index,
                    },
                    "properties": {
                        "pixelSize": pixel_size
                    },
                    "fields": "pixelSize"
                }
            })
        
        body = {"requests": requests}
        response = sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=body
        ).execute()
        
        repair_contour = '7 - Repair Plot.jpeg'
        image_id = optimized_save_plot_and_upload(
            optimizer, x_all, y_all, z_all, point_id,
            plots['Mesh Repair Plan']['ax'], plots['Mesh Repair Plan']['fig'],
            repair_contour, graphics_id, config['scale_factor'],
            request_data, global_vars
        )
        image_url = f'https://drive.google.com/uc?export=view&id={image_id}'
        formula = f'=IMAGE("{image_url}")'
        ws = sheet.worksheet('GRAPHICS')
        ws.update_acell('H8', formula)
        imageURLs['repair_plan'] = image_url
        
        mesh_map_name_contour = '2- Contours Mesh.jpeg'
        image_id = optimized_save_plot_and_upload(
            optimizer, x_all, y_all, z_all, point_id,
            plots['Mesh Contour B']['ax'], plots['Mesh Contour B']['fig'],
            mesh_map_name_contour, graphics_id, config['scale_factor'],
            request_data, global_vars
        )
        image_url = f'https://drive.google.com/uc?export=view&id={image_id}'
        formula = f'=IMAGE("{image_url}")'
        ws = sheet.worksheet('GRAPHICS')
        ws.update_acell('D6', formula)
        imageURLs['contours'] = image_url
        
        fig = plots['Mesh All Profiles']['fig']
        ax = plots['Mesh All Profiles']['ax']
        line_ids = add_profile_lines_optimized(ax, boundary_points, left_boundary, 
                                              right_boundary, bottom_boundary, top_boundary)
        mesh_map_name_profiles = '3 - All Profiles.jpeg'
        image_id = optimized_save_plot_and_upload(
            optimizer, x_all, y_all, z_all, point_id,
            plots['Mesh All Profiles']['ax'], plots['Mesh All Profiles']['fig'],
            mesh_map_name_profiles, graphics_id, config['scale_factor'],
            request_data, global_vars
        )
        image_url = f'https://drive.google.com/uc?export=view&id={image_id}'
        formula = f'=IMAGE("{image_url}")'
        ws = sheet.worksheet('GRAPHICS')
        ws.update_acell('H6', formula)
        imageURLs['all_profiles'] = image_url
        
        ws2 = sheet.worksheet('FP0')
        keep_running = ws2.acell('K6').value
        
        if keep_running == 'Initial Graphics':
            ws2.update('K7', [['Complete']])
            return jsonify(imageURLs)
        
        ws_ob = sheet.worksheet('2-EFE')
        range_to_clear = 'P7:DA88'
        ws_ob.batch_clear([range_to_clear])
        
        ws2.update('K7', [['Line Analysis Started - Batch Processing']])
        
        calculation_results = batch_process_all_lines(
            line_ids=line_ids,
            xi=xi, yi=yi, zi=zi,
            total_point_per_profile=config['total_point_per_profile'],
            effective_length=config['effective_length'],
            scale_factor=config['scale_factor'],
            sa=sa,
            spreadsheet_id=spreadsheet_id,
            plots=plots,
            failed_profiles_id=failed_profiles_id,
            x_all=x_all, y_all=y_all, z_all=z_all, point_id=point_id,
            optimizer=optimizer, request_data=request_data, global_vars=global_vars
        )
        
        ws2.update('K7', [['Batch Processing Complete']])
        
        mesh_map_name_deflection = '4 - Failed Deflections.jpeg'
        image_id = optimized_save_plot_and_upload(
            optimizer, x_all, y_all, z_all, point_id,
            plots['Mesh Deflection Exceeds']['ax'], plots['Mesh Deflection Exceeds']['fig'],
            mesh_map_name_deflection, graphics_id, config['scale_factor'],
            request_data, global_vars
        )
        image_url = f'https://drive.google.com/uc?export=view&id={image_id}'
        formula = f'=IMAGE("{image_url}")'
        ws = sheet.worksheet('GRAPHICS')
        ws.update_acell('D7', formula)
        imageURLs['failed_deflections'] = image_url
        
        mesh_map_name_tilt = '5 - Failed Tilt.jpeg'
        image_id = optimized_save_plot_and_upload(
            optimizer, x_all, y_all, z_all, point_id,
            plots['Mesh Tilt Exceeds']['ax'], plots['Mesh Tilt Exceeds']['fig'],
            mesh_map_name_tilt, graphics_id, config['scale_factor'],
            request_data, global_vars
        )
        image_url = f'https://drive.google.com/uc?export=view&id={image_id}'
        formula = f'=IMAGE("{image_url}")'
        ws = sheet.worksheet('GRAPHICS')
        ws.update_acell('H7', formula)
        imageURLs['failed_tilt'] = image_url
        
        ws2.update('K7', [['Complete']])
        return jsonify(imageURLs)
        
    except Exception as e:
        print(f"Error in script_run: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
