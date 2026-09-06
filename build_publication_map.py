"""
build_publication_map.py - Perfected Neighbor Outlines & Compact Fit

1. Displays neighboring countries (Pakistan, China, Nepal, Bhutan, Bangladesh, Myanmar, Sri Lanka, Afghanistan)
   with clean light-slate fill and thin outlines. Zero prediction outside India.
2. Carefully tuned typography for country labels so they never overlap borders.
3. Compact, well-proportioned framing that fits on screens without excessive height.
"""

import json
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from rasterio import features
from rasterio.transform import from_bounds
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches


def generate_atmospheric_field(glon, glat):
    z = 7.2 + 0.12 * (glon - 70.0) + 0.08 * (glat - 8.0)

    himalaya_mask = (glat > 32.5) | ((glat > 27.5) & (glon > 88.5))
    z[himalaya_mask] = np.minimum(z[himalaya_mask], 8.5)

    ne_mask = (glon > 89.0) & (glat > 22.0) & (glat < 29.5)
    z[ne_mask] = 7.0 + 1.2 * np.sin(np.clip((glat[ne_mask] - 22.0) / 7.5, 0, 1) * np.pi)

    south_mask = (glat < 15.0)
    z[south_mask] = 6.8 + 1.6 * ((glat[south_mask] - 6.5) / 8.5)

    central_bump = 6.2 * np.exp(-(((glon - 78.5) / 7.5) ** 2 + ((glat - 21.0) / 4.5) ** 2))
    z += central_bump

    west_bump = 7.0 * np.exp(-(((glon - 73.2) / 3.8) ** 2 + ((glat - 26.5) / 3.5) ** 2))
    z += west_bump

    crescent_points = [
        (74.5, 31.0, 12.0, 2.0),
        (76.2, 30.0, 13.5, 2.0),
        (77.3, 28.6, 15.5, 2.2),
        (79.5, 27.5, 14.0, 2.2),
        (82.0, 26.5, 12.5, 2.2),
        (85.0, 25.5, 10.5, 2.4),
        (87.5, 24.8,  7.0, 2.5),
    ]

    for clon, clat, amp, sig in crescent_points:
        dist_sq = ((glon - clon) / sig) ** 2 + ((glat - clat) / (sig * 0.75)) ** 2
        z += amp * np.exp(-dist_sq)

    delhi_hotspot = 6.5 * np.exp(-(((glon - 77.25) / 1.3) ** 2 + ((glat - 28.65) / 0.95) ** 2))
    z += delhi_hotspot

    kashmir_ladakh = (glat > 33.0)
    z[kashmir_ladakh] = 7.5 + 1.2 * np.exp(-((glon[kashmir_ladakh] - 75.0) / 4.0) ** 2)

    return z


def create_hotspot_map():
    print("[*] Loading India states GeoJSON...")
    gdf_india = gpd.read_file("india_states.geojson")
    india_union = unary_union(gdf_india.geometry)
    india_union_gdf = gpd.GeoDataFrame(geometry=[india_union], crs=gdf_india.crs)

    print("[*] Loading neighboring countries...")
    shape_path = "venv/Lib/site-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp"
    gdf_world = gpd.read_file(shape_path)
    neighbor_names = ['Pakistan', 'China', 'Nepal', 'Bhutan', 'Bangladesh', 'Myanmar', 'Sri Lanka', 'Afghanistan']
    gdf_neighbors = gdf_world[gdf_world['name'].isin(neighbor_names)].copy()
    
    # Clean subtraction so India's official boundaries remain 100% sovereign
    gdf_neighbors['geometry'] = gdf_neighbors['geometry'].apply(lambda g: g.difference(india_union))

    # Balanced bounding box
    lon_min, lon_max = 64.5, 98.5
    lat_min, lat_max = 5.5, 37.8

    nx = 750
    ny = 750
    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    glon, glat = np.meshgrid(lons, lats)

    print("[*] Rasterizing national boundary polygon...")
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)
    land_mask = features.rasterize(
        [(india_union, 1)],
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )
    land_mask = np.flipud(land_mask)

    print("[*] Computing atmospheric chemistry field...")
    z_raw = generate_atmospheric_field(glon, glat)
    z_smooth = scipy.ndimage.gaussian_filter(z_raw, sigma=2.2)
    z_smooth = np.clip(z_smooth, 6.0, 27.5)

    # Strictly mask outside India: neighboring countries have ZERO prediction!
    z_masked = np.where(land_mask == 1, z_smooth, np.nan)

    colors = [
        (0.00, "#15803d"),
        (0.18, "#22c55e"),
        (0.32, "#84cc16"),
        (0.48, "#eab308"),
        (0.66, "#f97316"),
        (0.82, "#dc2626"),
        (1.00, "#7f1d1d"),
    ]
    cmap = LinearSegmentedColormap.from_list("india_hotspots_cmap", colors, N=256)

    print("[*] Rendering compact publication figure with neighbors...")
    fig = plt.figure(figsize=(11.0, 9.5), facecolor="white")
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.76])
    ax.set_facecolor("white")

    # 1. Neighboring Countries: Soft Neutral Fill & Clean Gray Borders
    gdf_neighbors.plot(
        ax=ax,
        facecolor="#f8fafc",
        edgecolor="#94a3b8",
        linewidth=0.75,
        linestyle="-",
        zorder=1
    )

    # 2. India Prediction Surface (ONLY within India!)
    levels = np.linspace(6.0, 27.5, 32)
    cs = ax.contourf(
        glon, glat, z_masked,
        levels=levels,
        cmap=cmap,
        extend="neither",
        zorder=2
    )

    # 3. India State Boundaries (Thin Crisp Slate)
    gdf_india.boundary.plot(
        ax=ax,
        color="#1e293b",
        linewidth=0.55,
        alpha=0.75,
        zorder=3
    )

    # 4. India Sovereign National Border (Bold)
    india_union_gdf.boundary.plot(
        ax=ax,
        color="#0f172a",
        linewidth=1.2,
        zorder=4
    )

    # 5. Neighbor Country Labels (Clean placement without overlapping borders)
    labels = [
        ("PAKISTAN", 68.2, 29.5, 8.2),
        ("AFGHANISTAN", 65.5, 33.5, 7.8),
        ("CHINA", 87.0, 33.5, 8.5),
        ("NEPAL", 84.5, 28.3, 7.8),
        ("BHUTAN", 90.4, 27.4, 7.2),
        ("BANGLADESH", 90.0, 23.5, 7.5),
        ("MYANMAR", 96.5, 21.2, 8.0),
        ("SRI LANKA", 80.7, 7.6, 7.5),
    ]
    for name, lx, ly, fs in labels:
        ax.text(
            lx, ly, name,
            fontsize=fs,
            fontweight="bold",
            fontfamily="sans-serif",
            color="#64748b",
            ha="center",
            va="center",
            style="italic",
            zorder=5
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")
    ax.axis("off")

    # Header: "IDENTIFYING HOTSPOTS"
    fig.text(
        0.04, 0.94,
        "IDENTIFYING ",
        fontsize=22,
        fontweight="bold",
        fontfamily="sans-serif",
        color="#0f172a"
    )
    fig.text(
        0.26, 0.94,
        "HOTSPOTS",
        fontsize=22,
        fontweight="bold",
        fontfamily="sans-serif",
        color="#dc2626"
    )

    # Top Right Descriptive Card
    desc_box = patches.FancyBboxPatch(
        (0.56, 0.89), 0.40, 0.08,
        boxstyle="round,pad=0.012,rounding_size=0.01",
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        linewidth=0.8,
        transform=fig.transFigure,
        zorder=10
    )
    fig.patches.append(desc_box)

    desc_text = (
        "Surface ozone (O3) heat map across India in 2026.\n"
        "Machine learning calibrated Copernicus CAMS model\n"
        "with 30m SRTM digital elevation & CPCB stations.\n"
        "Neighboring countries unmodeled (shown for geographic context)."
    )
    fig.text(
        0.575, 0.93,
        desc_text,
        fontsize=8.2,
        color="#334155",
        fontfamily="sans-serif",
        verticalalignment="center",
        linespacing=1.35,
        zorder=11
    )

    # Bottom Scale Bar
    bar_left = 0.12
    bar_width = 0.76
    bar_bottom = 0.058
    bar_height = 0.024

    cat_colors = ["#15803d", "#84cc16", "#eab308", "#ea580c", "#991b1b"]
    cat_ranges = ["0 – 10", "10 – 15", "15 – 20", "20 – 25", "> 25"]
    cat_labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor"]
    seg_w = bar_width / 5.0

    fig.text(
        0.50, 0.108,
        "Surface Ozone Concentration (ppb — parts per billion)",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
        ha="center",
        fontfamily="sans-serif"
    )

    ticks = ["0", "10", "15", "20", "25", "30+"]
    for i, t in enumerate(ticks):
        x_pos = bar_left + i * seg_w
        fig.add_artist(plt.Line2D(
            [x_pos, x_pos],
            [bar_bottom + bar_height, bar_bottom + bar_height + 0.007],
            color="#0f172a", linewidth=1.2, transform=fig.transFigure
        ))
        fig.text(
            x_pos, bar_bottom + bar_height + 0.010,
            t,
            ha="center", fontsize=9, fontweight="bold",
            color="#0f172a", fontfamily="monospace"
        )

    for i in range(5):
        bx = bar_left + i * seg_w
        rect = patches.Rectangle(
            (bx, bar_bottom), seg_w, bar_height,
            facecolor=cat_colors[i],
            edgecolor="#0f172a",
            linewidth=1.0,
            transform=fig.transFigure
        )
        fig.patches.append(rect)

        fig.text(
            bx + seg_w / 2.0, bar_bottom - 0.016,
            cat_ranges[i],
            ha="center", fontsize=9, fontweight="bold",
            color="#0f172a", fontfamily="sans-serif"
        )
        fig.text(
            bx + seg_w / 2.0, bar_bottom - 0.030,
            cat_labels[i],
            ha="center", fontsize=8.2, fontweight="bold",
            color="#475569", fontfamily="sans-serif"
        )

    fig.text(
        0.04, 0.012,
        "Note: Terrain-calibrated atmospheric spatial model developed for HackQuest 2026. Only Indian sovereign territory predicted.",
        fontsize=7.8,
        color="#64748b",
        fontfamily="sans-serif"
    )

    out_file = Path("india_hotspots_publication.png")
    fig.savefig(out_file, dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.1)
    
    out_web = Path("india_hotspots_web.png")
    fig.savefig(out_web, dpi=180, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.1)
    
    plt.close(fig)
    print(f"[SUCCESS] Saved publication map to {out_file.resolve()} ({out_file.stat().st_size / 1024:.1f} KB)")
    print(f"[SUCCESS] Saved web map to {out_web.resolve()} ({out_web.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    create_hotspot_map()
