import os
import sys
import re
import tempfile
import zipfile
import io
import textwrap

# ---------------------------------------------------------
# 1. Environment & PROJ Setup
# ---------------------------------------------------------
VENV_PATH = r"C:\Users\Syste\AppData\Local\Programs\Python\Python312\tree_env"
PROJ_DIR = os.path.join(VENV_PATH, "Lib", "site-packages", "pyproj", "proj_dir", "share", "proj")

if os.path.exists(os.path.join(PROJ_DIR, "proj.db")):
    os.environ["PROJ_LIB"] = PROJ_DIR
    os.environ["GDAL_DATA"] = os.path.dirname(PROJ_DIR)

import pyproj
try:
    if os.path.exists(PROJ_DIR):
        pyproj.datadir.set_data_dir(PROJ_DIR)
except Exception:
    pass

import streamlit as st
import folium
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from streamlit_folium import st_folium

CRS_4326 = "EPSG:4326"
CRS_METRIC = "EPSG:3857"

# ---------------------------------------------------------
# 2. Page Config & CSS Styling
# ---------------------------------------------------------
st.set_page_config(page_title="Toposheet Viewer and Map Exporter", layout="wide")

st.markdown("""
    <style>
    /* Prevent cut-off at top */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        overflow: visible !important;
    }

    /* Large Header Title */
    .big-main-header {
        font-size: 30px !important;
        font-weight: 800 !important;
        color: #1a1a1a !important;
        margin-top: 0px !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.2 !important;
        display: block !important;
    }

    /* Compact sizing */
    html, body, [class*="css"], .stMarkdown, p, span, label {
        font-size: 12px !important;
    }

    /* Prevent UI grayout */
    .stApp, [data-testid="stAppViewContainer"], .stMainBlockContainer {
        opacity: 1 !important;
        transition: none !important;
        filter: none !important;
    }
    div[data-st-mode="running"] {
        opacity: 1 !important;
    }

    /* Hide 'No results' popup overlay */
    div[data-baseweb="popover"]:has(div:contains("No results")) {
        display: none !important;
    }

    .stTextInput > div > div > input {
        font-size: 12px !important;
        padding: 4px 8px !important;
    }
    .stSelectbox > div > div {
        font-size: 12px !important;
    }
    .stButton > button {
        font-size: 12px !important;
        padding: 6px 12px !important;
        font-weight: 600 !important;
    }

    [data-testid="stStatusWidget"] {
        visibility: hidden !important;
        display: none !important;
    }

    hr {
        margin: 0.4rem 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Secrets Setup
def get_secret(key, default_val=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default_val)

HF_TOKEN = get_secret("HF_TOKEN", "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")
HF_REPO_ID = get_secret("HF_REPO_ID", "himanshuguptta/geotiff-hub")

@st.cache_resource
def get_hf_api():
    return HfApi(token=HF_TOKEN)

@st.cache_data(ttl=3600, show_spinner=False)
def get_legend_image_path():
    remote_path = "shared_rasters/legend_topsheet.png"
    try:
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_path,
            repo_type="dataset",
            token=HF_TOKEN
        )
        return path
    except Exception:
        return None

# ---------------------------------------------------------
# 3. Filename & Processing Engine
# ---------------------------------------------------------
def extract_toposheet_from_filename(filename):
    pattern = r'(\d{1,3}[A-P]/?\d{1,2})'
    matches = re.findall(pattern, filename, re.IGNORECASE)
    if matches:
        match = matches[0].upper()
        if '/' not in match:
            match = re.sub(r'([A-P])(\d+)', r'\1/\2', match)
        return match
    return None

def list_hf_geotiffs():
    try:
        api = get_hf_api()
        files = list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        return [f.replace("shared_rasters/", "") for f in files if f.startswith("shared_rasters/") and f.endswith(('.tif', '.tiff'))]
    except Exception:
        return []

def process_raster_file(src_raster, max_dim=800):
    scale = min(1.0, max_dim / max(src_raster.width, src_raster.height))
    dst_width = max(1, int(src_raster.width * scale))
    dst_height = max(1, int(src_raster.height * scale))

    if src_raster.crs != CRS_4326:
        transform, width, height = calculate_default_transform(
            src_raster.crs, CRS_4326, src_raster.width, src_raster.height, *src_raster.bounds,
            dst_width=dst_width, dst_height=dst_height
        )
        num_bands = min(src_raster.count, 3)
        reprojected_bands = []
        for b in range(1, num_bands + 1):
            destination = np.zeros((dst_height, dst_width), dtype=src_raster.dtypes[0])
            reproject(
                source=rasterio.band(src_raster, b),
                destination=destination,
                src_transform=src_raster.transform,
                src_crs=src_raster.crs,
                dst_transform=transform,
                dst_crs=CRS_4326,
                resampling=Resampling.bilinear
            )
            reprojected_bands.append(destination)
        bounds = rasterio.transform.array_bounds(dst_height, dst_width, transform)
        rgb = np.stack(reprojected_bands[:3], axis=-1) if len(reprojected_bands) >= 3 else np.stack([reprojected_bands[0]]*3, axis=-1)
    else:
        bounds = src_raster.bounds
        out_shape = (src_raster.count, dst_height, dst_width)
        data = src_raster.read(out_shape=out_shape, resampling=Resampling.bilinear)
        rgb = np.stack([data[0], data[1], data[2]], axis=-1) if src_raster.count >= 3 else np.stack([data[0]]*3, axis=-1)

    rgb = np.nan_to_num(rgb, nan=0.0)
    p_min, p_max = np.percentile(rgb, (2, 98))
    if p_max == p_min:
        p_max += 1e-5
    rgb_scaled = np.clip((rgb - p_min) / (p_max - p_min) * 255, 0, 255).astype(np.uint8)
    bbox_4326 = [bounds[0], bounds[1], bounds[2], bounds[3]]

    return rgb_scaled, bbox_4326

@st.cache_data(ttl=3600, show_spinner=False)
def process_cloud_geotiff_cached(filename, max_dim=800):
    try:
        remote_path = f"shared_rasters/{filename}"
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_path,
            repo_type="dataset",
            token=HF_TOKEN
        )
        with rasterio.open(local_path) as src:
            rgb_scaled, bbox_4326 = process_raster_file(src, max_dim=max_dim)
        return rgb_scaled, bbox_4326, local_path
    except Exception:
        return None, None, None

def process_local_raster(uploaded_file, max_dim=800):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        with rasterio.open(tmp_path) as src:
            rgb_scaled, bbox_4326 = process_raster_file(src, max_dim=max_dim)
        
        return rgb_scaled, bbox_4326, tmp_path
    except Exception:
        return None, None, None

# ---------------------------------------------------------
# 4. Vector & Buffer Engine
# ---------------------------------------------------------
def process_uploaded_vector(uploaded_file):
    filename = uploaded_file.name.lower()
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if filename.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)
            shp_files = [f for f in os.listdir(tmp_dir) if f.endswith(".shp")]
            if shp_files:
                gdf = gpd.read_file(os.path.join(tmp_dir, shp_files[0]))
            else:
                return None
        else:
            gdf = gpd.read_file(file_path)

        if gdf.crs is None:
            gdf = gdf.set_crs(CRS_4326)
        else:
            gdf = gdf.to_crs(CRS_4326)
        return gdf

def create_buffers(gdf):
    if gdf is None:
        return None, None, None
    gdf_metric = gdf.to_crs(CRS_METRIC)
    buf_2km = gdf_metric.buffer(2000).to_crs(CRS_4326)
    buf_5km = gdf_metric.buffer(5000).to_crs(CRS_4326)
    buf_10km = gdf_metric.buffer(10000).to_crs(CRS_4326)
    return buf_2km, buf_5km, buf_10km

# ---------------------------------------------------------
# 5. High-Res Export Engine
# ---------------------------------------------------------
def export_high_res_map(selected_cloud_rasters, local_raster_files, gdf_site, buf_2km, buf_5km, buf_10km, project_title, toposheet_str, dpi=800):
    fig = plt.figure(figsize=(16, 11), dpi=dpi)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.6, 1.2], wspace=0.012)

    ax_map = plt.subplot(gs[0])

    for r_name in selected_cloud_rasters:
        rgb_img, bbox, _ = process_cloud_geotiff_cached(r_name, max_dim=2400)
        if rgb_img is not None:
            ax_map.imshow(rgb_img, extent=[bbox[0], bbox[2], bbox[1], bbox[3]], origin='upper')

    for file_obj in local_raster_files:
        rgb_img, bbox, _ = process_local_raster(file_obj, max_dim=2400)
        if rgb_img is not None:
            ax_map.imshow(rgb_img, extent=[bbox[0], bbox[2], bbox[1], bbox[3]], origin='upper')

    if gdf_site is not None:
        gpd.GeoSeries(buf_10km).plot(ax=ax_map, facecolor='none', edgecolor='#333333', linestyle='--', linewidth=1.2)
        gpd.GeoSeries(buf_5km).plot(ax=ax_map, facecolor='none', edgecolor='#000000', linestyle='--', linewidth=1.5)
        gpd.GeoSeries(buf_2km).plot(ax=ax_map, facecolor='none', edgecolor='#000000', linestyle=':', linewidth=1.8)
        gdf_site.plot(ax=ax_map, facecolor='none', edgecolor='#000000', linewidth=2.0)

        bounds_10 = buf_10km.total_bounds
        dx = (bounds_10[2] - bounds_10[0]) * 0.15
        dy = (bounds_10[3] - bounds_10[1]) * 0.15
        ax_map.set_xlim([bounds_10[0] - dx, bounds_10[2] + dx])
        ax_map.set_ylim([bounds_10[1] - dy, bounds_10[3] + dy])

    ax_map.set_title("Topographic MAP", fontsize=22, fontweight='bold', pad=12)
    ax_map.grid(True, linestyle=':', alpha=0.5)

    ax_side = plt.subplot(gs[1])
    ax_side.set_xlim([0, 1])
    ax_side.set_ylim([0, 1])
    ax_side.axis('off')

    rect = Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=1.5, transform=ax_side.transAxes)
    ax_side.add_patch(rect)

    scale_y = 0.945
    ax_side.plot([0.08, 0.45], [scale_y, scale_y], color='black', lw=1.5)
    for t in [0.08, 0.265, 0.45]:
        ax_side.plot([t, t], [scale_y, scale_y + 0.012], color='black', lw=1.5)
    
    ax_side.text(0.08, scale_y + 0.015, "0", fontsize=8.5, fontweight='bold', ha='center', va='bottom')
    ax_side.text(0.265, scale_y + 0.015, "1", fontsize=8.5, fontweight='bold', ha='center', va='bottom')
    ax_side.text(0.45, scale_y + 0.015, "2 km", fontsize=8.5, fontweight='bold', ha='center', va='bottom')

    ax_side.text(0.80, 0.965, "N", fontsize=18, fontweight='bold', ha='center', va='center')
    ax_side.text(0.80, 0.930, "▲", fontsize=28, ha='center', va='center')

    ax_side.plot([0, 1], [0.905, 0.905], color='black', lw=1)

    ax_side.text(0.5, 0.880, "Title :", fontsize=11.5, fontweight='bold', ha='center', va='center')
    wrapped_lines = textwrap.wrap(project_title.replace('\n', ' '), width=26)
    formatted_title = "\n".join(wrapped_lines)
    ax_side.text(0.5, 0.815, formatted_title, fontsize=10.0, fontweight='medium', ha='center', va='center', clip_on=True, linespacing=1.2)

    ax_side.plot([0, 1], [0.740, 0.740], color='black', lw=1)

    ax_side.text(0.5, 0.715, "Legends", fontsize=10.5, fontweight='bold', ha='center', va='center')
    ax_side.add_patch(Rectangle((0.08, 0.672), 0.12, 0.022, fill=False, edgecolor='black', lw=1.8))
    ax_side.text(0.25, 0.683, "Project Site", fontsize=8.5, va='center')

    ax_side.add_patch(Rectangle((0.08, 0.637), 0.12, 0.022, fill=False, edgecolor='black', linestyle=':', lw=1.8))
    ax_side.text(0.25, 0.648, "Core Zone : 2 km", fontsize=8.5, va='center')

    ax_side.add_patch(Rectangle((0.08, 0.602), 0.12, 0.022, fill=False, edgecolor='black', linestyle='--', lw=1.5))
    ax_side.text(0.25, 0.613, "Buffer Zone 5 km", fontsize=8.5, va='center')

    ax_side.add_patch(Rectangle((0.08, 0.567), 0.12, 0.022, fill=False, edgecolor='#333333', linestyle='--', lw=1.2))
    ax_side.text(0.25, 0.578, "Buffer Zone 10 km", fontsize=8.5, va='center')

    ax_side.plot([0, 1], [0.545, 0.545], color='black', lw=1)

    legend_img_path = get_legend_image_path()
    if legend_img_path and os.path.exists(legend_img_path):
        try:
            leg_img = Image.open(legend_img_path)
            ax_img = ax_side.inset_axes([0.005, 0.005, 0.990, 0.535])
            ax_img.imshow(leg_img, aspect='auto')
            ax_img.axis('off')
        except Exception:
            pass

    fig.text(0.15, 0.02, "Source : Survey of India", fontsize=12, fontweight='bold')
    fig.text(0.42, 0.02, f"Topo sheet No. {toposheet_str}", fontsize=12, fontweight='bold')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------------------------------------------------
# 6. Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Data Inputs")

selected_basemap = st.sidebar.selectbox(
    "🗺️ Choose Base Map",
    ["OpenStreetMap", "Google Satellite", "Google Hybrid", "Google Roadmap", "Esri Satellite"]
)

uploaded_rasters = st.sidebar.file_uploader(
    "Upload Raster GeoTIFF", 
    type=["tif", "tiff"], 
    accept_multiple_files=True
)

shared_geotiffs = list_hf_geotiffs()

selected_cloud_rasters = st.sidebar.multiselect(
    "Cloud Rasters",
    options=shared_geotiffs,
    default=[]
)

RASTER_OPACITY = 0.85

uploaded_vector = st.sidebar.file_uploader("Upload Vector (.geojson, .kml, .zip)", type=["geojson", "kml", "gpx", "zip"])

gdf_site, buf_2km, buf_5km, buf_10km = None, None, None, None
if uploaded_vector is not None:
    try:
        gdf_site = process_uploaded_vector(uploaded_vector)
        if gdf_site is not None:
            buf_2km, buf_5km, buf_10km = create_buffers(gdf_site)
            st.sidebar.caption("✅ Vectors Loaded")
    except Exception:
        pass

# ---------------------------------------------------------
# 7. Dashboard Layout
# ---------------------------------------------------------

st.markdown('<div class="big-main-header">🌐 Toposheet Viewer and Map Exporter</div>', unsafe_allow_html=True)

project_title = st.text_input(
    "Project Title",
    value="Preliminary Assessment Study for the 14MW Solar Project at Gadgi, Bidar District, Karnataka, India"
)

st.markdown("---")

# --- FOLIUM MAP SETUP ---
m = folium.Map(location=[20.0, 78.0], zoom_start=5, tiles=None, control_scale=True)

if selected_basemap == "OpenStreetMap":
    folium.TileLayer('OpenStreetMap').add_to(m)
elif selected_basemap == "Google Satellite":
    folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google Satellite').add_to(m)
elif selected_basemap == "Google Hybrid":
    folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid').add_to(m)
elif selected_basemap == "Google Roadmap":
    folium.TileLayer('https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google Maps').add_to(m)
elif selected_basemap == "Esri Satellite":
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri World Imagery').add_to(m)

all_bounds = []
extracted_toposheets = []

topo_layer_group = folium.FeatureGroup(name="Combined Topo Map", overlay=True, control=True)

for r_name in selected_cloud_rasters:
    sheet_no = extract_toposheet_from_filename(r_name)
    if sheet_no and sheet_no not in extracted_toposheets:
        extracted_toposheets.append(sheet_no)

    rgb_img, bbox, _ = process_cloud_geotiff_cached(r_name, max_dim=800)
    if rgb_img is not None:
        all_bounds.append(bbox)
        folium.raster_layers.ImageOverlay(
            image=rgb_img,
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            opacity=RASTER_OPACITY,
            zindex=1
        ).add_to(topo_layer_group)

if uploaded_rasters:
    for r_file in uploaded_rasters:
        sheet_no = extract_toposheet_from_filename(r_file.name)
        if sheet_no and sheet_no not in extracted_toposheets:
            extracted_toposheets.append(sheet_no)

        rgb_img, bbox, _ = process_local_raster(r_file, max_dim=800)
        if rgb_img is not None:
            all_bounds.append(bbox)
            folium.raster_layers.ImageOverlay(
                image=rgb_img,
                bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
                opacity=RASTER_OPACITY,
                zindex=2
            ).add_to(topo_layer_group)

topo_layer_group.add_to(m)

if gdf_site is not None:
    folium.GeoJson(buf_10km, name="Buffer Zone 10 km", style_function=lambda f: {'color': '#555555', 'weight': 2.0, 'dashArray': '6, 6', 'fillOpacity': 0.0, 'fill': False}).add_to(m)
    folium.GeoJson(buf_5km, name="Buffer Zone 5 km", style_function=lambda f: {'color': '#333333', 'weight': 2.5, 'dashArray': '4, 4', 'fillOpacity': 0.0, 'fill': False}).add_to(m)
    folium.GeoJson(buf_2km, name="Core Zone 2 km", style_function=lambda f: {'color': '#000000', 'weight': 2.5, 'dashArray': '3, 3', 'fillOpacity': 0.0, 'fill': False}).add_to(m)
    folium.GeoJson(gdf_site, name="Project Site", style_function=lambda f: {'color': '#000000', 'weight': 3.0, 'fillOpacity': 0.0, 'fill': False}).add_to(m)

    bounds_10 = buf_10km.total_bounds
    dx = (bounds_10[2] - bounds_10[0]) * 0.15
    dy = (bounds_10[3] - bounds_10[1]) * 0.15
    m.fit_bounds([[bounds_10[1] - dy, bounds_10[0] - dx], [bounds_10[3] + dy, bounds_10[2] + dx]])

elif all_bounds:
    min_x, min_y = min([b[0] for b in all_bounds]), min([b[1] for b in all_bounds])
    max_x, max_y = max([b[2] for b in all_bounds]), max([b[3] for b in all_bounds])
    m.fit_bounds([[min_y, min_x], [max_y, max_x]])

toposheet_str = ", ".join(extracted_toposheets) if extracted_toposheets else "56F/12, 56F/8, 56G/5, 56G/9"

# --- MAIN DASHBOARD MAP & LEGEND ---
MAP_HEIGHT = 480

col_map, col_legend = st.columns([3.2, 1.1])

with col_map:
    st_folium(m, use_container_width=True, height=MAP_HEIGHT, returned_objects=[])

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Source : Survey of India**")
    with c2:
        st.markdown(f"**Topo sheet No. {toposheet_str}**")

with col_legend:
    legend_box = st.container(height=MAP_HEIGHT, border=True)
    with legend_box:
        st.markdown("### **Legends**")
        st.markdown("""
        - 🔲 **Project Site**: Outline
        - ┈┈ **Core Zone 2 km**: Dotted
        - ╌╌ **Buffer Zone 5 km**: Dashed
        - ╌╌ **Buffer Zone 10 km**: Fine Dashed
        """)
        
        leg_path = get_legend_image_path()
        if leg_path and os.path.exists(leg_path):
            st.image(leg_path, caption="CONVENTIONAL SYMBOLS", use_container_width=True)

st.markdown("---")

# --- EXPORT SECTION DIRECTLY BELOW MAP ---
col_lbl, col_dpi, col_btn = st.columns([1.2, 1.5, 3])

with col_lbl:
    st.markdown("### 📥 Export Map")

with col_dpi:
    export_dpi = st.selectbox("Resolution (DPI)", [300, 600, 800], index=2, label_visibility="collapsed")

with col_btn:
    if st.button("🚀 Generate High-Res Map", use_container_width=True):
        with st.spinner("Generating High-Resolution PNG..."):
            st.session_state["generated_map_bytes"] = export_high_res_map(
                selected_cloud_rasters, uploaded_rasters or [], gdf_site, buf_2km, buf_5km, buf_10km,
                project_title, toposheet_str, dpi=export_dpi
            )

# Download button appears cleanly below when ready
if "generated_map_bytes" in st.session_state and st.session_state["generated_map_bytes"] is not None:
    st.download_button(
        label="⬇️ Download Ready High-Res Map (PNG)",
        data=st.session_state["generated_map_bytes"],
        file_name="Topographic_Map_HighRes.png",
        mime="image/png",
        use_container_width=True
    )

# --- COMPACT FOOTER ---
st.markdown("""
    <style>
    .custom-footer {
        width: 100%;
        background-color: transparent;
        color: #888888;
        text-align: center;
        font-size: 10px;
        padding: 4px 0px;
        border-top: 1px solid #e0e0e0;
        margin-top: 10px;
    }
    .custom-footer a {
        color: #0066cc;
        text-decoration: none;
    }
    .custom-footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="custom-footer">
        Developed by <b>Himanshu Gupta</b> &nbsp;|&nbsp; 
        📧 <a href="mailto:himanshu.guptta.311@gmail.com">himanshu.guptta.311@gmail.com</a> &nbsp;|&nbsp; 
        🔗 <a href="https://www.linkedin.com/in/himanshu-gupta-4a7942224" target="_blank">LinkedIn Profile</a>
    </div>
""", unsafe_allow_html=True)