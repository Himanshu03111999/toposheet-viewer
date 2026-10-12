import os, re, io, json, zipfile, tempfile, textwrap, threading, math
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
import streamlit as st
import geopandas as gpd
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image
from pmtiles.reader import Reader

Image.MAX_IMAGE_PIXELS = None
GITHUB_USER, GITHUB_REPO, RELEASE_TAG = "Himanshu03111999", "toposheet-storage", "v1.0.0"
PROXY_PORT = 8765
API = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}"
R = 6378137.0

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    try:
        TOKEN = st.secrets.get("GITHUB_TOKEN")
    except Exception:
        TOKEN = None
AUTH = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

st.set_page_config(page_title="Toposheet Viewer and Map Exporter", layout="wide")
st.markdown("""
<style>
header[data-testid="stHeader"]{display:none!important}
.block-container{padding:2.5rem 1.5rem .5rem!important}
.big-main-title{font-size:38px;font-weight:800;line-height:1.25;margin-bottom:.6rem}
html,body,[class*="css"],.stMarkdown,p,span,label{font-size:12px!important}
.stApp,[data-testid="stAppViewContainer"]{opacity:1!important;transition:none!important}
[data-testid="stStatusWidget"]{display:none!important}
.stButton>button{font-size:12px!important;font-weight:600!important}
hr{margin:.4rem 0!important}
.custom-footer{text-align:center;color:#888;font-size:10px;padding:4px 0;border-top:1px solid #e0e0e0;margin-top:10px}
.custom-footer a{color:#06c;text-decoration:none}
</style>
<div class="big-main-title">🌐 Toposheet Viewer and Map Exporter</div>
""", unsafe_allow_html=True)

# ---------------- GitHub assets + range proxy ----------------
@st.cache_data(ttl=300)
def release_assets():
    r = requests.get(f"{API}/releases/tags/{RELEASE_TAG}",
                     headers={"Accept": "application/vnd.github+json", **AUTH}, timeout=30)
    r.raise_for_status()
    return {a["name"]: str(a["id"]) for a in r.json()["assets"]}

@lru_cache(maxsize=2048)
def fetch_range(asset_id, rng=None):
    h = {"Accept": "application/octet-stream", **AUTH}
    if rng:
        h["Range"] = rng
    r = requests.get(f"{API}/releases/assets/{asset_id}", headers=h, timeout=60)
    return r.status_code, r.headers.get("Content-Range"), r.content

class Proxy(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        try:
            code, cr, body = fetch_range(self.path.strip("/").split("/")[-1], self.headers.get("Range"))
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            if cr:
                self.send_header("Content-Range", cr)
            self._cors(); self.end_headers(); self.wfile.write(body)
        except Exception:
            pass

    def log_message(self, *a):
        pass

@st.cache_resource
def start_proxy():
    srv = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), Proxy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

start_proxy()

def legend_image(assets):
    if "legend_topsheet.png" in assets:
        code, _, body = fetch_range(assets["legend_topsheet.png"], None)
        if code == 200:
            return body
    return None

# ---------------- Vector + buffers ----------------
def load_vector(f):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, f.name)
        open(p, "wb").write(f.getbuffer())
        if f.name.lower().endswith(".zip"):
            zipfile.ZipFile(p).extractall(d)
            p = next((os.path.join(d, x) for x in os.listdir(d) if x.endswith(".shp")), None)
            if not p:
                return None
        g = gpd.read_file(p)
    return g.set_crs(4326) if g.crs is None else g.to_crs(4326)

def sheet_no(name):
    s = re.findall(r"\d{1,3}[A-P]/?\d{1,2}", name, re.I)
    if not s:
        return None
    s = s[0].upper()
    return s if "/" in s else re.sub(r"([A-P])(\d+)", r"\1/\2", s)

# ---------------- High-res export (stitches PMTiles tiles) ----------------
def mosaic(asset_id, bounds, max_tiles=12):
    rd = Reader(lambda o, l: fetch_range(asset_id, f"bytes={o}-{o + l - 1}")[2])
    h = rd.header()
    z = h["max_zoom"]

    def rng(z):
        n = 2 ** z
        tx = lambda lon: int((lon + 180) / 360 * n)
        ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
        x0, x1 = sorted((tx(bounds[0]), tx(bounds[2])))
        y0, y1 = sorted((ty(bounds[1]), ty(bounds[3])))
        return n, max(x0, 0), min(x1, n - 1), max(y0, 0), min(y1, n - 1)

    n, x0, x1, y0, y1 = rng(z)
    while z > h["min_zoom"] and (x1 - x0 + 1 > max_tiles or y1 - y0 + 1 > max_tiles):
        z -= 1
        n, x0, x1, y0, y1 = rng(z)

    canvas = Image.new("RGBA", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256), (0, 0, 0, 0))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            data = rd.get(z, x, y)
            if data:
                t = Image.open(io.BytesIO(data)).convert("RGBA").resize((256, 256))
                canvas.alpha_composite(t, ((x - x0) * 256, (y - y0) * 256))
    ts = 2 * math.pi * R / n
    ext = [-math.pi * R + x0 * ts, -math.pi * R + (x1 + 1) * ts,
           math.pi * R - (y1 + 1) * ts, math.pi * R - y0 * ts]
    return canvas, ext

def export_map(sheets_pm, site, title, sheets_str, dpi, legend_bytes):
    fig = plt.figure(figsize=(16, 11), dpi=dpi)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.6, 1.2], wspace=0.012)
    ax = plt.subplot(gs[0])

    if site is not None:
        s = site.to_crs(3857)
        bufs = {km: s.buffer(km * 1000) for km in (10, 5, 2)}
        b = bufs[10].total_bounds
        dx, dy = (b[2] - b[0]) * .15, (b[3] - b[1]) * .15
        view = [b[0] - dx, b[2] + dx, b[1] - dy, b[3] + dy]
        ll = gpd.GeoSeries.from_xy([view[0], view[1]], [view[2], view[3]], crs=3857).to_crs(4326)
        bbox = [ll.x.min(), ll.y.min(), ll.x.max(), ll.y.max()]
    else:
        view, bbox = None, [68, 6, 98, 36]

    for aid in sheets_pm:
        img, ext = mosaic(aid, bbox)
        ax.imshow(img, extent=ext, origin="upper")

    if site is not None:
        for km, c, ls, lw in ((10, "#333", "--", 1.2), (5, "#000", "--", 1.5), (2, "#000", ":", 1.8)):
            bufs[km].plot(ax=ax, facecolor="none", edgecolor=c, linestyle=ls, linewidth=lw)
        s.plot(ax=ax, facecolor="none", edgecolor="#000", linewidth=2)
        ax.set_xlim(view[:2]); ax.set_ylim(view[2:])

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{math.degrees(x / R):.2f}°"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{math.degrees(math.atan(math.sinh(y / R))):.2f}°"))
    ax.set_title("Topographic MAP", fontsize=22, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=.5)

    sd = plt.subplot(gs[1]); sd.set_xlim(0, 1); sd.set_ylim(0, 1); sd.axis("off")
    sd.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="k", lw=1.5, transform=sd.transAxes))
    sd.plot([.08, .45], [.945, .945], "k", lw=1.5)
    for t, lab in ((.08, "0"), (.265, "1"), (.45, "2 km")):
        sd.plot([t, t], [.945, .957], "k", lw=1.5)
        sd.text(t, .96, lab, fontsize=8.5, fontweight="bold", ha="center", va="bottom")
    sd.text(.8, .965, "N", fontsize=18, fontweight="bold", ha="center", va="center")
    sd.text(.8, .93, "▲", fontsize=28, ha="center", va="center")
    for y in (.905, .74, .545):
        sd.plot([0, 1], [y, y], "k", lw=1)
    sd.text(.5, .88, "Title :", fontsize=11.5, fontweight="bold", ha="center", va="center")
    sd.text(.5, .815, "\n".join(textwrap.wrap(title.replace("\n", " "), 26)), fontsize=10,
            ha="center", va="center", linespacing=1.2)
    sd.text(.5, .715, "Legends", fontsize=10.5, fontweight="bold", ha="center", va="center")
    for y, lab, ls, lw, c in ((.672, "Project Site", "-", 1.8, "k"), (.637, "Core Zone : 2 km", ":", 1.8, "k"),
                              (.602, "Buffer Zone 5 km", "--", 1.5, "k"), (.567, "Buffer Zone 10 km", "--", 1.2, "#333")):
        sd.add_patch(Rectangle((.08, y), .12, .022, fill=False, edgecolor=c, linestyle=ls, lw=lw))
        sd.text(.25, y + .011, lab, fontsize=8.5, va="center")
    if legend_bytes:
        try:
            li = sd.inset_axes([.005, .005, .99, .535])
            li.imshow(Image.open(io.BytesIO(legend_bytes)), aspect="auto"); li.axis("off")
        except Exception:
            pass

    fig.text(.15, .02, "Source : Survey of India", fontsize=12, fontweight="bold")
    fig.text(.42, .02, f"Topo sheet No. {sheets_str}", fontsize=12, fontweight="bold")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Data Inputs")
assets = {}
try:
    assets = release_assets()
except Exception as e:
    st.sidebar.error(f"GitHub error: {e}")
pm_assets = {n: i for n, i in assets.items() if n.endswith(".pmtiles")}

basemap = st.sidebar.selectbox("🗺️ Choose Base Map",
                               ["OpenStreetMap", "Google Satellite", "Google Hybrid", "Google Roadmap", "Esri Satellite"])
show_topo = st.sidebar.toggle("🗾 Show Topo Layer", value=True)
chosen = st.sidebar.multiselect("Toposheets (PMTiles)", list(pm_assets))
vec = st.sidebar.file_uploader("Upload Vector (.geojson, .kml, .zip)", type=["geojson", "kml", "gpx", "zip"])

site = None
if vec:
    try:
        site = load_vector(vec)
        st.sidebar.caption("✅ Vectors Loaded")
    except Exception as e:
        st.sidebar.error(f"Vector error: {e}")

sheets = [s for s in dict.fromkeys(sheet_no(n) for n in chosen) if s]
sheets_str = ", ".join(sheets) or "56F/12, 56F/8, 56G/5, 56G/9"

cfg = {"basemap": basemap, "site": None, "buffers": [],
       "pm": [{"name": n, "url": f"http://127.0.0.1:{PROXY_PORT}/{pm_assets[n]}"} for n in chosen] if show_topo else []}
if site is not None:
    m3857 = site.to_crs(3857)
    cfg["site"] = json.loads(site.to_json())
    cfg["buffers"] = [{"km": km, "gj": json.loads(m3857.buffer(km * 1000).to_crs(4326).to_json())}
                      for km in (10, 5, 2)]

# ---------------- Page ----------------
project_title = st.text_input(
    "Project Title",
    value="Preliminary Assessment Study for the 14MW Solar Project at Gadgi, Bidar District, Karnataka, India")
st.markdown("---")

HTML = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/pmtiles@3.0.7/dist/pmtiles.js"></script>
<div id="m" style="height:480px;border:1px solid #ccc"></div>
<script>
const C = __CFG__;
const BASE = {
 "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
 "Google Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
 "Google Hybrid": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
 "Google Roadmap": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
 "Esri Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
};
const map = L.map('m', {preferCanvas: true, zoomSnap: 0.5}).setView([20, 78], 5);
L.control.scale().addTo(map);
L.tileLayer(BASE[C.basemap], {maxZoom: 20, attribution: C.basemap}).addTo(map);
map.createPane('topo'); map.getPane('topo').style.zIndex = 300;
let fit = L.latLngBounds([]);
if (C.site) {
  const sty = {10: ['#555', 2, '6,6'], 5: ['#333', 2.5, '4,4'], 2: ['#000', 2.5, '2,4']};
  C.buffers.forEach(b => {
    const g = L.geoJSON(b.gj, {style: {color: sty[b.km][0], weight: sty[b.km][1], dashArray: sty[b.km][2], fill: false}}).addTo(map);
    if (b.km == 10) { fit = g.getBounds().pad(0.15); map.fitBounds(fit); }
  });
  L.geoJSON(C.site, {style: {color: '#000', weight: 3, fill: false}}).addTo(map);
}
(async () => {
  for (const p of C.pm) {
    try {
      const src = new pmtiles.PMTiles(p.url);
      const h = await src.getHeader();
      pmtiles.leafletRasterLayer(src, {pane: 'topo', maxNativeZoom: h.maxZoom, maxZoom: 20,
                                       opacity: 0.85, keepBuffer: 4, updateWhenZooming: false}).addTo(map);
      if (!C.site) fit.extend([[h.minLat, h.minLon], [h.maxLat, h.maxLon]]);
    } catch (e) { console.error(p.name, e); }
  }
  if (!C.site && fit.isValid()) map.fitBounds(fit);
})();
</script>
"""

leg_bytes = legend_image(assets)
col_map, col_leg = st.columns([3.2, 1.1])
with col_map:
    components.html(HTML.replace("__CFG__", json.dumps(cfg)), height=490)
    c1, c2 = st.columns([1, 2])
    c1.markdown("**Source : Survey of India**")
    c2.markdown(f"**Topo sheet No. {sheets_str}**")
with col_leg:
    with st.container(height=480, border=True):
        st.markdown("### **Legends**")
        st.markdown("""
- 🔲 **Project Site**: Outline
- ┈┈ **Core Zone 2 km**: Dotted
- ╌╌ **Buffer Zone 5 km**: Dashed
- ╌╌ **Buffer Zone 10 km**: Fine Dashed
""")
        if leg_bytes:
            st.image(leg_bytes, caption="CONVENTIONAL SYMBOLS", use_container_width=True)

st.markdown("---")
cl, cd, cb = st.columns([1.2, 1.5, 3])
cl.markdown("### 📥 Export Map")
dpi = cd.selectbox("Resolution (DPI)", [300, 600, 800], index=1, label_visibility="collapsed")
if cb.button("🚀 Generate High-Res Map", use_container_width=True):
    with st.spinner("Generating High-Resolution PNG..."):
        try:
            st.session_state["map_png"] = export_map([pm_assets[n] for n in chosen], site,
                                                     project_title, sheets_str, dpi, leg_bytes)
        except Exception as e:
            st.error(f"Export failed: {e}")
if st.session_state.get("map_png"):
    st.download_button("⬇️ Download Ready High-Res Map (PNG)", st.session_state["map_png"],
                       file_name="Topographic_Map_HighRes.png", mime="image/png", use_container_width=True)

st.markdown("""
<div class="custom-footer">Developed by <b>Himanshu Gupta</b> &nbsp;|&nbsp;
📧 <a href="mailto:himanshu.guptta.311@gmail.com">himanshu.guptta.311@gmail.com</a> &nbsp;|&nbsp;
🔗 <a href="https://www.linkedin.com/in/himanshu-gupta-4a7942224" target="_blank">LinkedIn Profile</a></div>
""", unsafe_allow_html=True)
