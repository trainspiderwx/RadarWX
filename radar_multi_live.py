import os
import io
import time
import threading
import webbrowser
import gzip
import numpy as np
import boto3
import xarray as xr  
import xradar        
from flask import Flask, jsonify, send_file, request, make_response
from botocore import UNSIGNED
from botocore.config import Config
from datetime import datetime, timezone

app = Flask(__name__)

# --- CONFIGURATION ---
STATIONS = {
    "KTLX": {"name": "Oklahoma City", "lat": 35.3331, "lon": -97.2778},
    "KBOX": {"name": "Boston",        "lat": 42.1325, "lon": -71.1306},
    "KHGX": {"name": "Houston",       "lat": 29.4719, "lon": -95.0792}
}

# Grenzen erweitert, um die neue Farbpalette komplett abzubilden
MIN_VAL = 0.0
MAX_VAL = 100.0

MAX_RANGE_M = 230000  # Passend zur typischen Level-2 Reichweite (230km)
PORT = 8000

# Cache für geladene Sweep-Daten im RAM
_radar_cache = {}

# ═══════════════════════════════════════════════════════════════════════════
# S3 Data Fetcher (Level 2 Live Feed)
# ═══════════════════════════════════════════════════════════════════════════

def update_station_data(station_id):
    """Sucht die neueste Level-2 Datei auf S3 und lädt sie direkt in den RAM."""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = "unidata-nexrad-level2"

    now = datetime.now(timezone.utc)
    prefix = f"{now.strftime('%Y/%m/%d')}/{station_id}/"

    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' not in response:
            return False

        latest_file = sorted(response['Contents'], key=lambda x: x['LastModified'])[-1]
        remote_key = latest_file['Key']
        local_filename = remote_key.split('/')[-1]

        # Nur laden, wenn wir die Datei nicht schon im Cache haben
        if _radar_cache.get(station_id, {}).get("filename") == local_filename:
            return True

        print(f"[S3 Live] Lade neues Signal für {station_id}: {local_filename}")

        # Direkt in den RAM laden -- kein Schreiben/Löschen auf der Platte nötig
        obj = s3.get_object(Bucket=bucket_name, Key=remote_key)
        file_bytes = obj['Body'].read()

        sweep_data = xr.open_dataset(file_bytes, group="sweep_0", engine="nexradlevel2")

        da_dbzh = sweep_data["DBZH"]
        _radar_cache[station_id] = {
            "filename": local_filename,
            "azimuth": da_dbzh["azimuth"].values,
            "range": da_dbzh["range"].values,
            "values": da_dbzh.values.copy(),
            "lat": STATIONS[station_id]["lat"],
            "lon": STATIONS[station_id]["lon"]
        }
        sweep_data.close()
        return True
    except Exception as e:
        print(f"Fehler beim S3-Update für {station_id}: {e}")
        return False

def live_background_worker():
    """Hintergrund-Thread, der alle 60 Sekunden nach neuen Radar-Daten sucht."""
    while True:
        for station_id in STATIONS:
            update_station_data(station_id)
        time.sleep(60)

# ═══════════════════════════════════════════════════════════════════════════
# Flask API: Mathematisch perfekte Vektor-Gates (GeoJSON mit GZIP)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/radar_polygons")
def get_radar_polygons():
    station_id = request.args.get("station")
    if not station_id or station_id not in STATIONS:
        return jsonify({"error": "Ungültige oder fehlende Station"}), 400
        
    # Sicherstellen, dass Daten da sind
    if station_id not in _radar_cache:
        update_station_data(station_id)
        if station_id not in _radar_cache:
            return jsonify({"error": "Keine Daten verfügbar"}), 500
            
    cache = _radar_cache[station_id]
    azimuths = cache["azimuth"]
    ranges = cache["range"]
    data = cache["values"]
    
    lon_0 = cache["lon"]
    lat_0 = cache["lat"]
    
    num_radials = len(azimuths)
    num_bins = len(ranges)
    
    # 1. Winkel-Grenzen nahtlos berechnen
    az_bounds = np.zeros(num_radials + 1)
    az_bounds[1:-1] = (azimuths[:-1] + azimuths[1:]) / 2.0
    az_bounds[0] = azimuths[0] - (azimuths[1] - azimuths[0]) / 2.0
    az_bounds[-1] = azimuths[-1] + (azimuths[-1] - azimuths[-2]) / 2.0

    # 2. Entfernungs-Grenzen nahtlos berechnen
    ranges_km = ranges / 1000.0
    gate_spacing = ranges_km[1] - ranges_km[0] if num_bins > 1 else 0.25
    r_bounds = np.zeros(num_bins + 1)
    r_bounds[:-1] = ranges_km - (gate_spacing / 2.0)
    r_bounds[-1] = ranges_km[-1] + (gate_spacing / 2.0)

    # Maskierung über NumPy (Vektorisiert für maximale Performance)
    valid_mask = np.isfinite(data) & (data >= MIN_VAL) & (data <= MAX_VAL)
    az_indices, r_indices = np.where(valid_mask)

    # Max Range Limit einhalten
    within_range = r_bounds[r_indices] <= (MAX_RANGE_M / 1000.0)
    az_indices = az_indices[within_range]
    r_indices = r_indices[within_range]

    if len(az_indices) == 0:
        return jsonify({"type": "FeatureCollection", "features": []})

    lat_to_km = 111.32
    lon_to_km = 111.32 * np.cos(np.radians(lat_0))

    theta1 = np.radians(az_bounds[az_indices])
    theta2 = np.radians(az_bounds[az_indices + 1])
    r_inner = r_bounds[r_indices]
    r_outer = r_bounds[r_indices + 1]

    sin_t1, cos_t1 = np.sin(theta1), np.cos(theta1)
    sin_t2, cos_t2 = np.sin(theta2), np.cos(theta2)

    p1_lon = lon_0 + (r_inner * sin_t1) / lon_to_km
    p1_lat = lat_0 + (r_inner * cos_t1) / lat_to_km
    p2_lon = lon_0 + (r_outer * sin_t1) / lon_to_km
    p2_lat = lat_0 + (r_outer * cos_t1) / lat_to_km
    p3_lon = lon_0 + (r_outer * sin_t2) / lon_to_km
    p3_lat = lat_0 + (r_outer * cos_t2) / lat_to_km
    p4_lon = lon_0 + (r_inner * sin_t2) / lon_to_km
    p4_lat = lat_0 + (r_inner * cos_t2) / lat_to_km

    vals = data[az_indices, r_indices]

    # Ultra-schneller String-Concatenation Builder mit NaN-Schutz
    chunks = []
    for i in range(len(az_indices)):
        safe_val = vals[i] if np.isfinite(vals[i]) else 0.0
        chunks.append(
            f'{{"type":"Feature","geometry":{{"type":"Polygon","coordinates":[[['
            f'{p1_lon[i]:.5f},{p1_lat[i]:.5f}],[{p2_lon[i]:.5f},{p2_lat[i]:.5f}],'
            f'[{p3_lon[i]:.5f},{p3_lat[i]:.5f}],[{p4_lon[i]:.5f},{p4_lat[i]:.5f}],'
            f'[{p1_lon[i]:.5f},{p1_lat[i]:.5f}]]]}},"properties":{{"dbz":{safe_val:.2f}}}}}'
        )
    
    json_str = '{"type":"FeatureCollection","features":[' + ','.join(chunks) + ']}'
    
    # Kompression für minimale Übertragungszeit
    compressed_content = gzip.compress(json_str.encode('utf-8'))
    response = make_response(compressed_content)
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Type'] = 'application/json'
    return response

# ═══════════════════════════════════════════════════════════════════════════
# Web-Interface (MapLibre Karte)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def serve_index():
    return send_file("index.html")

def generate_index_html():
    """Erzeugt ein MapLibre-Frontend mit der hochauflösenden BR Farbpalette."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>NEXRAD Level 2 Mathematically Perfect Multi-Vector Map</title>
    <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no" />
    <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
    <style>
        body { margin: 0; padding: 0; background-color: #111; font-family: sans-serif; }
        #map { position: absolute; top: 0; bottom: 0; width: 100%; }
        .station-control {
            position: absolute; top: 10px; left: 10px; z-index: 10;
            background: rgba(20,20,20,0.9); padding: 15px; border-radius: 8px;
            color: white; border: 1px solid #333;
        }
        .station-btn {
            display: block; width: 100%; margin: 5px 0; padding: 8px 12px;
            background: #222; color: #ccc; border: 1px solid #444;
            text-align: left; cursor: pointer; border-radius: 4px;
        }
        .station-btn:hover { background: #333; color: white; }
        .station-btn.active { background: #0078ff; color: white; border-color: #00a2ff; }
    </style>
</head>
<body>

<div class="station-control">
    <h3 style="margin:0 0 10px 0; font-size:14px;">NEXRAD Stationen (Vektor)</h3>
    <button class="station-btn active" onclick="switchStation('KTLX', this)">KTLX - Oklahoma</button>
    <button class="station-btn" onclick="switchStation('KBOX', this)">KBOX - Boston</button>
    <button class="station-btn" onclick="switchStation('KHGX', this)">KHGX - Houston</button>
</div>

<div id="map"></div>

<script>
    const map = new maplibregl.Map({
        container: 'map',
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        center: [-97.2778, 35.3331],
        zoom: 6
    });

    let currentStation = 'KTLX';

    // Komplette hochauflösende BR Farbpalette (RGB umgewandelt in Hex/Hex-Strings)
    const dbzExpression = [
        'interpolate', ['linear'], ['get', 'dbz'],
        0.0, 'rgb(1,243,247)',   0.5, 'rgb(3,231,239)',   1.0, 'rgb(5,219,231)',
        1.5, 'rgb(7,207,223)',   2.0, 'rgb(9,195,215)',   2.5, 'rgb(11,183,207)',
        3.0, 'rgb(13,171,199)',  3.5, 'rgb(15,195,191)',  4.0, 'rgb(17,147,183)',
        4.5, 'rgb(19,135,175)',  5.0, 'rgb(21,123,167)',  5.5, 'rgb(23,112,159)',
        6.0, 'rgb(21,114,163)',  6.5, 'rgb(20,117,168)',  7.0, 'rgb(19,120,173)',
        7.5, 'rgb(18,123,178)',  8.0, 'rgb(17,126,182)',  8.5, 'rgb(16,129,187)',
        9.0, 'rgb(15,132,192)',  9.5, 'rgb(14,135,197)',  10.0, 'rgb(12,137,201)',
        10.5, 'rgb(11,140,206)', 11.0, 'rgb(10,143,211)', 11.5, 'rgb(9,146,216)',
        12.0, 'rgb(8,149,220)',  12.5, 'rgb(7,152,255)',  13.0, 'rgb(6,155,230)',
        13.5, 'rgb(5,158,235)',  14.0, 'rgb(21,191,180)', 14.5, 'rgb(37,225,125)',
        15.0, 'rgb(36,221,121)', 15.5, 'rgb(35,218,118)', 16.0, 'rgb(34,214,115)',
        16.5, 'rgb(33,211,112)', 17.0, 'rgb(32,207,108)', 17.5, 'rgb(31,204,105)',
        18.0, 'rgb(30,200,102)', 18.5, 'rgb(29,197,99)',   19.0, 'rgb(28,194,96)',
        19.5, 'rgb(27,190,93)',  20.0, 'rgb(26,187,90)',  20.5, 'rgb(28,184,87)',
        21.0, 'rgb(24,180,84)',  21.5, 'rgb(24,177,81)',  22.0, 'rgb(23,174,77)',
        22.5, 'rgb(22,170,74)',  23.0, 'rgb(21,167,71)',  23.5, 'rgb(20,164,68)',
        24.0, 'rgb(19,160,65)',  24.5, 'rgb(18,157,62)',  25.0, 'rgb(17,154,59)',
        25.5, 'rgb(16,150,56)',  26.0, 'rgb(15,147,53)',  26.5, 'rgb(15,144,50)',
        27.0, 'rgb(14,140,46)',  27.5, 'rgb(13,137,43)',  28.0, 'rgb(12,133,40)',
        28.5, 'rgb(11,130,37)',  29.0, 'rgb(10,127,34)',  29.5, 'rgb(9,123,31)',
        30.0, 'rgb(8,120,27)',   30.5, 'rgb(7,117,24)',   31.0, 'rgb(6,113,21)',
        31.5, 'rgb(5,110,18)',   32.0, 'rgb(4,107,15)',   32.5, 'rgb(3,103,12)',
        33.0, 'rgb(2,100,9)',    33.5, 'rgb(1,96,5)',     34.0, 'rgb(128,175,19)',
        34.5, 'rgb(255,255,33)', 35.0, 'rgb(255,247,28)', 35.5, 'rgb(255,239,23)',
        36.0, 'rgb(255,231,18)', 36.5, 'rgb(255,223,14)', 37.0, 'rgb(255,215,9)',
        37.5, 'rgb(255,207,4)',  38.0, 'rgb(255,199,0)',  38.5, 'rgb(255,191,0)',
        39.0, 'rgb(255,183,0)',  39.5, 'rgb(255,175,0)',  40.0, 'rgb(255,157,0)',
        40.5, 'rgb(255,140,0)',  41.0, 'rgb(255,122,0)',  41.5, 'rgb(255,105,0)',
        42.0, 'rgb(255,87,0)',   42.5, 'rgb(255,70,0)',   43.0, 'rgb(255,52,0)',
        43.5, 'rgb(255,35,0)',   44.0, 'rgb(255,17,0)',   44.5, 'rgb(255,0,0)',
        45.0, 'rgb(249,0,0)',    45.5, 'rgb(244,0,0)',    46.0, 'rgb(239,0,0)',
        46.5, 'rgb(233,0,0)',    47.0, 'rgb(228,0,0)',    47.5, 'rgb(223,0,0)',
        48.0, 'rgb(217,0,0)',    48.5, 'rgb(212,0,0)',    49.0, 'rgb(207,0,0)',
        49.5, 'rgb(201,0,0)',    50.0, 'rgb(195,0,0)',    50.5, 'rgb(190,0,0)',
        51.0, 'rgb(185,0,0)',    51.5, 'rgb(180,0,0)',    52.0, 'rgb(175,0,0)',
        52.5, 'rgb(170,0,0)',    53.0, 'rgb(165,0,0)',    53.5, 'rgb(160,0,0)',
        54.0, 'rgb(154,0,0)',    54.5, 'rgb(180,0,180)',  55.0, 'rgb(186,9,185)',
        55.5, 'rgb(192,19,190)', 56.0, 'rgb(198,29,195)', 56.5, 'rgb(204,39,201)',
        57.0, 'rgb(210,49,206)', 57.5, 'rgb(216,59,211)', 58.0, 'rgb(223,68,216)',
        58.5, 'rgb(229,78,222)', 59.0, 'rgb(235,88,227)', 59.5, 'rgb(241,98,232)',
        60.0, 'rgb(247,108,237)',60.5, 'rgb(253,117,243)',61.0, 'rgb(232,109,232)',
        61.5, 'rgb(212,104,204)',62.0, 'rgb(192,93,184)', 62.5, 'rgb(171,85,165)',
        63.0, 'rgb(151,77,146)', 63.5, 'rgb(131,69,126)', 64.0, 'rgb(111,61,107)',
        64.5, 'rgb(90,53,88)',   65.0, 'rgb(70,45,68)',   65.5, 'rgb(50,37,49)',
        66.0, 'rgb(29,30,29)',   66.5, 'rgb(33,34,33)',   67.0, 'rgb(37,38,37)',
        67.5, 'rgb(41,42,41)',   68.0, 'rgb(45,46,45)',   68.5, 'rgb(49,50,49)',
        69.0, 'rgb(53,54,53)',   69.5, 'rgb(57,58,57)',   70.0, 'rgb(61,62,61)',
        70.5, 'rgb(65,66,65)',   71.0, 'rgb(69,70,69)',   71.5, 'rgb(73,74,73)',
        72.0, 'rgb(77,78,77)',   72.5, 'rgb(81,82,81)',   73.0, 'rgb(85,86,85)',
        73.5, 'rgb(89,90,89)',   74.0, 'rgb(93,94,93)',   74.5, 'rgb(97,98,97)',
        75.0, 'rgb(101,102,101)',75.5, 'rgb(105,106,105)',76.0, 'rgb(109,110,109)',
        76.5, 'rgb(113,114,113)',77.0, 'rgb(117,118,117)',77.5, 'rgb(121,122,121)',
        78.0, 'rgb(125,126,125)',78.5, 'rgb(129,130,129)',79.0, 'rgb(133,134,133)',
        79.5, 'rgb(137,138,137)',80.0, 'rgb(142,142,142)',80.5, 'rgb(146,146,146)',
        81.0, 'rgb(150,150,150)',81.5, 'rgb(154,154,154)',82.0, 'rgb(158,158,158)',
        82.5, 'rgb(162,162,162)',83.0, 'rgb(166,166,166)',83.5, 'rgb(170,170,170)',
        84.0, 'rgb(174,174,174)',84.5, 'rgb(178,178,178)',85.0, 'rgb(182,182,182)',
        85.5, 'rgb(186,186,186)',86.0, 'rgb(190,190,190)',86.5, 'rgb(194,194,194)',
        87.0, 'rgb(198,198,198)',87.5, 'rgb(202,202,202)',88.0, 'rgb(206,206,206)',
        88.5, 'rgb(210,210,210)',89.0, 'rgb(214,214,214)',89.5, 'rgb(218,218,218)',
        90.0, 'rgb(222,222,222)',90.5, 'rgb(226,226,226)',91.0, 'rgb(230,230,230)',
        91.5, 'rgb(234,234,234)',92.0, 'rgb(238,238,238)',92.5, 'rgb(242,242,242)',
        93.0, 'rgb(246,246,246)',93.5, 'rgb(250,250,250)',94.0, 'rgb(254,254,254)',
        94.5, 'rgb(258,258,258)',95.0, 'rgb(262,262,262)',100.0,'rgb(262,262,262)'
    ];

    map.on('load', () => {
        map.addSource('radar-vector', {
            type: 'geojson',
            data: `/api/radar_polygons?station=${currentStation}`
        });

        map.addLayer({
            id: 'radar-layer',
            type: 'fill',
            source: 'radar-vector',
            paint: {
                'fill-color': dbzExpression,
                'fill-opacity': 0.8,
                'fill-antialias': false
            }
        });

        setInterval(() => {
            loadRadarData(currentStation);
        }, 30000);
    });

    function loadRadarData(stationId) {
        fetch(`/api/radar_polygons?station=${stationId}`)
            .then(res => res.json())
            .then(data => {
                requestAnimationFrame(() => {
                    const source = map.getSource('radar-vector');
                    if (source) source.setData(data);
                });
            });
    }

    function switchStation(stationId, btn) {
        currentStation = stationId;
        document.querySelectorAll('.station-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        const coords = {
            'KTLX': [-97.2778, 35.3331],
            'KBOX': [-71.1306, 42.1325],
            'KHGX': [-95.0792, 29.4719]
        };
        
        map.flyTo({ center: coords[stationId], zoom: 6 });
        loadRadarData(stationId);
    }
</script>

</body>
</html>
"""
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

# ═══════════════════════════════════════════════════════════════════════════
# Main Starter
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Pre-fetching initialen Datensatz für Hauptstation KTLX...")
    update_station_data("KTLX")
    
    generate_index_html()
    
    # Hintergrund-Thread für fortlaufende S3 Live-Updates starten
    worker = threading.Thread(target=live_background_worker, daemon=True)
    worker.start()
    
    # Browser automatisch öffnen
    time.sleep(1)
    webbrowser.open(f"http://localhost:{PORT}")
    
    print(f"\n🚀 Starte den Mathematisch perfekten Multi-Vector Server auf http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)