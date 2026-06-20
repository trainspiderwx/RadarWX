"""
Multi-Stationen / Multi-Produkt NEXRAD Server - LEVEL 3 (NIDS) Version
- Mehrere Produkte (Reflectivity, Velocity, Correlation Coefficient, ...)
- Eigene .pal-Farbpaletten (GR2Analyst-Format) hoch- und runterladbar
- Stationen werden als "Billboards" direkt auf der Karte angezeigt (kein Button-Panel)
- Legende links wird live aus der aktiven Palette generiert
"""

import os
import time
import struct
import threading
import webbrowser
import zlib
import bz2
import colorsys
import numpy as np
import boto3
from flask import Flask, jsonify, send_file, request
from botocore import UNSIGNED
from botocore.config import Config
from datetime import datetime, timezone, timedelta

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Erweiterte Stationsliste (Koordinaten sind Naeherungswerte fuer die jeweilige
# Radaranlage - bei Bedarf gegen die offizielle NOAA-Liste pruefen/anpassen)
STATIONS = {
    "KTLX": {"name": "Oklahoma City, OK", "lat": 35.3331, "lon": -97.2778},
    "KEWX": {"name": "Austin/San Antonio, TX", "lat": 29.7039, "lon": -98.0286},
    "KHGX": {"name": "Houston/Galveston, TX", "lat": 29.4719, "lon": -95.0792},
    "KFWS": {"name": "Dallas/Fort Worth, TX", "lat": 32.5731, "lon": -97.3031},
    "KLBB": {"name": "Lubbock, TX", "lat": 33.6541, "lon": -101.8141},
    "KBOX": {"name": "Boston, MA", "lat": 41.9558, "lon": -71.1369},
    "KOKX": {"name": "New York City, NY", "lat": 40.8656, "lon": -72.8639},
    "KDIX": {"name": "Philadelphia, PA", "lat": 39.9470, "lon": -74.4109},
    "KLWX": {"name": "Washington D.C./Sterling, VA", "lat": 38.9753, "lon": -77.4778},
    "KMHX": {"name": "Newport, NC", "lat": 34.7759, "lon": -76.8762},
    "KTBW": {"name": "Tampa Bay, FL", "lat": 27.7056, "lon": -82.4019},
    "KAMX": {"name": "Miami, FL", "lat": 25.6111, "lon": -80.4128},
    "KAMA": {"name": "Amarillo, TX", "lat": 35.2334, "lon": -101.7092},
    "KFFC": {"name": "Atlanta, GA", "lat": 33.3636, "lon": -84.5658},
    "KLIX": {"name": "New Orleans, LA", "lat": 30.3367, "lon": -89.8256},
    "KLOT": {"name": "Chicago, IL", "lat": 41.6044, "lon": -88.0847},
    "KMKX": {"name": "Milwaukee, WI", "lat": 42.9678, "lon": -88.5506},
    "KEAX": {"name": "Kansas City, MO", "lat": 38.8103, "lon": -94.2644},
    "KICT": {"name": "Wichita, KS", "lat": 37.6546, "lon": -97.4431},
    "KFTG": {"name": "Denver, CO", "lat": 39.7867, "lon": -104.5458},
    "KMSO": {"name": "Missoula, MT", "lat": 46.9169, "lon": -114.0931},
    "KOTX": {"name": "Spokane, WA", "lat": 47.6803, "lon": -117.6261},
    "KATX": {"name": "Seattle, WA", "lat": 48.1947, "lon": -122.4956},
    "KMUX": {"name": "San Francisco, CA", "lat": 37.1551, "lon": -121.8983},
    "KVTX": {"name": "Los Angeles, CA", "lat": 34.4117, "lon": -119.1794},
    "KNKX": {"name": "San Diego, CA", "lat": 32.9189, "lon": -117.0419},
    "KIWA": {"name": "Phoenix, AZ", "lat": 33.2891, "lon": -111.6700},
    "KESX": {"name": "Las Vegas, NV", "lat": 35.7011, "lon": -114.8914},
}

L3_BUCKET = "unidata-nexrad-level3"
PORT = 8000

s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))

# Welche NIDS-Produktcodes stehen hinter welchem Kuerzel (wie es auch in .pal-
# Dateien als "product:" auftaucht)? supports_rf -> Range-Folded-Erkennung aktiv.
PRODUCT_CATALOG = {
    "BR":  {"l3_code": "N0B", "label": "Reflectivity",               "supports_rf": False},
    "BV":  {"l3_code": "N0U", "label": "Velocity",                    "supports_rf": True},
    "CC":  {"l3_code": "N0C", "label": "Correlation Coefficient",     "supports_rf": False},
    "ZDR": {"l3_code": "N0X", "label": "Differential Reflectivity",   "supports_rf": False},
    "KDP": {"l3_code": "N0K", "label": "Specific Differential Phase", "supports_rf": False},
    "SW":  {"l3_code": "N0W", "label": "Spectrum Width",              "supports_rf": False},
}

# Cache: (station_id, product) -> geparste Sweep-Daten im RAM
_radar_cache: dict[tuple, dict] = {}
# Welche (station, produkt)-Kombinationen wurden aktiv angefragt? Nur diese
# werden im Hintergrund automatisch aktualisiert.
_active_combos: set[tuple] = set()

# Palette-Speicher: Produktcode -> fertig aufbereitete Palette (JSON-faehig)
PALETTES: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════════════════
# .pal Farbpaletten - Parsing (GR2Analyst-Format)
# ═══════════════════════════════════════════════════════════════════════════

def _clamp_rgb(nums) -> tuple:
    """Manche .pal-Dateien enthalten Werte > 255 (z.B. als 'maximal heller'
    Marker) - hart auf den gueltigen 0-255 Bereich clampen."""
    return tuple(int(max(0, min(255, round(n)))) for n in nums[:3])


def _rgb_str(c) -> str:
    r, g, b = c
    return f"rgb({r},{g},{b})"


def parse_pal(text: str) -> dict:
    """Parst eine .pal-Datei im GR2Analyst-Format.

    Unterstuetzte Zeilen (Gross-/Kleinschreibung egal):
      product: XXX
      units: ...
      step: <zahl>
      scale: <zahl>
      color: <wert> <r> <g> <b> [<r2> <g2> <b2>]   (2. RGB = interner Verlauf)
      rf: <r> <g> <b>                              (Range-Folded Sonderfarbe)
    """
    meta = {"product": None, "units": "", "step": None, "scale": 1.0}
    raw_stops = []   # (wert, (r,g,b), (r2,g2,b2)|None)
    rf_color = None

    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip().lower()
        rest = rest.strip()

        if key == "product":
            meta["product"] = rest.strip().upper() or None
        elif key == "units":
            meta["units"] = rest.strip()
        elif key == "step":
            try:
                meta["step"] = float(rest)
            except ValueError:
                pass
        elif key == "scale":
            try:
                meta["scale"] = float(rest)
            except ValueError:
                pass
        elif key == "rf":
            try:
                nums = [float(x) for x in rest.split()]
            except ValueError:
                continue
            if len(nums) >= 3:
                rf_color = _clamp_rgb(nums[:3])
        elif key in ("color", "solidcolor"):
            try:
                nums = [float(x) for x in rest.split()]
            except ValueError:
                continue
            if len(nums) < 4:
                continue
            value = nums[0]
            c1 = _clamp_rgb(nums[1:4])
            c2 = _clamp_rgb(nums[4:7]) if len(nums) >= 7 else None
            raw_stops.append((value, c1, c2))
        # unbekannte Schluessel werden ignoriert

    raw_stops.sort(key=lambda s: s[0])
    return {
        "product": meta["product"],
        "units": meta["units"],
        "step": meta["step"],
        "scale": meta["scale"] if meta["scale"] else 1.0,
        "rf_color": rf_color,
        "raw_stops": raw_stops,
    }


def flatten_stops(raw_stops) -> list:
    """Wandelt die geparsten (wert, c1, c2) Eintraege in eine streng monoton
    steigende Stop-Liste [(wert, (r,g,b)), ...] fuer MapLibres 'interpolate'
    Expression um.

    GR2Analyst-Logik: hat ein Eintrag eine zweite Farbe (c2), gibt es einen
    internen Farbverlauf von c1 -> c2 *innerhalb* dieses Segments, gefolgt von
    einem harten Sprung auf die Startfarbe des naechsten Eintrags. Ohne c2
    wird stattdessen weich in die Startfarbe des naechsten Eintrags
    uebergeblendet (klassischer kontinuierlicher Verlauf).
    """
    EPS = 1e-3
    out = []
    n = len(raw_stops)
    for i, (val, c1, c2) in enumerate(raw_stops):
        out.append((val, c1))
        if i < n - 1 and c2 is not None:
            next_val = raw_stops[i + 1][0]
            end_val = max(val, next_val - EPS)
            out.append((end_val, c2))

    # Strikte Monotonie erzwingen (Pflicht fuer MapLibre 'interpolate')
    cleaned = []
    last_v = None
    for v, c in out:
        if last_v is not None and v <= last_v:
            v = last_v + EPS
        cleaned.append((v, c))
        last_v = v
    return cleaned


def _store_palette(code: str, parsed: dict):
    stops = flatten_stops(parsed["raw_stops"])
    if len(stops) < 2:
        return False
    PALETTES[code] = {
        "product": code,
        "units": parsed["units"],
        "step": parsed["step"],
        "scale": parsed["scale"],
        "rf_color": _rgb_str(parsed["rf_color"]) if parsed["rf_color"] else None,
        "stops": [[v, _rgb_str(c)] for v, c in stops],
    }
    return True


def _fallback_palette(code: str, units: str, vmin: float, vmax: float, n: int = 6):
    """Generiert eine einfache Blau->Rot Verlaufspalette fuer Produkte, fuer
    die noch keine echte .pal-Datei hochgeladen wurde."""
    stops = []
    for i in range(n):
        t = i / (n - 1)
        v = vmin + t * (vmax - vmin)
        hue = (1 - t) * 0.66  # 0.66 = blau, 0.0 = rot
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        stops.append([round(v, 2), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"])
    PALETTES[code] = {
        "product": code,
        "units": units,
        "step": round((vmax - vmin) / (n - 1), 2),
        "scale": 1.0,
        "rf_color": None,
        "stops": stops,
    }


# ── Standard-Paletten (von dir bereitgestellt, GR2Analyst-Format) ──────────

_PAL_BR = """
product:BR
units: dBZ
step: 5
COLOR: 0       1 243 247
COLOR: 0.5     3 231 239
COLOR: 1.0     5 219 231
COLOR: 1.5     7 207 223
COLOR: 2.0     9 195 215
COLOR: 2.5    11 183 207
COLOR: 3.0    13 171 199
COLOR: 3.5    15 195 191
COLOR: 4.0    17 147 183
COLOR: 4.5    19 135 175
COLOR: 5.0    21 123 167
COLOR: 5.5    23 112 159
COLOR: 6.0    21 114 163
COLOR: 6.5    20 117 168
COLOR: 7.0    19 120 173
COLOR: 7.5    18 123 178
COLOR: 8.0    17 126 182
COLOR: 8.5    16 129 187
COLOR: 9.0    15 132 192
COLOR: 9.5    14 135 197
COLOR: 10.0   12 137 201
COLOR: 10.5   11 140 206
COLOR: 11.0   10 143 211
COLOR: 11.5    9 146 216
COLOR: 12.0    8 149 220
COLOR: 12.5    7 152 255
COLOR: 13.0    6 155 230
COLOR: 13.5    5 158 235
COLOR: 14.0   21 191 180
COLOR: 14.5   37 225 125
COLOR: 15.0   36 221 121
COLOR: 15.5   35 218 118
COLOR: 16.0   34 214 115
COLOR: 16.5   33 211 112
COLOR: 17.0   32 207 108
COLOR: 17.5   31 204 105
COLOR: 18.0   30 200 102
COLOR: 18.5   29 197  99
COLOR: 19.0   28 194  96
COLOR: 19.5   27 190  93
COLOR: 20.0   26 187  90
COLOR: 20.5   28 184  87
COLOR: 21.0   24 180  84
COLOR: 21.5   24 177  81
COLOR: 22.0   23 174  77
COLOR: 22.5   22 170  74
COLOR: 23.0   21 167  71
COLOR: 23.5   20 164  68
COLOR: 24.0   19 160  65
COLOR: 24.5   18 157  62
COLOR: 25.0   17 154  59
COLOR: 25.5   16 150  56
COLOR: 26.0   15 147  53
COLOR: 26.5   15 144  50
COLOR: 27.0   14 140  46
COLOR: 27.5   13 137  43
COLOR: 28.0   12 133  40
COLOR: 28.5   11 130  37
COLOR: 29.0   10 127  34
COLOR: 29.5    9 123  31
COLOR: 30.0    8 120  27
COLOR: 30.5    7 117  24
COLOR: 31.0    6 113  21
COLOR: 31.5    5 110  18
COLOR: 32.0    4 107  15
COLOR: 32.5    3 103  12
COLOR: 33.0    2 100   9
COLOR: 33.5    1  96   5
COLOR: 34.0  128 175  19
COLOR: 34.5  255 255  33
COLOR: 35.0  255 247  28
COLOR: 35.5  255 239  23
COLOR: 36.0  255 231  18
COLOR: 36.5  255 223  14
COLOR: 37.0  255 215   9
COLOR: 37.5  255 207   4
COLOR: 38.0  255 199   0
COLOR: 38.5  255 191   0
COLOR: 39.0  255 183   0
COLOR: 39.5  255 175   0
COLOR: 40.0  255 157   0
COLOR: 40.5  255 140   0
COLOR: 41.0  255 122   0
COLOR: 41.5  255 105   0
COLOR: 42.0  255  87   0
COLOR: 42.5  255  70   0
COLOR: 43.0  255  52   0
COLOR: 43.5  255  35   0
COLOR: 44.0  255  17   0
COLOR: 44.5  255   0   0
COLOR: 45.0  249   0   0
COLOR: 45.5  244   0   0
COLOR: 46.0  239   0   0
COLOR: 46.5  233   0   0
COLOR: 47.0  228   0   0
COLOR: 47.5  223   0   0
COLOR: 48.0  217   0   0
COLOR: 48.5  212   0   0
COLOR: 49.0  207   0   0
COLOR: 49.5  201   0   0
COLOR: 50.0  195   0   0
COLOR: 50.5  190   0   0
COLOR: 51.0  185   0   0
COLOR: 51.5  180   0   0
COLOR: 52.0  175   0   0
COLOR: 52.5  170   0   0
COLOR: 53.0  165   0   0
COLOR: 53.5  160   0   0
COLOR: 54.0  154   0   0
COLOR: 54.5  180   0 180
COLOR: 55.0  186   9 185
COLOR: 55.5  192  19 190
COLOR: 56.0  198  29 195
COLOR: 56.5  204  39 201
COLOR: 57.0  210  49 206
COLOR: 57.5  216  59 211
COLOR: 58.0  223  68 216
COLOR: 58.5  229  78 222
COLOR: 59.0  235  88 227
COLOR: 59.5  241  98 232
COLOR: 60.0  247 108 237
COLOR: 60.5  253 117 243
COLOR: 61.0  232 109 232
COLOR: 61.5  212 104 204
COLOR: 62.0  192  93 184
COLOR: 62.5  171  85 165
COLOR: 63.0  151  77 146
COLOR: 63.5  131  69 126
COLOR: 64.0  111  61 107
COLOR: 64.5   90  53  88
COLOR: 65.0   70  45  68
COLOR: 65.5   50  37  49
COLOR: 66.0   29  30  29
COLOR: 66.5   33  34  33
COLOR: 67.0   37  38  37
COLOR: 67.5   41  42  41
COLOR: 68.0   45  46  45
COLOR: 68.5   49  50  49
COLOR: 69.0   53  54  53
COLOR: 69.5   57  58  57
COLOR: 70.0   61  62  61
COLOR: 70.5   65  66  65
COLOR: 71.0   69  70  69
COLOR: 71.5   73  74  73
COLOR: 72.0   77  78  77
COLOR: 72.5   81  82  81
COLOR: 73.0   85  86  85
COLOR: 73.5   89  90  89
COLOR: 74.0   93  94  93
COLOR: 74.5   97  98  97
COLOR: 75.0  101 102 101
COLOR: 75.5  105 106 105
COLOR: 76.0  109 110 109
COLOR: 76.5  113 114 113
COLOR: 77.0  117 118 117
COLOR: 77.5  121 122 121
COLOR: 78.0  125 126 125
COLOR: 78.5  129 130 129
COLOR: 79.0  133 134 133
COLOR: 79.5  137 138 137
COLOR: 80.0  142 142 142
COLOR: 80.5  146 146 146
COLOR: 81.0  150 150 150
COLOR: 81.5  154 154 154
COLOR: 82.0  158 158 158
COLOR: 82.5  162 162 162
COLOR: 83.0  166 166 166
COLOR: 83.5  170 170 170
COLOR: 84.0  174 174 174
COLOR: 84.5  178 178 178
COLOR: 85.0  182 182 182
COLOR: 85.5  186 186 186
COLOR: 86.0  190 190 190
COLOR: 86.5  194 194 194
COLOR: 87.0  198 198 198
COLOR: 87.5  202 202 202
COLOR: 88.0  206 206 206
COLOR: 88.5  210 210 210
COLOR: 89.0  214 214 214
COLOR: 89.5  218 218 218
COLOR: 90.0  222 222 222
COLOR: 90.5  226 226 226
COLOR: 91.0  230 230 230
COLOR: 91.5  234 234 234
COLOR: 92.0  238 238 238
COLOR: 92.5  242 242 242
COLOR: 93.0  246 246 246
COLOR: 93.5  250 250 250
COLOR: 94.0  254 254 254
COLOR: 94.5  258 258 258
COLOR: 95.0  262 262 262
COLOR: 100.0 262 262 262
"""

_PAL_BV = """
product: BV
units: MPH
step: 10
scale: 2.237
color: -120 255 0 128
color: -90.5 0 0 160
color: -70 0 224 255
color: -60 0 255 225
color: -50 160 255 208
color: -40 0 255 0
color: -10 16 96 16
color: -9.99 16 96 16
color: -.01 112 128 112
color: 0 144 128 144
color: 10 135 69 88
color: 20 119 0 0
color: 45 255 0 0
color: 55 255 128 0
color: 65 255 255 0
color: 100 190 190 0
color: 120 0 0 0
RF: 128 0 208
"""

_PAL_CC = """
Product: CC
Units: %
Scale: 100
Step: 4
Color: 100 25 25 25
Color: 96 255 11 0
Color: 92 255 187 0
Color: 88 121 235 17 130 230 0
Color: 84 93 229 119
Color: 80 120 120 255
Color: 76 30 30 255
Color: 72 0 0 203
Color: 68 0 0 163
Color: 64 0 0 146
Color: 60 0 0 130
Color: 56 0 0 113
Color: 52 9 0 101
Color: 48 22 0 88
Color: 44 30 0 79
Color: 40 28 0 73
Color: 36 20 0 69
Color: 32 25 0 64
Color: 28 23 0 59
Color: 24 22 0 55
Color: 20 20 0 50
"""


def _load_default_palettes():
    for code, text in (("BR", _PAL_BR), ("BV", _PAL_BV), ("CC", _PAL_CC)):
        parsed = parse_pal(text)
        _store_palette(code, parsed)
    # Fallback-Paletten fuer Produkte ohne eigene .pal-Datei (bis der Nutzer
    # eine echte GR2Analyst-Palette hochlaedt)
    _fallback_palette("ZDR", "dB", -2, 6)
    _fallback_palette("KDP", "deg/km", -2, 6)
    _fallback_palette("SW", "kt", 0, 30)


_load_default_palettes()


# ═══════════════════════════════════════════════════════════════════════════
# NIDS (Level 3) Parsing
# ═══════════════════════════════════════════════════════════════════════════

def _skip_wmo_header(data: bytes) -> int:
    pos = 0
    while pos < min(200, len(data)):
        b = data[pos]
        if 0x20 <= b < 0x80:
            idx = data.find(b'\r\r\n', pos)
            if idx != -1 and idx < pos + 100:
                pos = idx + 3
                continue
            idx = data.find(b'\n', pos)
            if idx != -1 and idx < pos + 100:
                pos = idx + 1
                continue
        break
    return pos


def _parse_nids(raw: bytes) -> dict:
    d = raw
    pos = _skip_wmo_header(d)
    msg_start = pos

    pos += 18  # Message Header

    pdb = pos
    lat = struct.unpack_from('>i', d, pdb + 2)[0] / 1000.0
    lon = struct.unpack_from('>i', d, pdb + 6)[0] / 1000.0

    vol_date = struct.unpack_from('>H', d, pdb + 22)[0]
    vol_time = struct.unpack_from('>I', d, pdb + 24)[0]
    try:
        scan_time = datetime(1970, 1, 1) + timedelta(days=vol_date - 1, seconds=vol_time)
    except Exception:
        scan_time = datetime.now(timezone.utc)

    comp_method = struct.unpack_from('>H', d, pdb + 82)[0]
    pdb_end = msg_start + 120
    if comp_method == 1:
        d = d[:pdb_end] + bz2.decompress(d[pdb_end:])
    elif comp_method == 2:
        d = d[:pdb_end] + zlib.decompress(d[pdb_end:])

    sym_hw = struct.unpack_from('>I', d, pdb + 90)[0]
    if sym_hw == 0:
        raise ValueError("NIDS-Datei enthaelt keinen Symbology Block")
    sym_pos = msg_start + sym_hw * 2

    packet_pos = sym_pos + 10 + 6
    packet_code = struct.unpack_from('>H', d, packet_pos)[0]

    if packet_code == 16:
        radials = _parse_packet_16(d, packet_pos)
    elif packet_code == 0xAF1F:
        radials = _parse_packet_af1f(d, packet_pos)
    else:
        raise ValueError(f"Nicht unterstuetzter NIDS-Packet-Code 0x{packet_code:04X}")

    radials["radar_lat"] = lat
    radials["radar_lon"] = lon
    radials["scan_time"] = scan_time
    return radials


def _parse_packet_16(d: bytes, pos: int) -> dict:
    """Legacy 8-bit Format. ACHTUNG: die Werteformel (*0.5 - 33.0) ist die
    klassische Reflectivity-Codierung. Moderne Super-Res-Produkte (N0B, N0U,
    N0C, ...) verwenden so gut wie immer das generische AF1F/Radial-Format
    unten, das Scale/Offset direkt aus der Datei liest und damit fuer JEDES
    Produkt korrekt ist. Dieser Pfad ist nur ein Fallback fuer aeltere Dateien."""
    first_bin   = struct.unpack_from('>H', d, pos + 2)[0]
    num_bins    = struct.unpack_from('>H', d, pos + 4)[0]
    num_radials = struct.unpack_from('>H', d, pos + 12)[0]

    azimuths = np.empty(num_radials, dtype=np.float32)
    raw_data = np.zeros((num_radials, num_bins), dtype=np.uint8)

    rpos = pos + 14
    for i in range(num_radials):
        nbytes   = struct.unpack_from('>H', d, rpos)[0]
        start_az = struct.unpack_from('>H', d, rpos + 2)[0] / 10.0
        delta_az = struct.unpack_from('>H', d, rpos + 4)[0] / 10.0
        azimuths[i] = start_az + delta_az / 2.0

        n = min(nbytes, num_bins)
        gate_bytes = np.frombuffer(d, dtype=np.uint8, count=n, offset=rpos + 6)
        raw_data[i, :n] = gate_bytes

        rpos += 6 + nbytes
        if nbytes % 2:
            rpos += 1

    dbz = raw_data.astype(np.float32) * 0.5 - 33.0
    dbz[raw_data <= 1] = np.nan
    ranges = (first_bin + np.arange(num_bins, dtype=np.float32)) * 0.25  # km

    return {"azimuths": azimuths, "ranges": ranges, "data": dbz, "raw": raw_data}


def _parse_packet_af1f(d: bytes, pos: int) -> dict:
    data_len = struct.unpack_from('>I', d, pos + 4)[0]
    cpos = pos + 8
    end = cpos + data_len
    while cpos < end:
        comp_type = struct.unpack_from('>H', d, cpos)[0]
        if comp_type == 1:
            return _parse_generic_radial_component(d, cpos)
        comp_len = struct.unpack_from('>I', d, cpos + 4)[0]
        cpos += 8 + comp_len
    raise ValueError("Keine Radial-Komponente im AF1F-Paket gefunden")


def _parse_generic_radial_component(d: bytes, cpos: int) -> dict:
    desc_pos = cpos + 6
    num_gates      = struct.unpack_from('>I', d, desc_pos + 36)[0]
    first_gate_m   = struct.unpack_from('>I', d, desc_pos + 40)[0]
    gate_size_m    = struct.unpack_from('>I', d, desc_pos + 44)[0]
    word_size_bits = struct.unpack_from('>H', d, desc_pos + 52)[0]
    scale          = struct.unpack_from('>f', d, desc_pos + 54)[0]
    offset         = struct.unpack_from('>f', d, desc_pos + 58)[0]
    bytes_per_gate = max(word_size_bits // 8, 1)
    num_radials    = struct.unpack_from('>H', d, desc_pos + 62)[0]

    radial_pos = desc_pos + 64
    azimuths = np.empty(num_radials, dtype=np.float32)
    raw_data = np.zeros((num_radials, num_gates), dtype=np.float32)

    for i in range(num_radials):
        start_az = struct.unpack_from('>H', d, radial_pos)[0] / 10.0
        delta_az = struct.unpack_from('>H', d, radial_pos + 2)[0] / 10.0
        azimuths[i] = start_az + delta_az / 2.0

        gate_start = radial_pos + 4
        if bytes_per_gate == 1:
            raw = np.frombuffer(d, dtype=np.uint8, count=num_gates, offset=gate_start)
            raw_data[i, :] = raw.astype(np.float32)
        elif bytes_per_gate == 2:
            raw = np.frombuffer(d, dtype='>u2', count=num_gates, offset=gate_start)
            raw_data[i, :] = raw.astype(np.float32)
        else:
            raw = np.frombuffer(d, dtype=np.uint8, count=num_gates * bytes_per_gate, offset=gate_start)
            raw_data[i, :] = raw.reshape(num_gates, bytes_per_gate)[:, 0].astype(np.float32)

        radial_pos += 4 + num_gates * bytes_per_gate

    if scale != 0:
        dbz = (raw_data - offset) / scale
    else:
        dbz = raw_data * 0.5 - 33.0
    dbz[raw_data <= 1] = np.nan
    ranges = (first_gate_m + np.arange(num_gates, dtype=np.float32) * gate_size_m) / 1000.0  # km

    return {"azimuths": azimuths, "ranges": ranges, "data": dbz, "raw": raw_data}


# ═══════════════════════════════════════════════════════════════════════════
# S3 Data Fetcher (Level 3 Live Feed)
# ═══════════════════════════════════════════════════════════════════════════

def _l3_station(station_id: str) -> str:
    return station_id[1:] if (len(station_id) == 4 and station_id[0] == "K") else station_id


def _find_latest_key(station_id: str, l3_code: str) -> str | None:
    l3id = _l3_station(station_id)
    for days_back in (0, 1):
        dt = datetime.now(timezone.utc) - timedelta(days=days_back)
        prefix = f"{l3id}_{l3_code}_{dt:%Y_%m_%d}"
        try:
            resp = s3.list_objects_v2(Bucket=L3_BUCKET, Prefix=prefix)
        except Exception as e:
            print(f"S3-Listfehler fuer {station_id}/{l3_code}: {e}")
            return None
        contents = resp.get("Contents")
        if contents:
            latest = sorted(contents, key=lambda o: o["Key"])[-1]
            return latest["Key"]
    return None


def update_station_data(station_id: str, product: str) -> bool:
    info = PRODUCT_CATALOG.get(product)
    if info is None or not info.get("l3_code"):
        return False
    l3_code = info["l3_code"]

    latest_key = _find_latest_key(station_id, l3_code)
    if latest_key is None:
        return False

    cache_key = (station_id, product)
    if _radar_cache.get(cache_key, {}).get("key") == latest_key:
        return True

    try:
        print(f"[S3 Live/L3] {station_id}/{product}: lade {latest_key}")
        obj = s3.get_object(Bucket=L3_BUCKET, Key=latest_key)
        raw = obj["Body"].read()
        parsed = _parse_nids(raw)

        rf_mask = None
        if info.get("supports_rf") and parsed.get("raw") is not None:
            rf_mask = (parsed["raw"] == 1)

        _radar_cache[cache_key] = {
            "key": latest_key,
            "azimuths": parsed["azimuths"],
            "ranges": parsed["ranges"],
            "data": parsed["data"],
            "rf_mask": rf_mask,
            "scan_time": parsed["scan_time"],
        }
        return True
    except Exception as e:
        print(f"Fehler beim S3-Update ({product}) fuer {station_id}: {e}")
        return False


def live_background_worker():
    """Aktualisiert nur (Station, Produkt)-Kombinationen, die tatsaechlich
    schon mal angefragt wurden (= auf der Karte aktiviert sind)."""
    while True:
        for combo in list(_active_combos):
            update_station_data(*combo)
        time.sleep(30)


# ═══════════════════════════════════════════════════════════════════════════
# Flask API
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def serve_index():
    return send_file("index.html")


@app.route("/api/stations")
def get_stations():
    return jsonify([{"id": sid, **info} for sid, info in STATIONS.items()])


@app.route("/api/products")
def get_products():
    return jsonify([
        {"code": code, "label": info.get("label", code),
         "l3_code": info.get("l3_code"), "has_palette": code in PALETTES}
        for code, info in PRODUCT_CATALOG.items()
    ])


@app.route("/api/palette")
def get_palette():
    code = (request.args.get("product") or "").strip().upper()
    pal = PALETTES.get(code)
    if pal is None:
        return jsonify({"error": f"Keine Palette fuer Produkt '{code}'"}), 404
    return jsonify(pal)


@app.route("/api/palette/upload", methods=["POST"])
def upload_palette():
    if "file" not in request.files:
        return jsonify({"error": "Keine Datei erhalten"}), 400
    f = request.files["file"]
    try:
        text = f.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return jsonify({"error": f"Datei konnte nicht gelesen werden: {e}"}), 400

    parsed = parse_pal(text)
    code = (request.form.get("product") or parsed["product"] or "").strip().upper()
    if not code:
        return jsonify({"error": "Kein Produktcode gefunden (weder in Datei noch angegeben)"}), 400

    if not _store_palette(code, parsed):
        return jsonify({"error": "Palette enthaelt zu wenige gueltige Farbpunkte (mind. 2 noetig)"}), 400

    l3_code = (request.form.get("l3_code") or "").strip().upper()
    if code not in PRODUCT_CATALOG:
        if not l3_code:
            return jsonify({
                "status": "palette_only",
                "warning": (f"Palette fuer '{code}' gespeichert, aber es ist noch kein NIDS-"
                            f"Produktcode hinterlegt - bitte 'l3_code' angeben (z.B. N0K), "
                            f"damit Daten geladen werden koennen."),
                "product": code,
            }), 200
        PRODUCT_CATALOG[code] = {"l3_code": l3_code, "label": code, "supports_rf": False}
    elif l3_code:
        PRODUCT_CATALOG[code]["l3_code"] = l3_code

    return jsonify({"status": "ok", "product": code,
                     "label": PRODUCT_CATALOG.get(code, {}).get("label", code)})


@app.route("/api/radar_polygons")
def get_radar_polygons():
    station_id = request.args.get("station")
    product = (request.args.get("product") or "BR").strip().upper()

    if not station_id or station_id not in STATIONS:
        return jsonify({"error": "Ungueltige oder fehlende Station"}), 400
    if product not in PRODUCT_CATALOG:
        return jsonify({"error": f"Unbekanntes Produkt '{product}'"}), 400

    _active_combos.add((station_id, product))

    cache_key = (station_id, product)
    if cache_key not in _radar_cache:
        update_station_data(station_id, product)
        if cache_key not in _radar_cache:
            return jsonify({"error": "Keine Daten verfuegbar"}), 500

    cache = _radar_cache[cache_key]
    azimuths = cache["azimuths"]
    ranges = cache["ranges"]
    data = cache["data"]
    rf_mask = cache.get("rf_mask")

    st = STATIONS[station_id]
    lon_0, lat_0 = st["lon"], st["lat"]

    palette = PALETTES.get(product)
    cutoff = None
    if palette and palette["stops"]:
        scale = palette.get("scale") or 1.0
        if scale:
            cutoff = palette["stops"][0][0] / scale

    lat_to_km = 111.32
    lon_to_km = 111.32 * np.cos(np.radians(lat_0))

    features = []
    num_radials = len(azimuths)
    num_bins = len(ranges)

    az_bounds = np.zeros(num_radials + 1)
    for i in range(num_radials - 1):
        az_bounds[i + 1] = (azimuths[i] + azimuths[i + 1]) / 2.0
    az_bounds[0] = azimuths[0] - (azimuths[1] - azimuths[0]) / 2.0
    az_bounds[-1] = azimuths[-1] + (azimuths[-1] - azimuths[-2]) / 2.0

    gate_spacing = ranges[1] - ranges[0] if num_bins > 1 else 0.25
    r_bounds = np.zeros(num_bins + 1)
    r_bounds[:-1] = ranges - (gate_spacing / 2.0)
    r_bounds[-1] = ranges[-1] + (gate_spacing / 2.0)

    for az_idx in range(num_radials):
        theta1 = np.radians(az_bounds[az_idx])
        theta2 = np.radians(az_bounds[az_idx + 1])
        sin_t1, cos_t1 = np.sin(theta1), np.cos(theta1)
        sin_t2, cos_t2 = np.sin(theta2), np.cos(theta2)

        for r_idx in range(num_bins):
            val = data[az_idx, r_idx]
            is_rf = bool(rf_mask[az_idx, r_idx]) if rf_mask is not None else False

            if not is_rf:
                if np.isnan(val):
                    continue
                if cutoff is not None and val < cutoff:
                    continue

            out_val = 0.0 if is_rf else float(val)

            r_inner = r_bounds[r_idx]
            r_outer = r_bounds[r_idx + 1]

            p1_lon = lon_0 + (r_inner * sin_t1) / lon_to_km
            p1_lat = lat_0 + (r_inner * cos_t1) / lat_to_km
            p2_lon = lon_0 + (r_outer * sin_t1) / lon_to_km
            p2_lat = lat_0 + (r_outer * cos_t1) / lat_to_km
            p3_lon = lon_0 + (r_outer * sin_t2) / lon_to_km
            p3_lat = lat_0 + (r_outer * cos_t2) / lat_to_km
            p4_lon = lon_0 + (r_inner * sin_t2) / lon_to_km
            p4_lat = lat_0 + (r_inner * cos_t2) / lat_to_km

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [p1_lon, p1_lat], [p2_lon, p2_lat],
                        [p3_lon, p3_lat], [p4_lon, p4_lat],
                        [p1_lon, p1_lat],
                    ]],
                },
                "properties": {"value": out_val, "rf": is_rf},
            })

    return jsonify({"type": "FeatureCollection", "features": features})


# ═══════════════════════════════════════════════════════════════════════════
# Web-Interface (MapLibre Karte + Billboards + Legende)
# ═══════════════════════════════════════════════════════════════════════════

def generate_index_html():
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>NEXRAD Level 3 Multi-Product Vector Map</title>
    <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no" />
    <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
    <style>
        body { margin: 0; padding: 0; background-color: #111; font-family: sans-serif; }
        #map { position: absolute; top: 0; bottom: 0; width: 100%; }

        /* ── Billboard-Marker direkt auf den Radarstationen ── */
        .radar-billboard {
            width: 34px; height: 34px;
            border-radius: 50%;
            background: rgba(30,30,35,0.92);
            border: 2px solid #555;
            color: #aaa;
            display: flex; align-items: center; justify-content: center;
            font-size: 9px; font-weight: 700; letter-spacing: 0.3px;
            cursor: pointer;
            transition: transform 0.15s ease, border-color 0.15s ease, color 0.15s ease;
            user-select: none;
        }
        .radar-billboard:hover { border-color: #999; color: #eee; transform: scale(1.12); }
        .radar-billboard.active {
            background: #0078ff; border-color: #00d4ff; color: #fff;
            box-shadow: 0 0 14px rgba(0,212,255,0.85);
            animation: rb-pulse 1.8s infinite;
        }
        .radar-billboard.loading {
            border-color: #f9f225;
            animation: rb-spin 0.8s linear infinite;
        }
        @keyframes rb-pulse {
            0%   { box-shadow: 0 0 6px rgba(0,212,255,0.55); }
            50%  { box-shadow: 0 0 18px rgba(0,212,255,1); }
            100% { box-shadow: 0 0 6px rgba(0,212,255,0.55); }
        }
        @keyframes rb-spin {
            0%   { border-top-color: #f9f225; }
            50%  { border-top-color: transparent; }
            100% { border-top-color: #f9f225; }
        }

        /* ── Legenden-Panel links (GR2Analyst-Stil) ── */
        .legend-panel {
            position: absolute; top: 10px; left: 10px; z-index: 10;
            background: rgba(18,18,20,0.93); border: 1px solid #333;
            border-radius: 8px; padding: 12px; color: #ddd;
            width: 150px; font-size: 12px;
        }
        .legend-title { font-size: 12px; font-weight: 700; margin-bottom: 8px; color: #eee; }
        .product-tabs { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }
        .product-tab {
            flex: 1 1 auto; min-width: 38px; background: #222; color: #aaa; border: 1px solid #444;
            border-radius: 4px; padding: 5px 4px; cursor: pointer; font-size: 10.5px; font-weight: 700;
        }
        .product-tab.active { background: #0078ff; color: #fff; border-color: #00a2ff; }
        .add-product-btn {
            width: 100%; margin: 4px 0 8px 0; background: none; border: 1px dashed #444;
            color: #888; border-radius: 4px; padding: 4px; cursor: pointer; font-size: 10.5px;
        }
        .add-product-btn:hover { color: #ccc; border-color: #666; }
        .add-product-form { display: none; flex-direction: column; gap: 4px; margin-bottom: 8px; }
        .add-product-form input {
            background: #1a1a1a; border: 1px solid #444; color: #ddd; border-radius: 4px;
            padding: 4px 6px; font-size: 11px;
        }
        .scan-time { font-size: 10px; color: #777; margin-bottom: 8px; min-height: 12px; }

        .legend-bar-container { position: relative; height: 220px; margin: 10px 6px 10px 8px; display: flex; }
        .legend-bar { width: 22px; height: 100%; border-radius: 3px; border: 1px solid #444; }
        .legend-labels { position: relative; flex: 1; margin-left: 6px; height: 100%; }
        .legend-tick { position: absolute; left: 0; transform: translateY(-50%); font-size: 10px; color: #bbb; white-space: nowrap; }
        .legend-units { text-align: center; color: #888; margin-bottom: 8px; font-size: 11px; }
        .legend-rf { display: flex; align-items: center; gap: 6px; font-size: 10px; color: #999; margin-bottom: 8px; }
        .legend-rf-swatch { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #555; }

        .upload-btn {
            display: block; text-align: center; background: #222; border: 1px dashed #555;
            border-radius: 4px; padding: 6px; cursor: pointer; font-size: 11px; color: #ccc;
        }
        .upload-btn:hover { background: #2a2a2a; border-color: #777; }
    </style>
</head>
<body>

<div class="legend-panel">
    <div class="legend-title">NEXRAD Level 3</div>
    <div class="product-tabs" id="productTabs"></div>
    <button class="add-product-btn" id="addProductBtn" onclick="toggleAddProduct()">+ neues Produkt registrieren</button>
    <div class="add-product-form" id="addProductForm">
        <input type="text" id="newProductCode" placeholder="Kuerzel (z.B. SW)" maxlength="8">
        <input type="text" id="newProductL3" placeholder="NIDS-Code (z.B. N0W)" maxlength="8">
    </div>

    <div class="scan-time" id="scanTime"></div>

    <div class="legend-bar-container">
        <div class="legend-bar" id="legendBar"></div>
        <div class="legend-labels" id="legendLabels"></div>
    </div>
    <div class="legend-units" id="legendUnits"></div>
    <div class="legend-rf" id="legendRf" style="display:none;">
        <div class="legend-rf-swatch" id="legendRfSwatch"></div>
        <span>RF (Range Folded)</span>
    </div>

    <label class="upload-btn">
        📁 .pal hochladen
        <input type="file" id="palUpload" accept=".pal,.txt" style="display:none">
    </label>
</div>

<div id="map"></div>

<script>
    const map = new maplibregl.Map({
        container: 'map',
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        center: [-97.0, 38.0],
        zoom: 4
    });

    let markers = {};
    let activeStation = null;
    let activeProduct = 'BR';
    let currentPalette = null;
    let refreshTimer = null;

    function ensureLayer() {
        if (!map.getSource('radar-vector')) {
            map.addSource('radar-vector', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
            map.addLayer({
                id: 'radar-layer',
                type: 'fill',
                source: 'radar-vector',
                paint: { 'fill-color': '#888', 'fill-opacity': 0.75 }
            });
        }
    }

    function buildColorExpression(palette) {
        const valueExpr = ['*', ['get', 'value'], palette.scale || 1];
        const interpStops = [];
        palette.stops.forEach(([v, c]) => { interpStops.push(v, c); });
        const interpolateExpr = ['interpolate', ['linear'], valueExpr, ...interpStops];
        if (palette.rf_color) {
            return ['case', ['==', ['get', 'rf'], true], palette.rf_color, interpolateExpr];
        }
        return interpolateExpr;
    }

    function applyColorExpression() {
        if (!map.getLayer('radar-layer') || !currentPalette) return;
        map.setPaintProperty('radar-layer', 'fill-color', buildColorExpression(currentPalette));
    }

    function renderLegend(palette) {
        const bar = document.getElementById('legendBar');
        const labels = document.getElementById('legendLabels');
        const unitsEl = document.getElementById('legendUnits');
        const rfRow = document.getElementById('legendRf');
        const rfSwatch = document.getElementById('legendRfSwatch');

        if (!palette) {
            bar.style.background = '#222';
            labels.innerHTML = '';
            unitsEl.textContent = '';
            rfRow.style.display = 'none';
            return;
        }

        const stops = palette.stops;
        const minV = stops[0][0], maxV = stops[stops.length - 1][0];
        const range = (maxV - minV) || 1;
        const parts = stops.map(([v, c]) => {
            const pct = 100 - ((v - minV) / range) * 100;
            return `${c} ${pct.toFixed(2)}%`;
        });
        bar.style.background = `linear-gradient(to bottom, ${parts.join(', ')})`;

        labels.innerHTML = '';
        const step = palette.step && palette.step > 0 ? palette.step : (range / 5);
        let v = Math.ceil(minV / step) * step;
        let guard = 0;
        for (; v <= maxV + 1e-6 && guard < 60; v += step, guard++) {
            const pct = 100 - ((v - minV) / range) * 100;
            const lbl = document.createElement('div');
            lbl.className = 'legend-tick';
            lbl.style.top = `${pct}%`;
            lbl.textContent = Math.round(v * 100) / 100;
            labels.appendChild(lbl);
        }
        unitsEl.textContent = palette.units || '';

        if (palette.rf_color) {
            rfRow.style.display = 'flex';
            rfSwatch.style.background = palette.rf_color;
        } else {
            rfRow.style.display = 'none';
        }
    }

    async function loadStations() {
        const stations = await (await fetch('/api/stations')).json();
        stations.forEach(st => {
            const el = document.createElement('div');
            el.className = 'radar-billboard';
            el.textContent = st.id.replace(/^K/, '');
            el.title = `${st.id} - ${st.name}`;
            el.addEventListener('click', () => activateStation(st.id));
            const marker = new maplibregl.Marker({ element: el })
                .setLngLat([st.lon, st.lat])
                .addTo(map);
            markers[st.id] = { marker, el };
        });
    }

    async function loadProducts() {
        const products = await (await fetch('/api/products')).json();
        const tabsEl = document.getElementById('productTabs');
        tabsEl.innerHTML = '';
        products.forEach(p => {
            const btn = document.createElement('button');
            btn.className = 'product-tab' + (p.code === activeProduct ? ' active' : '');
            btn.textContent = p.code;
            btn.title = p.label;
            btn.onclick = () => switchProduct(p.code, btn);
            tabsEl.appendChild(btn);
        });
    }

    async function switchProductByCode(code) {
        let btn = null;
        document.querySelectorAll('.product-tab').forEach(b => { if (b.textContent === code) btn = b; });
        await switchProduct(code, btn);
    }

    async function switchProduct(code, btnEl) {
        activeProduct = code;
        document.querySelectorAll('.product-tab').forEach(b => b.classList.remove('active'));
        if (btnEl) btnEl.classList.add('active');
        await loadPalette(code);
        if (activeStation) {
            loadRadarData(activeStation, activeProduct);
        }
    }

    async function loadPalette(code) {
        const res = await fetch(`/api/palette?product=${code}`);
        if (!res.ok) { currentPalette = null; renderLegend(null); return; }
        currentPalette = await res.json();
        renderLegend(currentPalette);
        applyColorExpression();
    }

    async function activateStation(stationId) {
        if (activeStation && markers[activeStation]) {
            markers[activeStation].el.classList.remove('active');
        }
        activeStation = stationId;
        markers[stationId].el.classList.add('active');
        markers[stationId].el.classList.add('loading');

        map.flyTo({ center: markers[stationId].marker.getLngLat(), zoom: 7 });

        await loadRadarData(stationId, activeProduct);
        markers[stationId].el.classList.remove('loading');

        if (refreshTimer) clearInterval(refreshTimer);
        refreshTimer = setInterval(() => loadRadarData(activeStation, activeProduct), 20000);
    }

    async function loadRadarData(stationId, product) {
        ensureLayer();
        if (currentPalette) applyColorExpression();
        try {
            const res = await fetch(`/api/radar_polygons?station=${stationId}&product=${product}`);
            const geojson = await res.json();
            map.getSource('radar-vector').setData(geojson);
            document.getElementById('scanTime').textContent =
                `${stationId} / ${product} - ${new Date().toLocaleTimeString()}`;
        } catch (err) {
            console.error('Fehler beim Laden der Radardaten', err);
        }
    }

    function toggleAddProduct() {
        const f = document.getElementById('addProductForm');
        f.style.display = (f.style.display === 'flex') ? 'none' : 'flex';
    }

    document.getElementById('palUpload').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formVisible = document.getElementById('addProductForm').style.display === 'flex';
        const fd = new FormData();
        fd.append('file', file);
        if (formVisible) {
            const code = document.getElementById('newProductCode').value.trim();
            const l3 = document.getElementById('newProductL3').value.trim();
            if (code) fd.append('product', code);
            if (l3) fd.append('l3_code', l3);
        } else {
            fd.append('product', activeProduct);
        }

        const res = await fetch('/api/palette/upload', { method: 'POST', body: fd });
        const result = await res.json();
        if (result.error) { alert('Fehler: ' + result.error); return; }
        if (result.warning) { alert(result.warning); }

        await loadProducts();
        await switchProductByCode(result.product);
        document.getElementById('addProductForm').style.display = 'none';
        e.target.value = '';
    });

    map.on('load', async () => {
        await loadProducts();
        await loadPalette(activeProduct);
        await loadStations();
    });
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
    print("Pre-fetching initialen Datensatz (BR) fuer KTLX...")
    update_station_data("KTLX", "BR")

    generate_index_html()

    worker = threading.Thread(target=live_background_worker, daemon=True)
    worker.start()

    time.sleep(1)
    webbrowser.open(f"http://localhost:{PORT}")

    print(f"\n🚀 Starte den Multi-Produkt Level-3 Server auf http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
