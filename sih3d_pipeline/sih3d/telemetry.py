"""Load drone telemetry into one shape, whatever the source.

Everything is returned on the video clock (seconds from the first video frame):
  GPS       t, lat, lon, alt, h_acc, v_acc   (accuracies are 1-sigma metres, NaN when unknown)
  barometer t, alt                            (optional)
  attitude  t, yaw, pitch, roll in degrees    (optional)

Sources:
  DJI .SRT             subtitle telemetry, already on the video clock
  flight-log .csv      needs time / latitude / longitude columns
  Zurich AGZ log dir   OnboardGPS / BarometricPressure / OnboardPose share a microsecond clock with the frames
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.prepare_dataset import parse_csv, parse_srt


@dataclass
class Telemetry:
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    h_acc: np.ndarray
    v_acc: np.ndarray
    baro_t: np.ndarray | None = None
    baro_alt: np.ndarray | None = None
    att_t: np.ndarray | None = None
    yaw: np.ndarray | None = None
    pitch: np.ndarray | None = None
    roll: np.ndarray | None = None
    source: str = ""

    def summary(self):
        s = {"source": self.source, "gps_fixes": int(len(self.t)),
             "duration_s": round(float(self.t[-1] - self.t[0]), 1) if len(self.t) else 0.0,
             "median_h_acc_m": None if np.all(np.isnan(self.h_acc)) else round(float(np.nanmedian(self.h_acc)), 2),
             "barometer": self.baro_t is not None, "attitude": self.att_t is not None}
        return s


def _dedupe(t, lat, lon, alt, *extra):
    """Receivers often repeat the last fix until a new one arrives; keep real updates only."""
    keep = np.ones(len(t), bool)
    keep[1:] = (np.diff(lat) != 0) | (np.diff(lon) != 0) | (np.diff(alt) != 0)
    return (t[keep], lat[keep], lon[keep], alt[keep], *[e[keep] for e in extra])


def from_srt(path):
    rows = np.array(parse_srt(path), float)
    t, lat, lon, alt = _dedupe(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3])
    nan = np.full(len(t), np.nan)
    return Telemetry(t, lat, lon, alt, nan, nan.copy(), source=f"DJI SRT {Path(path).name}")


def from_csv(path):
    rows = np.array(parse_csv(path), float)
    t, lat, lon, alt = _dedupe(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3])
    nan = np.full(len(t), np.nan)
    return Telemetry(t, lat, lon, alt, nan, nan.copy(), source=f"flight-log CSV {Path(path).name}")


def _read_columns(path):
    with open(path, newline="", errors="ignore") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        cols = {h: [] for h in header if h}
        for row in reader:
            for h, v in zip(header, row):
                if h:
                    try:
                        cols[h].append(float(v))
                    except ValueError:
                        cols[h].append(np.nan)
    return {h: np.array(v) for h, v in cols.items()}


def _quat_to_ypr(w, x, y, z):
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return yaw, pitch, roll


def from_agz(log_dir, first_imgid, last_imgid, margin_s=10.0):
    """Zurich AGZ logs for frames first_imgid..last_imgid, re-timed so first_imgid is t = 0."""
    log_dir = Path(log_dir)
    g = _read_columns(log_dir / "OnboardGPS.csv")
    img = g["imgid"]
    t0 = g["timpstemp"][img == first_imgid][0] / 1e6
    t1 = g["timpstemp"][img == last_imgid][0] / 1e6
    win = lambda ts: (ts >= t0 - margin_s) & (ts <= t1 + margin_s)

    ts = g["timpstemp"] / 1e6
    m = win(ts) & (g["fix_type"] >= 3)
    eph = g["eph_m"][m]
    epv = np.where(g["epv_m"][m] > 0.01, g["epv_m"][m], np.nan)  # the AGZ epv column is unset (~1e-43)
    t, lat, lon, alt, eph, epv = _dedupe(ts[m] - t0, g["lat"][m], g["lon"][m], g["alt"][m], eph, epv)
    tel = Telemetry(t, lat, lon, alt, eph, epv, source=f"Zurich AGZ logs, imgid {first_imgid}-{last_imgid}")

    b = _read_columns(log_dir / "BarometricPressure.csv")
    bt = b["timpstemp"] / 1e6
    mb = win(bt)
    tel.baro_t, tel.baro_alt = bt[mb] - t0, b["altitude"][mb]

    p = _read_columns(log_dir / "OnboardPose.csv")
    pt = p["timpstemp"] / 1e6
    mp = win(pt)
    tel.att_t = pt[mp] - t0
    tel.yaw, tel.pitch, tel.roll = _quat_to_ypr(p["attitude_w"][mp], p["attitude_x"][mp],
                                               p["attitude_y"][mp], p["attitude_z"][mp])
    return tel


def agz_frame_times(log_dir, first_imgid, last_imgid):
    """Video-clock time of every AGZ frame (its own log timestamp, not an assumed 30 fps)."""
    g = _read_columns(Path(log_dir) / "OnboardGPS.csv")
    sel = (g["imgid"] >= first_imgid) & (g["imgid"] <= last_imgid)
    ts = g["timpstemp"][sel] / 1e6
    order = np.argsort(g["imgid"][sel])
    return g["imgid"][sel][order].astype(int), ts[order] - ts[order][0]


def load(path, agz_range=None):
    path = Path(path)
    if path.is_dir():
        if not agz_range:
            raise SystemExit("A telemetry folder is read as Zurich AGZ logs: pass --agz-range FIRST LAST")
        return from_agz(path, *agz_range)
    if path.suffix.lower() == ".srt":
        return from_srt(path)
    if path.suffix.lower() == ".csv":
        return from_csv(path)
    raise SystemExit(f"{path}: telemetry must be a DJI .SRT, a flight-log .csv, or an AGZ 'Log Files' folder")
