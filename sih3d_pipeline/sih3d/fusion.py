"""GPS + barometer fusion: a Kalman filter with a Rauch-Tung-Striebel smoother per local ENU axis.

Why: consumer GPS on a drone is noisy (AGZ: ~4 m median, 24 m worst vs survey poses) and jumps.
Positions are what anchor the reconstruction's scale and georeference, so they are cleaned first:
  * constant-velocity motion model (process noise q, m^2/s^3)
  * GPS east/north/up weighted by the receiver's own accuracy estimate
  * barometer altitude (smooth short-term) with its slowly varying offset to GPS removed
  * innovation gating rejects fixes that jump beyond `gate` sigmas
  * the backward smoothing pass uses future fixes too, since processing is offline
"""
import math

import numpy as np

from scripts.prepare_dataset import WGS84_A, WGS84_E2, geodetic_to_ecef, geodetic_to_enu

DEFAULT_H_ACC = 3.0  # metres, when the source gives no accuracy


def enu_to_geodetic(enu, origin):
    lat0, lon0, alt0 = origin
    la, lo = math.radians(lat0), math.radians(lon0)
    r = np.array([[-math.sin(lo), math.cos(lo), 0.0],
                  [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                  [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
    x, y, z = (np.atleast_2d(enu) @ r + geodetic_to_ecef(lat0, lon0, alt0)).T
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + alt)))
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(lon), p / np.cos(lat) - n


def _smooth_axis(meas_t, meas_z, meas_var, query_t, q, gate):
    """1-D constant-velocity Kalman filter + RTS smoother. Returns position, std at query_t and rejections."""
    n_meas = len(meas_t)
    times = np.concatenate([meas_t, query_t])
    order = np.argsort(times, kind="stable")
    n = len(order)
    xp, pp = np.zeros((n, 2)), np.zeros((n, 2, 2))
    xf, pf = np.zeros((n, 2)), np.zeros((n, 2, 2))
    first = np.argsort(meas_t)[: min(10, n_meas)]
    x = np.array([np.median(meas_z[first]), 0.0])
    p = np.diag([float(np.median(meas_var[first])), 25.0])
    t_prev, rejected, streak = times[order[0]], 0, 0
    for k, idx in enumerate(order):
        dt = times[idx] - t_prev
        t_prev = times[idx]
        f = np.array([[1.0, dt], [0.0, 1.0]])
        x = f @ x
        p = f @ p @ f.T + q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        xp[k], pp[k] = x, p
        if idx < n_meas:
            s = p[0, 0] + meas_var[idx]
            innov = meas_z[idx] - x[0]
            if innov ** 2 <= gate ** 2 * s or streak >= 10:  # after 10 rejections in a row, trust the sensor again
                gain = p[:, 0] / s
                x = x + gain * innov
                p = p - np.outer(gain, p[0, :])
                streak = 0
            else:
                rejected += 1
                streak += 1
        xf[k], pf[k] = x, p
    xs, ps = xf.copy(), pf.copy()
    for k in range(n - 2, -1, -1):
        dt = times[order[k + 1]] - times[order[k]]
        f = np.array([[1.0, dt], [0.0, 1.0]])
        c = pf[k] @ f.T @ np.linalg.pinv(pp[k + 1])
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
        ps[k] = pf[k] + c @ (ps[k + 1] - pp[k + 1]) @ c.T
    out_pos, out_std = np.zeros(len(query_t)), np.zeros(len(query_t))
    for k, idx in enumerate(order):
        if idx >= n_meas:
            out_pos[idx - n_meas] = xs[k, 0]
            out_std[idx - n_meas] = math.sqrt(max(ps[k, 0, 0], 0.0))
    return out_pos, out_std, rejected


def _rolling_median(t, v, half_window):
    out = np.empty(len(t))
    for i, ti in enumerate(t):
        m = np.abs(t - ti) <= half_window
        out[i] = np.median(v[m])
    return out


def fuse(tel, query_t, q_h=0.02, q_v=0.2, gate=4.0, baro_sigma=0.5):
    """Smoothed lat, lon, alt (+ 1-sigma horizontal / vertical std) at each query time.

    q_h=0.02 was chosen on Zurich AGZ against Pix4D poses: worst horizontal error 24.3 -> ~15 m, p90 15.7 -> ~14 m,
    median unchanged (street-canyon GPS error is slowly varying, which no smoother removes). Raise q_h for
    aggressive manoeuvres, lower it for slow survey flights."""
    query_t = np.asarray(query_t, float)
    origin = (float(np.median(tel.lat)), float(np.median(tel.lon)), float(np.median(tel.alt)))
    enu = geodetic_to_enu(tel.lat, tel.lon, tel.alt, origin)
    h_acc = np.where(np.isnan(tel.h_acc), DEFAULT_H_ACC, np.clip(tel.h_acc, 0.02, 50))
    v_acc = np.where(np.isnan(tel.v_acc), 1.5 * h_acc, np.clip(tel.v_acc, 0.03, 75))

    e, e_std, rej_e = _smooth_axis(tel.t, enu[:, 0], h_acc ** 2, query_t, q_h, gate)
    n, n_std, rej_n = _smooth_axis(tel.t, enu[:, 1], h_acc ** 2, query_t, q_h, gate)

    up_t, up_z, up_var = tel.t, enu[:, 2], v_acc ** 2
    baro_used = False
    if tel.baro_t is not None and len(tel.baro_t) > 10:
        gps_up_at_baro = np.interp(tel.baro_t, tel.t, enu[:, 2])
        rel = tel.baro_alt - tel.baro_alt[0]
        offset = _rolling_median(tel.baro_t, gps_up_at_baro - rel, 30.0)  # baro drifts slowly vs GPS
        up_t = np.concatenate([up_t, tel.baro_t])
        up_z = np.concatenate([up_z, rel + offset])
        up_var = np.concatenate([up_var, np.full(len(tel.baro_t), baro_sigma ** 2)])
        baro_used = True
    u, u_std, rej_u = _smooth_axis(up_t, up_z, up_var, query_t, q_v, gate)

    lat, lon, alt = enu_to_geodetic(np.column_stack([e, n, u]), origin)
    return {"lat": lat, "lon": lon, "alt": alt,
            "h_std": np.hypot(e_std, n_std), "v_std": u_std,
            "stats": {"gps_fixes": int(len(tel.t)), "rejected_east": rej_e, "rejected_north": rej_n,
                      "rejected_up": rej_u, "barometer_used": baro_used,
                      "median_h_std_m": round(float(np.median(np.hypot(e_std, n_std))), 2),
                      "median_v_std_m": round(float(np.median(u_std)), 2)}}
