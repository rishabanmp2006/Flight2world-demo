"""GPS + barometer fusion: a Kalman filter with a Rauch-Tung-Striebel smoother per local ENU axis.

Why: consumer GPS on a drone is noisy (AGZ: ~4 m median, 24 m worst vs survey poses) and jumps.
Positions are what anchor the reconstruction's scale and georeference, so they are cleaned first:
  * constant-velocity motion model (process noise q, m^2/s^3)
  * GPS east/north/up weighted by the receiver's own accuracy estimate
  * barometer altitude (smooth short-term) with its slowly varying offset to GPS removed
  * innovation gating rejects fixes that jump beyond `gate` sigmas
  * the backward smoothing pass uses future fixes too, since processing is offline

Robustness against over-confident rejection (Aukerman regression: 65/77 legitimate fixes
rejected, fused track 32-95 m from the real survey):

  * the state is initialised from the data (median position, median slope of the first
    fixes), so a track that starts mid-motion is not fought as an outlier;
  * the initial velocity variance scales with the track's own per-fix dynamics;
  * measurement variance is floored at (STEP_ROBUST_FACTOR * robust per-fix step)^2 --
    thresholds thus adapt to the data and to the receiver's reported accuracy (which is
    used as-is when present, so no hard-coded metre threshold assumes a flight quality);
  * every consecutive rejection multiplies the process noise by STREAK_INFLATION (capped),
    i.e. the model quickly loses confidence when reality keeps disagreeing and re-locks
    instead of rejecting the whole flight.

Deterministic fallback criterion (checked after the filter pass, on the horizontal axes):

  fallback activates when EITHER
    - more than FALLBACK_REJECTION_RATIO of the GPS fixes were rejected on east or north, OR
    - the median distance between the smoothed track and the raw fixes exceeds
      FALLBACK_RESIDUAL_FACTOR * h_scale, where h_scale = max(MIN_H_SCALE_M, median of the
      reported per-fix horizontal accuracy, DEFAULT_H_ACC when unreported).
  Both criteria depend only on the telemetry itself.  When fallback activates, the fused
  output IS the raw GPS track linearly interpolated to the query times (no invented
  accuracy: the reported per-fix accuracy becomes the reported std), and a warning is
  returned in the fusion stats for the CLI to print.

The returned dict keeps its historical keys (lat/lon/alt/h_std/v_std and the stats block);
the stats block is extended with diagnostics and never loses a key.
"""
import math

import numpy as np

from scripts.prepare_dataset import WGS84_A, WGS84_E2, geodetic_to_ecef, geodetic_to_enu

DEFAULT_H_ACC = 3.0  # metres, when the source gives no accuracy

# --- robust gating -----------------------------------------------------------
STREAK_REACQUIRE = 10     # after this many consecutive rejections, trust the sensor again
STREAK_INFLATION = 4.0    # per consecutive rejection the process noise is scaled by this
INFLATION_CAP = 3         # cap on the inflation exponent (4^3 = 64): re-lock quickly, but the
                          # filter must stay bounded when the model simply cannot track the fix
                          # dynamics (e.g. photo-sequence telemetry) - there the validation
                          # fallback below, not the filter, is the correct answer
STEP_ROBUST_FACTOR = 1.0  # gate sigma floored at factor * the 90th-percentile per-fix step:
                          # gate ~= 4x the track's own typical per-fix change, so legitimate
                          # dynamics like survey U-turns are not gated away as outliers
STEP_QUANTILE = 0.90      # percentile of |per-fix change| used for that scale (a median is ~0
                          # when dynamics are sparse, e.g. straight legs between rare turns)

# --- validation / fallback ---------------------------------------------------
FALLBACK_REJECTION_RATIO = 0.30  # > 30% of horizontal fixes rejected -> filter lost vs raw
FALLBACK_RESIDUAL_FACTOR = 4.0   # median fused-vs-raw residual > factor * reference accuracy
WARN_REJECTION_RATIO = 0.10      # warn below fallback, above this ratio
MIN_H_SCALE_M = 2.0              # floor for the reference accuracy (guards over-confident receivers)


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


def _robust_step(z):
    """Typical per-fix change of the raw axis (metres), used to floor the innovation gate.

    The 90th percentile of |diff| (deterministic): unlike a median it stays sensitive to
    dynamics that occupy a minority of fixes (e.g. U-turns between long straight legs).
    0 for fewer than 3 fixes.
    """
    if len(z) < 3:
        return 0.0
    return float(np.quantile(np.abs(np.diff(z)), STEP_QUANTILE))


def _smooth_axis(meas_t, meas_z, meas_var, query_t, q, gate):
    """1-D constant-velocity Kalman filter + RTS smoother.

    Returns (position at query_t, std at query_t, rejected_count,
             position at each measurement time, |innovation| per measurement).

    Rejections here are the filter's own opinion: a fix whose innovation exceeds `gate`
    sigmas of (process + measurement) uncertainty.  Robustness details are in the module
    docstring; nothing in here is trajectory- or dataset-specific.
    """
    n_meas = len(meas_t)
    times = np.concatenate([meas_t, query_t])
    order = np.argsort(times, kind="stable")
    n = len(order)
    step_rob = _robust_step(meas_z)
    # Two different roles for measurement uncertainty:
    #   gate_var -> outlier DECISION: floored by the track's own dynamics, so legitimate
    #              per-fix motion is not rejected as impossible;
    #   meas_var -> update GAIN: the receiver's reported accuracy, so a plausible fix is
    #              weighted exactly as much as the receiver claims (no invented precision).
    gate_var = np.maximum(np.asarray(meas_var, float), (STEP_ROBUST_FACTOR * step_rob) ** 2)
    order_t = np.argsort(meas_t)
    first = order_t[: min(10, n_meas)]
    x = np.array([float(np.median(meas_z[first])), 0.0])
    if len(first) >= 2:
        dts = np.diff(meas_t[first])
        ok = dts > 0
        if ok.any():  # initialise the velocity from the data, not from rest
            x[1] = float(np.median(np.diff(meas_z[first])[ok] / dts[ok]))
    dt_med = float(np.median(np.diff(meas_t[order_t]))) if n_meas >= 2 else 1.0
    v_var = max(25.0, (step_rob / max(dt_med, 1e-3)) ** 2)
    xp, pp = np.zeros((n, 2)), np.zeros((n, 2, 2))
    xf, pf = np.zeros((n, 2)), np.zeros((n, 2, 2))
    p = np.diag([float(np.median(gate_var[first])), v_var])
    t_prev, rejected, streak = times[order[0]], 0, 0
    innovations = np.zeros(n_meas)
    for k, idx in enumerate(order):
        dt = times[idx] - t_prev
        t_prev = times[idx]
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q_eff = q * STREAK_INFLATION ** min(streak, INFLATION_CAP)  # model loses confidence while wrong
        x = f @ x
        p = f @ p @ f.T + q_eff * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        xp[k], pp[k] = x, p
        if idx < n_meas:
            s_gate = p[0, 0] + gate_var[idx]
            innov = meas_z[idx] - x[0]
            innovations[idx] = abs(float(innov))
            if innov ** 2 <= gate ** 2 * s_gate or streak >= STREAK_REACQUIRE:
                s = p[0, 0] + meas_var[idx]  # gain from the reported accuracy, not the gate floor
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
    meas_pos = np.zeros(n_meas)
    for k, idx in enumerate(order):
        if idx >= n_meas:
            out_pos[idx - n_meas] = xs[k, 0]
            out_std[idx - n_meas] = math.sqrt(max(ps[k, 0, 0], 0.0))
        else:
            meas_pos[idx] = xs[k, 0]
    return out_pos, out_std, rejected, meas_pos, innovations


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
    aggressive manoeuvres, lower it for slow survey flights.

    The filter's output is validated against the raw fixes (deterministic fallback
    criterion in the module docstring): when the filter has rejected too much of the
    flight or drifted away from the observations, the raw/interpolated GPS track is
    returned instead and the diagnostics in stats say so.  stats keeps every historical
    key and adds accepted/rejected counts, rejection_ratio, innovation magnitudes,
    fallback_used / fallback_reason, quality and warnings.
    """
    query_t = np.asarray(query_t, float)
    origin = (float(np.median(tel.lat)), float(np.median(tel.lon)), float(np.median(tel.alt)))
    enu = geodetic_to_enu(tel.lat, tel.lon, tel.alt, origin)
    h_acc = np.where(np.isnan(tel.h_acc), DEFAULT_H_ACC, np.clip(tel.h_acc, 0.02, 50))
    v_acc = np.where(np.isnan(tel.v_acc), 1.5 * h_acc, np.clip(tel.v_acc, 0.03, 75))

    e, e_std, rej_e, e_meas, innov_e = _smooth_axis(tel.t, enu[:, 0], h_acc ** 2, query_t, q_h, gate)
    n, n_std, rej_n, n_meas_pos, innov_n = _smooth_axis(tel.t, enu[:, 1], h_acc ** 2, query_t, q_h, gate)

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
    u, u_std, rej_u, _, _ = _smooth_axis(up_t, up_z, up_var, query_t, q_v, gate)

    # --- validate the filtered track against the raw observations -------------
    n_fixes = int(len(tel.t))
    resid = np.hypot(e_meas - enu[:, 0], n_meas_pos - enu[:, 1])
    resid_med = float(np.median(resid)) if n_fixes else 0.0
    h_scale = float(max(MIN_H_SCALE_M, float(np.median(h_acc)))) if n_fixes else MIN_H_SCALE_M
    rej_ratio = (max(rej_e, rej_n) / n_fixes) if n_fixes else 0.0

    fallback_reason = None
    if n_fixes and rej_ratio > FALLBACK_REJECTION_RATIO:
        fallback_reason = (f"filter rejected {rej_e}/{n_fixes} east and {rej_n}/{n_fixes} north fixes "
                           f"(ratio {rej_ratio:.0%} > {FALLBACK_REJECTION_RATIO:.0%})")
    elif resid_med > FALLBACK_RESIDUAL_FACTOR * h_scale:
        fallback_reason = (f"fused track sits median {resid_med:.1f} m from the raw fixes "
                           f"(> {FALLBACK_RESIDUAL_FACTOR:.0f}x the {h_scale:.1f} m reference accuracy)")

    if fallback_reason:
        # The filter is less trustworthy than the observations: fall back to the raw GPS
        # track.  Each query time gets the fix valid at that timestamp (the nearest fix;
        # DJI subtitles carry per-fix validity intervals).  For continuous video this
        # differs from linear interpolation by at most half a fix interval of motion,
        # while for photo-sequence/stop-motion inputs it returns the true photo fix
        # instead of a position invented between two survey lines.  No invented
        # precision: the reported std is the receiver's own accuracy for that fix.
        idx = np.abs(np.asarray(tel.t)[None, :] - query_t[:, None]).argmin(axis=1) if len(query_t) else np.zeros(0, int)
        lat = np.asarray(tel.lat)[idx]
        lon = np.asarray(tel.lon)[idx]
        alt = np.asarray(tel.alt)[idx]
        h_std = h_acc[idx]
        v_std = v_acc[idx]
        quality = "fallback"
    else:
        lat, lon, alt = enu_to_geodetic(np.column_stack([e, n, u]), origin)
        h_std = np.hypot(e_std, n_std)
        v_std = u_std
        quality = "degraded" if (rej_ratio > WARN_REJECTION_RATIO or resid_med > 2.0 * h_scale) else "ok"

    warnings = []
    if fallback_reason:
        warnings.append(f"GPS fusion fallback active: {fallback_reason}. "
                        f"Using the raw GPS track as the source of truth.")
    else:
        if rej_ratio > WARN_REJECTION_RATIO:
            warnings.append(f"GPS fusion rejected {rej_ratio:.0%} of horizontal fixes "
                            f"({max(rej_e, rej_n)}/{n_fixes}); check the fused track before delivering.")
        if resid_med > 2.0 * h_scale:
            warnings.append(f"fused track is {resid_med:.1f} m (median) from the raw GPS fixes.")

    innov_h = np.hypot(innov_e, innov_n) if n_fixes else np.zeros(0)
    stats = {
        # historical keys (unchanged meaning: per-axis rejection counts of the filter pass)
        "gps_fixes": n_fixes,
        "rejected_east": int(rej_e),
        "rejected_north": int(rej_n),
        "rejected_up": int(rej_u),
        "barometer_used": baro_used,
        "median_h_std_m": round(float(np.median(h_std)), 2) if len(h_std) else 0.0,
        "median_v_std_m": round(float(np.median(v_std)), 2) if len(v_std) else 0.0,
        # diagnostics
        "accepted_east": n_fixes - int(rej_e),
        "accepted_north": n_fixes - int(rej_n),
        "accepted_up": int(len(up_t)) - int(rej_u),
        "accepted": min(n_fixes - int(rej_e), n_fixes - int(rej_n)),  # fixes accepted on both horizontal axes
        "rejected": max(int(rej_e), int(rej_n)),                      # conservative upper bound
        "rejection_ratio": round(float(rej_ratio), 3),
        "max_innovation_m": round(float(np.max(innov_h)), 2) if n_fixes else 0.0,
        "median_innovation_m": round(float(np.median(innov_h)), 2) if n_fixes else 0.0,
        "median_residual_m": round(resid_med, 2),
        "fallback_used": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "quality": quality,
        "warnings": warnings,
    }
    return {"lat": lat, "lon": lon, "alt": alt, "h_std": h_std, "v_std": v_std, "stats": stats}
