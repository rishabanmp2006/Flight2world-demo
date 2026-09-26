"""Single-pass flight validation: does this input honour the input model the pipeline assumes?

Flight2World reconstructs **one continuous forward drone pass**: the drone enters the scene once, moves
through it while the camera keeps seeing the ground/building in front of it, and leaves.  The pipeline does
not require the drone to orbit a building, to fly several independent passes over the same street, or to
revisit the same place from independent angles — a normal curve, a slight turn, slowing down, hovering for a
moment or noisy GPS are all perfectly fine.

Nothing here changes how the reconstruction is done; this module only *measures* the fused (or raw) GPS
trajectory, *classifies* it against documented thresholds, and produces the machine-readable
`flight_validation` block of `report.json`.  It is pure logic: numpy arrays in, dictionaries out, no I/O, no
randomness, no dataset-specific constants.

What is measured (all on horizontal metres; vertical motion is never a criterion — a climb is normal):

  path_length_m          length of the trajectory after spike removal and thinning (see Robustness)
  displacement_m         distance between the first and the last position
  path_efficiency        displacement / path_length: 1.0 = straight, ~0.6 = a half-circle, ~0 = a closed loop
  dominant_axis          principal (PCA) travel direction of the track, as [east, north] + a compass heading
  lateral_deviation_m    p90 distance of the track from the dominant-axis line
  straightness           PCA explained variance of the dominant axis (informational: a curve is *not* invalid)
  forward_fraction       share of along-axis travel that goes in the dominant direction (1.0 = never backwards)
  direction_reversals    number of significant reversals of the along-axis motion (hysteresis, see thresholds)
  revisit_score          share of the track that comes back within `revisit_radius` of an earlier position
                         after travelling at least `revisit_min_away_m` of path in between
  loop_closures          how many such return excursions the track makes
  discontinuous_step_fraction  share of >= 1 s windows whose implied speed is not physically possible
  max_speed_mps          largest implied speed over a >= 1 s window (what the fraction above uses)
  max_fix_speed_mps      largest implied speed between two adjacent fixes (informational only: consumer GPS
                         jitter between dense fixes can be very large on perfectly good data)

Robustness (why a noisy single pass is not rejected):

  * GPS jitter inflates the *measured* path length of a flight, because the sum of many small noisy steps is
    longer than the true path.  Metrics are therefore computed on a **thinned centreline**: the track is
    smoothed and then decimated so that the kept points lie about `THIN_SIGMA_FACTOR * sigma` apart, where
    `sigma` is a robust estimate of the per-fix GPS noise (MAD of the second differences).  At that spacing
    the jitter adds only a few percent to the path length, while genuine curvature is preserved.
  * Isolated spikes are removed first: a point far from the median of the two fixes on each side of it (a
    track end is never dropped, or the measured start/end of the flight would be wrong).
  * Every threshold is a *fraction* of the flight's own scales (path length, along-axis span), or a fixed
    physical limit (speed), never a value tuned to one dataset.
  * A trajectory that is not a continuous flight path at all (a photo sequence played as a slideshow, a
    telemetry clock that does not match the video clock, GPS jump-outs) is reported as `unknown`: the
    limitation is stated in the report and the existing pipeline behaviour continues.
  * A track may only earn a positive verdict for the positions it actually describes.  When more than
    `KEYFRAME_COVERAGE_MISSING` of the keyframes lie outside the telemetry's time range, their positions are
    extrapolated and the keyframe track (the one OpenDroneMap is given) can be neither validated nor rejected,
    so a positive verdict is downgraded to `unknown` with the reason.

Status vocabulary (also used by the CLI policy):

  valid               one continuous forward pass; no warning
  valid_with_warning  single pass, but short track / weak evidence — warning only
  suspicious          evidence against a single pass (reversals, repeated coverage); needs --allow-multi-pass
  invalid             clear orbit / repeated passes / near-zero net progress; needs --allow-multi-pass
  unknown             telemetry cannot support (or rule out) a single pass: the report says so explicitly
"""

import math

import numpy as np

from scripts.prepare_dataset import geodetic_to_enu

MODE_SINGLE_PASS = "single_pass"

# --- statuses ----------------------------------------------------------------------------------------------
VALID = "valid"
VALID_WITH_WARNING = "valid_with_warning"
SUSPICIOUS = "suspicious"
INVALID = "invalid"
UNKNOWN = "unknown"
STATUSES = (VALID, VALID_WITH_WARNING, SUSPICIOUS, INVALID, UNKNOWN)

# Severity used when combining the keyframe trajectory with the denser telemetry track.  `unknown` carries no
# severity: a track that cannot be judged must not veto a verdict from a track that can (it only adds a
# warning), and if no track can be judged the result is `unknown`.
SEVERITY = {VALID: 0, VALID_WITH_WARNING: 1, SUSPICIOUS: 3, INVALID: 4}

# --- thresholds (every one is documented here, none is dataset-specific) ------------------------------------
MAX_PLAUSIBLE_SPEED_MPS = 30.0     # 108 km/h: above the top speed of the consumer drones this pipeline targets
                                   # (DJI Mavic-class ~21 m/s), so a track that keeps moving faster than this
                                   # is not a flight path
MIN_SPEED_WINDOW_S = 1.0           # speed is judged over >= 1 s, never between two adjacent fixes: at 5-10 Hz
                                   # consumer GPS jitter alone implies tens of m/s between neighbouring fixes
SPEED_NOISE_ALLOWANCE_M = 25.0     # generous per-window position-noise slack (~8 sigma of consumer GPS), so
                                   # noisy but genuine flight is never mistaken for a telemetry mismatch
DISCONTINUOUS_STEP_FRACTION = 0.15  # > this share of impossible windows => the telemetry is not a continuous
                                    # flight track (photo-sequence slideshow, clock mismatch): report unknown
MIN_TRACK_POINTS = 8               # fewer positions than this: no trajectory claim can be made
MIN_PATH_LENGTH_M = 20.0           # shorter than this: too short to judge forward progression
SHORT_TRACK_M = 50.0               # shorter than this (but judgeable): verdict + warning
THIN_SIGMA_FACTOR = 6.0            # measure on a centreline whose kept points are ~6 sigma apart, so GPS
                                   # jitter adds only a few percent to the path length
MIN_THINNED_POINTS = 10            # never reduce the track below this many points (keeps curves/reversals)
NOISE_LIMITED_STEP_SIGMA = 3.0     # if the thinned spacing is still < 3 sigma, a *negative* verdict is not
                                   # supportable and is downgraded to unknown (noise must never reject)
OUTLIER_MIN_M = 15.0               # a point further than this from its neighbourhood median is a spike
REVERSAL_MIN_FRACTION = 0.10       # a reversal counts when the along-axis motion turns by >= 10% of the span
REVERSAL_MIN_M = 5.0
REVISIT_RADIUS_FRACTION = 0.02     # "came back here" radius: 2% of the path length, clamped to 4..20 m
REVISIT_RADIUS_MIN_M = 4.0
REVISIT_RADIUS_MAX_M = 20.0
REVISIT_MIN_SEQ_FRACTION = 0.05    # ... and at least 5% of the track (and twice the radius) must lie in
                                   # between, measured along the path (a hover covers no ground: no revisit)
REVISIT_MAX_POINTS = 600           # cap for the O(n^2) revisit check (deterministic even spacing)
EFFICIENCY_INVALID = 0.15          # less net progress than this = closed loop / perfect back-and-forth
EFFICIENCY_SUSPICIOUS = 0.55       # below this the net progress no longer dominates the travelled path
                                    # (a half-circle is 0.64, so normal curved passes are comfortably above)
BACKWARD_INVALID = 0.55            # more than half the along-axis travel going backwards = repeated passes
BACKWARD_SUSPICIOUS = 0.20
REVISIT_INVALID = 0.60             # most of the track re-covering ground already flown over
REVISIT_SUSPICIOUS = 0.25
REVERSALS_SUSPICIOUS = 2           # more than this many significant reversals = several turns back
KEYFRAME_COVERAGE_MISSING = 0.05   # more keyframes than this outside the telemetry time range => their
                                   # positions are extrapolated and the keyframe track cannot be judged

POLICIES = ("enforce", "warn", "off")

#: Friendly names for the two trajectories the run validates (used in messages and warnings).
_TRACK_LABELS = {"keyframes_fused": "keyframe trajectory", "telemetry_fused": "telemetry trajectory"}


def _label(source):
    return _TRACK_LABELS.get(source, source or "trajectory")


# ---------------------------------------------------------------------------- geometry / robust track prep

def enu_track(lat, lon, alt, origin=None):
    """Local East-North-Up metres for a lat/lon/alt track.  Non-finite rows are dropped.

    Returns (enu Nx3, keep mask, origin).  The origin defaults to the median position so that the local frame
    is centred on the flight (metres, no projection distortion at drone scales).
    """
    lat, lon, alt = (np.asarray(a, float).ravel() for a in (lat, lon, alt))
    keep = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
    if not keep.all():
        lat, lon, alt = lat[keep], lon[keep], alt[keep]
    if not len(lat):
        return np.zeros((0, 3)), keep, (0.0, 0.0, 0.0)
    if origin is None:
        origin = (float(np.median(lat)), float(np.median(lon)), float(np.median(alt)))
    return geodetic_to_enu(lat, lon, alt, origin), keep, origin


def noise_sigma(points):
    """Robust estimate of the per-fix position jitter (metres) from the track itself.

    The second difference of a straight noisy track has standard deviation sigma * sqrt(6); its median
    absolute deviation is used per axis, so a few spikes or a loose curve do not dominate.  Deterministic.
    """
    p = np.asarray(points, float)[:, :2]
    if len(p) < 5:
        return 0.0
    second = p[2:] - 2.0 * p[1:-1] + p[:-2]
    mad = np.median(np.abs(second - np.median(second, axis=0)), axis=0)
    sigma = float(np.mean(mad) / (0.6745 * math.sqrt(6.0)))
    return max(sigma, 0.0)


def despike(points, min_gate=OUTLIER_MIN_M):
    """Drop isolated spikes: a point further than `min_gate` from its neighbours' median.

    Each point is compared with the median of the two fixes before and two after it, *excluding itself*: on a
    straight or gently curving track that median is the point's own expected position, while a real spike
    jumps away from it.  The first and last two points are never dropped — a track end has no two-sided
    support and the real start/end of the flight must survive, or path length and displacement lie.

    Single pass, deterministic.  The gate is a fixed 15 m or six times the track's own robust noise estimate,
    whichever is larger, so ordinary GPS jitter and normal curved flight are never touched.  Returns
    (kept points, removed mask) so the caller can keep the clock aligned with the kept rows.
    """
    p = np.asarray(points, float)
    n = len(p)
    if n < 5:
        return p, np.zeros(n, bool)
    mask = np.zeros(n, bool)
    for i in range(2, n - 2):
        neighbours = np.vstack([p[i - 2:i], p[i + 1:i + 3]])
        mask[i] = np.linalg.norm(p[i] - np.median(neighbours, axis=0)) > min_gate
    return p[~mask], mask


def smooth(points, window):
    """Centred moving average over `window` samples (shrinking at the ends).  Deterministic.

    Used together with the decimation stride: averaging `window` samples divides white GPS jitter by about
    sqrt(window), which is what keeps a dense, noisy track measurable without touching genuine motion.
    """
    p = np.asarray(points, float)
    n = len(p)
    window = int(window)
    if window <= 1 or n < 3:
        return p
    half = window // 2
    c = np.vstack([np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)])
    out = np.empty_like(p)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def thin_stride(points, times, sigma):
    """Index stride that spaces kept points ~THIN_SIGMA_FACTOR * sigma apart (at least MIN_THINNED_POINTS).

    The average speed is estimated as displacement/duration — a *lower* bound on the real speed (curves,
    pauses and revisits all make it smaller), so the stride is never shorter than the noise needs.  When the
    estimate collapses (a closed loop, an out-and-back) the stride is capped by MIN_THINNED_POINTS; such
    tracks are then measured coarsely, which is all a verdict of "not a forward pass" needs.
    """
    p = np.asarray(points, float)
    n = len(p)
    if n < 3 or sigma <= 0:
        return 1
    spacing = THIN_SIGMA_FACTOR * sigma
    if times is not None and len(times) == n:
        t = np.asarray(times, float)
        duration = float(t[-1] - t[0])
        dt_med = float(np.median(np.diff(t))) if n > 1 else 0.0
        if duration <= 0 or dt_med <= 0:
            return 1
        speed = float(np.linalg.norm(p[-1, :2] - p[0, :2])) / duration
        per_index = max(speed * dt_med, 1e-9)
    else:
        per_index = float(np.median(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)))
        if per_index <= 0:
            return 1
    stride = int(round(spacing / per_index))
    return max(1, min(stride, max(1, n // MIN_THINNED_POINTS)))


def count_reversals(along, epsilon):
    """Significant reversals of the along-axis motion, using hysteresis of `epsilon` metres.

    Small back-and-forth jitter below `epsilon` never counts; only a turn-back larger than `epsilon`
    (10% of the along-axis span, at least 5 m) counts as a reversal.  A point that only wobbles by less than
    `epsilon` therefore cannot create a spurious reversal, and normal noise on a straight pass scores 0.
    """
    a = np.asarray(along, float)
    if len(a) < 2:
        return 0
    direction = 0
    extreme = float(a[0])
    reversals = 0
    for value in a[1:]:
        value = float(value)
        if direction == 0:
            if abs(value - extreme) >= epsilon:
                direction = 1 if value > extreme else -1
                extreme = value
            continue
        if direction > 0:
            if value > extreme:
                extreme = value
            elif extreme - value >= epsilon:
                reversals += 1
                direction, extreme = -1, value
        else:
            if value < extreme:
                extreme = value
            elif value - extreme >= epsilon:
                reversals += 1
                direction, extreme = 1, value
    return reversals


def revisit_metrics(points, radius, away_m):
    """How much of the track re-covers earlier ground, and how many such excursions there are.

    A point counts as a revisit when it comes within `radius` of a position the drone reached at least
    `away_m` of *travelled path* earlier — i.e. it went somewhere else and came back.  Measuring the
    separation along the path (not in samples) is what keeps a hover, a pause or a dense-but-clean track
    from looking like a revisit: sitting still covers no ground, so nothing counts as a return.

    A single forward pass scores ~0; an out-and-back scores ~0.5 (one return excursion); a lawnmower survey
    scores high in every later lane.  `loop_closures` counts the excursions (runs of consecutive revisiting
    points) — the "how many times did it come back" number a report reader wants.  The check is O(n^2) on at
    most `REVISIT_MAX_POINTS` evenly spaced points, so it stays fast and deterministic.
    """
    p = np.asarray(points, float)[:, :2]
    n = len(p)
    if n > REVISIT_MAX_POINTS:  # even spacing keeps the result stable and platform independent
        p = p[np.unique(np.linspace(0, n - 1, REVISIT_MAX_POINTS).round().astype(int))]
        n = len(p)
    if n < 3 or radius <= 0:
        return {"revisit_score": 0.0, "loop_closures": 0}
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    flags = np.zeros(n, bool)
    for i in range(1, n):
        j = int(np.searchsorted(arc, arc[i] - away_m, side="right"))
        if j < 1:
            continue
        flags[i] = bool((np.linalg.norm(p[i] - p[:j], axis=1) <= radius).any())
    runs = int(np.sum(flags[1:] & ~flags[:-1])) + int(flags[0]) if flags.any() else 0
    return {"revisit_score": float(flags.mean()), "loop_closures": runs}


def _steps_metrics(points, times):
    """Speed sanity of the track: how much of it is not physically a flight path.

    Two numbers are reported: the largest speed between adjacent fixes (`max_fix_speed_mps`, informational —
    consumer GPS jitter between dense fixes can be large for perfectly good data) and the largest speed over
    a >= MIN_SPEED_WINDOW_S window (`max_speed_mps`), which is what decides `discontinuous_step_fraction`:
    a window is impossible when it implies more than MAX_PLAUSIBLE_SPEED_MPS plus SPEED_NOISE_ALLOWANCE_M.
    """
    p = np.asarray(points, float)
    base = {"steps_judged": 0, "discontinuous_step_fraction": 0.0, "max_speed_mps": None,
            "max_fix_speed_mps": None, "median_step_m": 0.0}
    if len(p) < 2:
        return base
    step = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    base["median_step_m"] = round(float(np.median(step)), 2)
    if times is None or len(times) != len(p):
        return base
    t = np.asarray(times, float)
    dt = np.diff(t)
    judged = dt > 0
    if judged.any():
        base["max_fix_speed_mps"] = round(float((step[judged] / dt[judged]).max()), 2)
    window_end = np.searchsorted(t, t + MIN_SPEED_WINDOW_S, side="left")
    i = np.arange(len(p))
    use = (window_end < len(p)) & (window_end > i)
    if not use.any():
        return base
    i, j = i[use], window_end[use]
    span = t[j] - t[i]
    window_step = np.linalg.norm(p[j, :2] - p[i, :2], axis=1)
    speed = window_step / span
    impossible = window_step > MAX_PLAUSIBLE_SPEED_MPS * span + SPEED_NOISE_ALLOWANCE_M
    base["steps_judged"] = int(use.sum())
    base["discontinuous_step_fraction"] = round(float(impossible.mean()), 3)
    base["max_speed_mps"] = round(float(speed.max()), 2)
    return base


# ---------------------------------------------------------------------------- metrics + classification

def analyse_track(lat, lon, alt, times=None, source="", origin=None):
    """All single-pass metrics + status for one trajectory (lat/lon/alt, optional times on one clock).

    Pure and deterministic.  Returns a flat dict; `status` is one of the module statuses and `single_pass` is
    True / False / None (None = unknown: the telemetry cannot support the claim in either direction).
    """
    enu, keep, origin = enu_track(lat, lon, alt, origin)
    t = None
    if times is not None:
        t = np.asarray(times, float).ravel()
        if len(t) == len(keep):          # one timestamp per input position: follow the dropped rows
            t = t[keep]
        elif len(t) != len(enu):         # unusable clock: measure geometry only
            t = None
    if t is not None and len(t) > 1 and np.any(np.diff(t) < 0):  # deterministic: judge in time order
        order = np.argsort(t, kind="stable")
        enu, t = enu[order], t[order]
    out = {"source": source, "points": int(len(enu)), "thinned_points": None, "thinning_stride": None,
           "duration_s": None,
           "path_length_m": None, "displacement_m": None, "path_efficiency": None, "dominant_axis": None,
           "dominant_heading_deg": None, "lateral_deviation_m": None, "lateral_deviation_relative": None,
           "straightness": None, "direction_reversals": None, "forward_fraction": None, "revisit_score": None,
           "revisit_radius_m": None, "revisit_min_away_m": None, "loop_closures": None, "position_sigma_m": None, "gps_outliers_removed": 0,
           "steps_judged": 0, "discontinuous_step_fraction": 0.0, "max_speed_mps": None,
           "max_fix_speed_mps": None, "median_step_m": None, "median_track_step_m": None,
           "status": UNKNOWN,
           "single_pass": None, "reasons": [], "warnings": []}
    if t is not None and len(t) and len(enu):
        out["duration_s"] = round(float(t[-1] - t[0]), 2)

    if len(enu) < MIN_TRACK_POINTS:
        out["reasons"].append(f"only {len(enu)} positions (at least {MIN_TRACK_POINTS} are needed to judge "
                              f"a trajectory)")
        out["warnings"].append(f"single-pass check not possible for the {_label(source)}: only {len(enu)} "
                               f"GPS positions")
        return out

    sigma = noise_sigma(enu)
    clean, removed = despike(enu, min_gate=max(OUTLIER_MIN_M, 6.0 * sigma))
    out["gps_outliers_removed"] = int(removed.sum())
    clean_t = None if t is None else t[~removed]
    out.update(_steps_metrics(clean, clean_t))   # judged after spike removal: a spike is not a wrong clock
    if len(clean) < MIN_TRACK_POINTS:
        out["reasons"].append("almost every position looks like an outlier")
        out["warnings"].append(f"single-pass check not possible for the {_label(source)}: nearly all positions "
                               f"look like outliers, so the track is not a continuous path")
        return out
    t = clean_t
    out["position_sigma_m"] = round(sigma, 2)
    stride = thin_stride(clean, t, sigma)
    out["thinning_stride"] = int(stride)
    track = smooth(clean, stride)[::stride]
    if len(track) < 2 or (len(clean) - 1) % stride:  # always keep the real last position as an endpoint
        track = np.vstack([track, clean[-1]])
    horiz = track[:, :2]
    step = np.linalg.norm(np.diff(horiz, axis=0), axis=1)
    out["median_track_step_m"] = round(float(np.median(step)), 2) if len(step) else 0.0
    path = float(step.sum())
    displacement = float(np.linalg.norm(horiz[-1] - horiz[0]))
    out["path_length_m"] = round(path, 2)
    out["displacement_m"] = round(displacement, 2)
    out["path_efficiency"] = round(displacement / path, 3) if path > 0 else 0.0
    out["thinned_points"] = int(len(track))

    if path < MIN_PATH_LENGTH_M:
        out["reasons"].append(f"trajectory is only {path:.1f} m long (needs at least "
                              f"{MIN_PATH_LENGTH_M:.0f} m)")
        out["warnings"].append(f"single-pass check not meaningful for the {_label(source)}: the flight covers "
                               f"only {path:.1f} m")
        return out

    centred = horiz - horiz.mean(axis=0)
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    if float(axis @ (horiz[-1] - horiz[0])) < 0:  # deterministic sign: point along the travel direction
        axis = -axis
    if abs(float(axis @ (horiz[-1] - horiz[0]))) < 1e-9:  # closed loop: tie-break, still deterministic
        if axis[0] < 0 or (axis[0] == 0 and axis[1] < 0):
            axis = -axis
    explained = float(sv[0] ** 2 / max((sv ** 2).sum(), 1e-12))
    out["dominant_axis"] = [round(float(axis[0]), 4), round(float(axis[1]), 4)]
    out["dominant_heading_deg"] = round(float(np.degrees(np.arctan2(axis[0], axis[1])) % 360.0), 1)
    out["straightness"] = round(explained, 3)

    lateral = np.abs((horiz - horiz[0]) @ np.array([-axis[1], axis[0]]))
    out["lateral_deviation_m"] = round(float(np.percentile(lateral, 90)), 2)
    out["lateral_deviation_relative"] = round(float(np.percentile(lateral, 90) / path), 4)

    along = (horiz - horiz[0]) @ axis
    delta = np.diff(along)
    forward = float(np.clip(delta, 0, None).sum())
    backward = float(np.clip(-delta, 0, None).sum())
    out["forward_fraction"] = round(forward / max(forward + backward, 1e-9), 3)
    epoch = float(along.max() - along.min())
    out["direction_reversals"] = count_reversals(along, max(REVERSAL_MIN_M, REVERSAL_MIN_FRACTION * epoch))

    radius = float(np.clip(REVISIT_RADIUS_FRACTION * path, REVISIT_RADIUS_MIN_M, REVISIT_RADIUS_MAX_M))
    # "Came back here" must mean the drone travelled away first: at least twice the radius (and at least 5% of
    # the flight) of *path* must lie between the two visits.
    away_m = max(2.0 * radius, REVISIT_MIN_SEQ_FRACTION * path)
    out["revisit_radius_m"] = round(radius, 2)
    out["revisit_min_away_m"] = round(away_m, 2)
    revisits = revisit_metrics(track, radius, away_m)
    out["revisit_score"] = round(revisits["revisit_score"], 3)
    out["loop_closures"] = revisits["loop_closures"]

    status, reasons, warnings = classify(out)
    if status in (SUSPICIOUS, INVALID) and sigma > 0 and out["median_track_step_m"] < \
            NOISE_LIMITED_STEP_SIGMA * sigma:
        # Noise may never produce a negative verdict: if the measured spacing is still dominated by GPS
        # jitter, the flight is reported as unjudgeable instead of suspicious/invalid.
        status = UNKNOWN
        reasons = [f"GPS jitter ({sigma:.1f} m) is too large against the trajectory spacing "
                   f"({out['median_track_step_m']:.1f} m) to support a negative verdict"]
        warnings = [f"single-pass check inconclusive for the {_label(source)}: {reasons[0]}"]
    out["status"], out["single_pass"] = status, {VALID: True, VALID_WITH_WARNING: True,
                                                 SUSPICIOUS: False, INVALID: False, UNKNOWN: None}[status]
    out["reasons"] += reasons
    out["warnings"] += warnings
    return out


def classify(m):
    """Deterministic threshold rules -> (status, reasons, warnings).

    Order matters: a track that is not a continuous flight path is reported as unknown *before* any verdict,
    then clear orbit/repeat evidence is invalid, weaker evidence is suspicious, and short-but-clean tracks
    pass with a warning.  Everything that fires is named in `reasons`, so a report always explains itself.
    """
    if m["discontinuous_step_fraction"] > DISCONTINUOUS_STEP_FRACTION:
        return UNKNOWN, [
            f"{m['discontinuous_step_fraction']:.0%} of the >= {MIN_SPEED_WINDOW_S:.0f} s windows imply more "
            f"than {MAX_PLAUSIBLE_SPEED_MPS:.0f} m/s (max {m['max_speed_mps']} m/s): the telemetry is not one "
            f"continuous flight path"], [
            f"single-pass check inconclusive for the {_label(m['source'])}: the positions are not a continuous "
            f"flight path (photo-sequence slideshow, telemetry/video clock mismatch, or GPS jump-outs). "
            f"Georeferencing relies on these positions, so no single-pass or metric-accuracy claim follows."]
    hard, soft = [], []
    if m["path_efficiency"] < EFFICIENCY_INVALID:
        hard.append(f"the flight ends almost where it started (path efficiency {m['path_efficiency']:.2f} < "
                    f"{EFFICIENCY_INVALID}): a closed loop or a full out-and-back, not a forward pass")
    if m["revisit_score"] >= REVISIT_INVALID and m["path_efficiency"] < 2 * EFFICIENCY_INVALID:
        hard.append(f"{m['revisit_score']:.0%} of the trajectory re-covers ground already flown over within "
                    f"{m['revisit_radius_m']} m while making little net progress")
    if (1 - m["forward_fraction"]) >= BACKWARD_INVALID:
        hard.append(f"{1 - m['forward_fraction']:.0%} of the along-axis travel goes backwards "
                    f"(>= {BACKWARD_INVALID:.0%}): repeated passes over the same line")
    if hard:
        return INVALID, hard, ["single-pass flight validation failed: " + "; ".join(hard)]
    if m["path_efficiency"] < EFFICIENCY_SUSPICIOUS:
        soft.append(f"weak net progress: path efficiency {m['path_efficiency']:.2f} < {EFFICIENCY_SUSPICIOUS} "
                    f"(a curved single pass stays above this; a half-circle is 0.64)")
    if (1 - m["forward_fraction"]) > BACKWARD_SUSPICIOUS:
        soft.append(f"{1 - m['forward_fraction']:.0%} of the along-axis travel goes backwards "
                    f"(> {BACKWARD_SUSPICIOUS:.0%})")
    if m["revisit_score"] > REVISIT_SUSPICIOUS:
        soft.append(f"{m['revisit_score']:.0%} of the trajectory comes back within {m['revisit_radius_m']} m of "
                    f"an earlier position (> {REVISIT_SUSPICIOUS:.0%})")
    if m["direction_reversals"] > REVERSALS_SUSPICIOUS:
        soft.append(f"{m['direction_reversals']} significant direction reversals "
                    f"(> {REVERSALS_SUSPICIOUS}): several turn-backs")
    if soft:
        return SUSPICIOUS, soft, ["single-pass flight validation: " + "; ".join(soft) +
                                  ". The flight may re-cover ground; pass --allow-multi-pass to reconstruct "
                                  "anyway (the known working behaviour) or use a real single-pass video."]
    warnings = []
    if m["path_length_m"] < SHORT_TRACK_M:
        warnings.append(f"trajectory is short ({m['path_length_m']:.1f} m): single-pass metrics are judged, "
                        f"but they carry wide uncertainty")
    if m["gps_outliers_removed"] >= max(3, 0.1 * m["points"]):
        warnings.append(f"{m['gps_outliers_removed']} GPS spikes were removed before measuring the trajectory")
    if m["points"] < 20:
        warnings.append(f"only {m['points']} positions: trajectory metrics are coarse")
    if warnings:
        return VALID_WITH_WARNING, [], warnings
    return VALID, [], []


# ---------------------------------------------------------------------------- combining + reporting

def logical_flights(shots, max_gap_s=30.0, slack_m=25.0, max_speed_mps=MAX_PLAUSIBLE_SPEED_MPS):
    """Group *video shots* into *logical flights* — the two are not the same thing.

    A shot is a run of frames between video cuts (the edit of the footage).  A flight is one physical
    trajectory of the drone.  A single continuous flight whose footage contains hard cuts is still ONE
    flight, and one video containing several flights (lands, takes off again, or a long pause) is several.

    Two consecutive shots belong to the same flight when they are *continuous*: the time gap is small
    (<= max_gap_s) and, when positions are known, the spatial shift is explainable by flight at a plausible
    speed (<= max_speed_mps * gap, plus `slack_m` of GPS slack).  Without positions only the time gap is
    used, so a continuous flight with detected cuts always stays one logical flight.

    `shots` is a list of dicts: {"start_s", "end_s", "frames", "start_enu", "end_enu"} — the ENU entries are
    optional Nx2/Nx3 arrays or [east, north] lists (None when unknown).  Returns a list of
    {"shots": [indices], "start_s", "end_s", "duration_s", "frames", "distance_m"}.
    """
    flights = []
    for i, shot in enumerate(shots):
        gap_s = None if not flights else shot["start_s"] - flights[-1]["end_s"]
        link_m = None
        prev = shots[flights[-1]["shots"][-1]] if flights else None
        if prev is not None and prev.get("end_enu") is not None and shot.get("start_enu") is not None:
            link_m = float(np.linalg.norm(np.asarray(shot["start_enu"], float)[:2] -
                                          np.asarray(prev["end_enu"], float)[:2]))
        join = False
        if flights:
            join = gap_s is not None and gap_s <= max_gap_s
            if join and link_m is not None:
                allowed = slack_m + max_speed_mps * max(gap_s, 0.0)
                join = link_m <= allowed
        if join:
            f = flights[-1]
            f["shots"].append(i)
            f["end_s"] = shot["end_s"]
            f["frames"] += shot["frames"]
            f["distance_m"] += (link_m or 0.0) + shot.get("distance_m", 0.0)
            if shot.get("end_enu") is not None:
                f["end_enu"] = shot["end_enu"]
        else:
            flights.append({"shots": [i], "start_s": shot["start_s"], "end_s": shot["end_s"],
                            "frames": shot["frames"], "distance_m": shot.get("distance_m", 0.0),
                            "start_enu": shot.get("start_enu"), "end_enu": shot.get("end_enu")})
    for f in flights:
        f["duration_s"] = round(f["end_s"] - f["start_s"], 2)
        f["distance_m"] = round(f["distance_m"], 2)
    return flights


def largest_flight(flights):
    """Index of the logical flight with the most frames (ties -> the earliest), or None when there are none."""
    if not flights:
        return None
    return max(range(len(flights)), key=lambda k: (flights[k]["frames"], -k))


def dedupe(items):
    """Order-preserving de-duplication (two tracks often report the same limitation)."""
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _track_summary(track):
    """The subset of a track's metrics that identifies it in the report (keeps report.json readable)."""
    return {k: track[k] for k in ("source", "points", "thinned_points", "thinning_stride",
                                  "median_track_step_m", "path_length_m", "displacement_m",
                                  "path_efficiency", "dominant_axis", "dominant_heading_deg",
                                  "direction_reversals", "forward_fraction", "revisit_score", "loop_closures",
                                  "revisit_radius_m", "revisit_min_away_m", "position_sigma_m", "discontinuous_step_fraction",
                                  "max_speed_mps", "status", "single_pass", "reasons", "warnings")}


def combine_tracks(tracks):
    """Combine per-track verdicts: the most severe *judgeable* verdict wins.

    Tracks that could not be judged (`unknown`) do not veto the others — their limitation is reported as a
    warning by the caller — but if no track could be judged, the result is `unknown`.
    """
    judged = [t for t in tracks if t["status"] in SEVERITY]
    warnings = [w for tr in tracks if tr["status"] == UNKNOWN for w in tr["warnings"]]
    if not judged:
        reasons = dedupe([r for tr in tracks for r in tr["reasons"]]) or ["no judgeable trajectory"]
        return UNKNOWN, None, reasons, dedupe(warnings)
    best = max(judged, key=lambda tr: SEVERITY[tr["status"]])  # ties -> the keyframe track first
    reasons = [r for tr in judged if SEVERITY[tr["status"]] == SEVERITY[best["status"]] for r in tr["reasons"]]
    warnings = [w for tr in judged if SEVERITY[tr["status"]] == SEVERITY[best["status"]] for w in tr["warnings"]] \
        + warnings
    return best["status"], best["single_pass"], dedupe(reasons), dedupe(warnings)


def validate_run(records, telemetry=None, times=None, *, fusion=None, keyframe_report=None,
                 policy="enforce", allow_multi_pass=False, mode=MODE_SINGLE_PASS, origin=None):
    """Build the `flight_validation` report block from the selected keyframes plus the telemetry.

    `records` are the fused keyframe records (lat/lon/alt written back by the GPS fusion stage); `telemetry`
    is a `sih3d.telemetry.Telemetry` (or anything with .t/.lat/.lon/.alt).  Both trajectories are measured:
    the keyframe track (what OpenDroneMap is given) and, when it exists, the denser telemetry track, because
    a handful of keyframes cannot always describe a flight by itself.

    Returns a plain dict, safe to json.dump: status, single_pass, metrics from the deciding track, per-track
    summaries, the policy action and every warning that applies.
    """
    records = list(records or [])
    block = {"mode": mode, "policy": policy, "status": UNKNOWN, "single_pass": None, "action": "continue",
             "overridden": False, "trajectory_source": None, "telemetry_fixes": 0, "keyframes": len(records),
             "keyframes_with_position": 0, "keyframes_outside_telemetry": None, "telemetry_coverage": None,
             "trajectory_start": None, "trajectory_end": None, "warnings": [], "reasons": [], "tracks": {},
             "thresholds": _thresholds()}
    if policy == "off":
        block["status"] = "skipped"
        block["warnings"] = ["single-pass flight validation disabled (--flight-validation off): no "
                             "single-pass compliance claim is made for this run"]
        return block
    block["policy"] = policy

    if records:
        lat = np.array([r.get("lat", np.nan) for r in records], float)
        lon = np.array([r.get("lon", np.nan) for r in records], float)
        alt = np.array([r.get("alt", np.nan) for r in records], float)
    else:
        lat = lon = alt = np.zeros(0)
    t_key = np.asarray(times, float).ravel() if times is not None else None
    block["keyframes_with_position"] = int(np.isfinite(lat).sum()) if len(lat) else 0
    if block["keyframes_with_position"]:
        keep = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
        block["trajectory_start"] = _position(lat[keep][0], lon[keep][0], alt[keep][0])
        block["trajectory_end"] = _position(lat[keep][-1], lon[keep][-1], alt[keep][-1])

    if keyframe_report:
        # Video shots are the edit, logical flights are the trajectory (see logical_flights()): if the selected
        # keyframes come from more than one flight, the input is not one continuous pass.
        block["video_shots"] = keyframe_report.get("shots")
        block["logical_flights"] = keyframe_report.get("logical_flights")
        if (keyframe_report.get("logical_flights") or 0) > 1:
            block["warnings"].append(
                f"the selected keyframes come from {keyframe_report['logical_flights']} logical flights "
                f"(video shots: {keyframe_report.get('shots')}): the input is not one continuous flight")

    telemetry_track = telemetry is not None and len(getattr(telemetry, "t", []))
    if telemetry_track and t_key is not None and len(t_key):
        inside = int(np.sum((t_key >= telemetry.t[0] - 0.5) & (t_key <= telemetry.t[-1] + 0.5)))
        block["keyframes_outside_telemetry"] = int(len(t_key) - inside)
        block["telemetry_coverage"] = round(inside / len(t_key), 3)

    kf = analyse_track(lat, lon, alt, t_key, source="keyframes_fused", origin=origin)
    if block["keyframes"] and (block["keyframes_outside_telemetry"] or 0) > \
            KEYFRAME_COVERAGE_MISSING * block["keyframes"]:
        # Those positions are extrapolated past the end of the telemetry, so they describe the video clock,
        # not the flight: the keyframe track may not claim anything either way.
        kf["status"], kf["single_pass"] = UNKNOWN, None
        kf["reasons"] = [f"{block['keyframes_outside_telemetry']} of {block['keyframes']} keyframes fall "
                         f"outside the telemetry time range, so their positions are extrapolated"] + kf["reasons"]
        kf["warnings"] = [f"single-pass check not possible for the keyframe trajectory: "
                          f"{kf['reasons'][0]}"] + kf["warnings"]
    block["tracks"]["keyframes_fused"] = _track_summary(kf)
    tracks = [kf]

    if telemetry_track:
        block["telemetry_fixes"] = int(len(telemetry.t))
        tel = analyse_track(telemetry.lat, telemetry.lon, telemetry.alt, telemetry.t,
                            source="telemetry_fused", origin=origin)
        block["tracks"]["telemetry_fused"] = _track_summary(tel)
        tracks.append(tel)
    else:
        block["warnings"].append("no telemetry track available: single-pass compliance and georeferencing "
                                 "cannot be established from GPS")

    status, single_pass, reasons, warnings = combine_tracks(tracks)
    if status in (VALID, VALID_WITH_WARNING) and kf["status"] == UNKNOWN:
        # The trajectory OpenDroneMap is actually given could not be judged (extrapolated positions, a
        # slideshow clock, too few fixes, ...): the flight may well be a single pass, but this run may not
        # claim it, so a positive verdict is downgraded to "unknown" (which never gates the run).
        status, single_pass = UNKNOWN, None
        reasons = [f"the reconstructed trajectory ({block['keyframes']} keyframe positions) could not be "
                   f"judged: {kf['reasons'][0] if kf['reasons'] else 'no verdict'}"] + reasons
    block["status"], block["single_pass"] = status, single_pass
    block["reasons"], block["warnings"] = reasons, dedupe(block["warnings"] + warnings)
    deciding = max(tracks, key=lambda tr: SEVERITY.get(tr["status"], -1))
    unjudged = [tr for tr in tracks if tr["status"] == UNKNOWN]
    if status in SEVERITY and unjudged:
        others = "; ".join(f"the {_label(tr['source'])} could not be judged ({tr['reasons'][0] if tr['reasons'] else 'no verdict'})"
                           for tr in unjudged)
        block["warnings"].append(f"the single-pass verdict comes from the {_label(deciding['source'])} "
                                 f"({deciding['points']} positions): {others}")
    if deciding["status"] == status:
        block["trajectory_source"] = deciding["source"]
        for key in ("path_length_m", "displacement_m", "path_efficiency", "dominant_axis",
                    "dominant_heading_deg", "lateral_deviation_m", "lateral_deviation_relative", "straightness",
                    "direction_reversals", "forward_fraction", "revisit_score", "loop_closures",
                    "revisit_radius_m", "revisit_min_away_m", "discontinuous_step_fraction",
                    "max_speed_mps", "max_fix_speed_mps",
                    "gps_outliers_removed", "position_sigma_m", "duration_s", "points", "thinned_points",
                    "thinning_stride", "median_track_step_m"):
            block[key] = deciding.get(key)
    if block["keyframes"] and block["keyframes_with_position"] < block["keyframes"]:
        missing = block["keyframes"] - block["keyframes_with_position"]
        block["warnings"].append(f"{missing} of {block['keyframes']} keyframes have no GPS position: those "
                                 f"frames are georeferenced by OpenDroneMap alone")
    if block["keyframes_outside_telemetry"]:
        block["warnings"].append(f"{block['keyframes_outside_telemetry']} of {block['keyframes']} keyframes fall "
                                 f"outside the telemetry time range ({block['telemetry_coverage']:.0%} coverage): "
                                 f"their positions are extrapolated, so single-pass compliance and metric "
                                 f"accuracy are limited for them")
    if fusion and fusion.get("fallback_used"):
        block["fusion_fallback"] = True
        block["warnings"].append("GPS fusion fell back to the raw GPS track: positions (and therefore the "
                                 "single-pass metrics above) inherit the raw receiver noise")
    block["action"], block["overridden"], block["decision_reason"] = decision(block, policy, allow_multi_pass)
    return block


def _position(lat, lon, alt):
    return {"lat": round(float(lat), 7), "lon": round(float(lon), 7), "alt": round(float(alt), 2)}


def _thresholds():
    """The thresholds that produced the verdict, recorded in the report so a verdict can be audited."""
    return {"max_plausible_speed_mps": MAX_PLAUSIBLE_SPEED_MPS,
            "discontinuous_step_fraction": DISCONTINUOUS_STEP_FRACTION, "min_track_points": MIN_TRACK_POINTS,
            "min_path_length_m": MIN_PATH_LENGTH_M, "short_track_m": SHORT_TRACK_M,
            "efficiency_invalid": EFFICIENCY_INVALID, "efficiency_suspicious": EFFICIENCY_SUSPICIOUS,
            "backward_invalid": BACKWARD_INVALID, "backward_suspicious": BACKWARD_SUSPICIOUS,
            "revisit_invalid": REVISIT_INVALID, "revisit_suspicious": REVISIT_SUSPICIOUS,
            "reversals_suspicious": REVERSALS_SUSPICIOUS, "min_speed_window_s": MIN_SPEED_WINDOW_S,
            "speed_noise_allowance_m": SPEED_NOISE_ALLOWANCE_M,
            "revisit_radius_m": [REVISIT_RADIUS_MIN_M, REVISIT_RADIUS_MAX_M],
            "reversal_min_m": [REVERSAL_MIN_M, REVERSAL_MIN_FRACTION],
            "keyframe_coverage_missing": KEYFRAME_COVERAGE_MISSING}


def decision(block, policy="enforce", allow_multi_pass=False):
    """What the run should do about a verdict -> (action, overridden, reason).

    enforce  valid / valid_with_warning continue; suspicious / invalid stop unless --allow-multi-pass
    warn     always continue (the verdict is still reported and logged)
    off      not reached: validate_run() fills the block without judging
    """
    status = block.get("status")
    if policy == "warn":
        return "continue", False, (f"flight validation is advisory (--flight-validation warn): status is "
                                   f"'{status}'")
    if status in (SUSPICIOUS, INVALID):
        if allow_multi_pass:
            return "continue", True, (f"continuing although the flight is '{status}': --allow-multi-pass "
                                      f"(research/testing override)")
        return "stop", False, (
            f"single-pass flight validation: this flight looks '{status}' - " + "; ".join(block.get("reasons", []))
            + ". Flight2World assumes one continuous forward pass; re-fly the input as a single pass, or pass "
              "--allow-multi-pass to reconstruct it anyway (known-working behaviour for older datasets), or "
              "--flight-validation warn to keep the report but not the gate.")
    return "continue", False, f"single-pass flight validation: {status}"


def report_lines(block):
    """Human-readable summary lines for the CLI log (deterministic order)."""
    if block.get("status") == "skipped":
        return ["Flight validation: skipped (--flight-validation off)"]
    lines = [f"Flight validation: {block['status'].upper()} (single_pass="
             f"{block['single_pass']})  path {block.get('path_length_m')} m  "
             f"displacement {block.get('displacement_m')} m  efficiency {block.get('path_efficiency')}  "
             f"reversals {block.get('direction_reversals')}  revisit {block.get('revisit_score')}  "
             f"source {block.get('trajectory_source')}"]
    for reason in block.get("reasons", []):
        lines.append(f"  - {reason}")
    for warning in block.get("warnings", []):
        lines.append(f"WARNING: {warning}")
    if block.get("overridden"):
        lines.append(f"  ! {block.get('decision_reason')}")
    return lines
