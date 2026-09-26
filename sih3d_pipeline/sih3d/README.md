# sih3d — single-pass drone video to a measurable 3D model

One command turns a drone video plus its telemetry into a georeferenced, textured, AI-classified 3D model with a
confidence score per point, an accuracy report and a browser viewer. Built for Smart India Hackathon 2026,
problem SIH26158 (NTRO).

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # AI model weights download on first run
open -a Docker && docker pull opendronemap/odm                        # OpenDroneMap runs in Docker

# Any DJI video with its .SRT (turn on "video captions" on the drone)
.venv/bin/python -m sih3d run --video DJI_0001.MP4 --telemetry DJI_0001.SRT --focal35 24 --name flight1

# Fast near-real-time preview first (about 1–2 minutes for ~50 keyframes), then full quality
.venv/bin/python -m sih3d run ... --mode preview
.venv/bin/python -m sih3d run ... --mode full

# Zurich AGZ test flight with lens calibration and reference data
.venv/bin/python -m sih3d run --video data/simulated/agz_zurich_seg59001.mp4 \
    --telemetry "data/agz_zurich_seg59001/Log Files" --agz-range 59001 62000 \
    --calib data/agz_zurich_seg59001/calibration_data.npz --rolling-shutter --name agz \
    --lidar data/ground_truth/zurich/2683_1248.las data/ground_truth/zurich/2683_1249.las \
    --citygml data/ground_truth/zurich/swissbuildings3d_3_0_2019_1091-23_2056_5728.citygml.zip

# View: serve the project folder, then open http://127.0.0.1:8765/viewer/
python3 -m http.server 8765 --bind 127.0.0.1
```

Outputs: `data/runs/<name>/report.json` (every stage's numbers and timings) and `data/odm_projects/<name>/`
(textured mesh, point clouds, orthophoto).

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Telemetry | `telemetry.py` | Reads DJI `.SRT`, flight-log CSV, or AGZ logs (GPS, barometer, autopilot attitude) onto the video clock |
| Keyframes | `frames.py` | Crops black bars, finds hard cuts, keeps the sharpest frame per ~10% image shift, drops blurry frames |
| Lens and image clean-up | `frames.py` | Removes lens distortion from a calibration, evens exposure across frames, light filtering against compression blocking |
| GPS fusion | `fusion.py` | Kalman filter + smoother on GPS and barometer; rejects jumps; per-frame position accuracy |
| Single-pass flight validation | `flight.py` | Measures the fused trajectory (keyframe track *and* telemetry track): path length, displacement, path efficiency, dominant axis and lateral spread, direction reversals, revisits, telemetry continuity; groups video shots into logical flights; reports `flight_validation` and applies the policy below |
| AI scene labels | `semantics.py` | SegFormer labels every keyframe: ground, low/high vegetation, building, water, road, vehicle, person, structure, sky |
| AI masks | `masks.py` | YOLO11-seg outlines moving objects; with sky/vehicle/person labels they are excluded from dense reconstruction |
| Reconstruction | `odm.py` | OpenDroneMap in Docker with per-frame GPS accuracy, frame-order matching, rolling-shutter correction, reconstruction limited to the area around the camera track (`--auto-boundary`), lens model choice (`--camera-lens`); threads sized to memory, automatic retry after out-of-memory |
| AI point classification | `semantics.py` | Projects every point into every keyframe (with occlusion test) and takes a majority vote of AI labels; stores how many cameras saw each point |
| AI gap filling | `holes.py` | Depth Anything V2, calibrated per frame against measured depth, fills surfaces the pass saw poorly; only extends measured surfaces (≤1.5 m), only from well-calibrated frames (≤10% depth error), capped at 15% extra points; filled points are flagged `inferred=1` |
| Tilt correction (experimental, `--level`) | `level.py` | Uses vertical AI-classified walls and level ground to remove tilt a straight single pass leaves; unreliable in narrow streets, so off by default |
| Accuracy | `evaluate.py`, `citygml.py` | Camera positions vs reference poses; cloud vs reference LiDAR (as georeferenced, after rigid alignment, scale); walls and roofs vs official LoD2 buildings |
| Viewer | `export.py`, `viewer/index.html` | Textured mesh or points coloured by photo, AI class, or confidence (AI-estimated points in blue); click-to-measure |

## Single-pass flight assumption

Flight2World reconstructs **one continuous forward drone pass**: the drone enters the scene once, flies
through it while the camera keeps looking at what is ahead of it, and leaves. The model is anchored to that
one GPS track, so the pipeline measures the track and states whether the input really is one pass.

**What single-pass means here.** One continuous trajectory that makes net forward progress — the drone ends
away from where it started and most of the travelled path is forward rather than back over ground already
covered. **What it does not mean.** A straight line, or a video without cuts. Curves are valid (a
half-circle still scores 0.64 path efficiency; the gate is 0.55), and turning, climbing, slowing down,
hovering briefly or noisy GPS are all normal. Nothing rotates, straightens or warps the model to make a track
look straight — `level.py` still refuses to level a flight that is not one straight pass.

**What is checked** (`sih3d/flight.py`, deterministic, on two trajectories: the fused keyframe positions that
OpenDroneMap is given, and the denser telemetry track):

| metric | meaning | role in the verdict |
|---|---|---|
| `path_length_m`, `displacement_m`, `path_efficiency` | travelled path vs start-to-end distance | `< 0.15` invalid, `< 0.55` suspicious |
| `dominant_axis`, `dominant_heading_deg`, `lateral_deviation_m`, `straightness` | PCA travel direction; how far the track strays from it | reported only — curvature never rejects |
| `forward_fraction`, `direction_reversals` | share of along-axis travel going backwards; significant turn-backs | `> 20%` / `> 2` suspicious, `> 55%` invalid |
| `revisit_score`, `loop_closures` | how much of the track comes back within 4–20 m of a place it left at least twice that radius earlier, and how many such excursions | `> 0.25` suspicious, `≥ 0.60` with little net progress invalid |
| `discontinuous_step_fraction`, `max_speed_mps` | share of ≥ 1 s windows implying more than 30 m/s (108 km/h) | `> 15%` → the telemetry is not a flight path: `unknown` |
| `position_sigma_m`, `gps_outliers_removed`, `thinning_stride` | estimated GPS jitter, spikes removed, and the noise-aware centreline the metrics are measured on | noise is measured and removed; it can never produce a negative verdict |

Every threshold is either a fraction of the flight's own scales or a fixed physical limit (the speed of a
consumer drone), and the values that produced a verdict are written into
`report.json → flight_validation.thresholds`. Isolated GPS spikes are removed before measuring (never the
first/last fix, or the measured start and end of the flight would be wrong), and a track whose spacing is
still dominated by GPS jitter can only ever come out `unknown`, never `suspicious`/`invalid`.

A track is judged only where its positions are real. If more than 5% of the keyframes fall outside the
telemetry's time range their positions are extrapolated, so the keyframe trajectory — the one OpenDroneMap
is given — is reported as `unknown` and a positive verdict is downgraded to `unknown` (never a rejection)
with the reason, even when the telemetry trajectory itself looks like a clean single pass.

**Verdicts and what the run does** (`--flight-validation enforce` is the default):

| status | meaning | default action |
|---|---|---|
| `valid` | one continuous forward pass | continue |
| `valid_with_warning` | single pass, but short track or coarse evidence | continue + warning |
| `unknown` | the telemetry can neither support nor rule out a single pass (photo-sequence slideshow, clock mismatch, too few fixes, all-noise track) | continue + the limitation is stated loudly — never a single-pass claim |
| `suspicious` | evidence of repeated coverage or several turn-backs | stop unless `--allow-multi-pass` |
| `invalid` | closed loop, out-and-back, or near-zero net progress | stop unless `--allow-multi-pass` |

```bash
# advisory only, or no trajectory check at all
python -m sih3d run ... --flight-validation warn
python -m sih3d run ... --flight-validation off      # the report says the check was skipped

# research/testing: reconstruct a circular or multi-pass flight anyway (the verdict is still recorded)
python -m sih3d run ... --allow-multi-pass

# keep only the keyframes of the largest logical flight (cuts inside one continuous flight change nothing)
python -m sih3d run ... --segment single-pass
```

A *video shot* (a run of frames between cuts) is edit information; a *logical flight* is the physical
trajectory. The keyframe stage therefore groups shots into logical flights from the telemetry, keeps
`all`/`longest` behaviour available, and never equates the two: a flight whose footage has hard cuts stays one
flight, and a video holding several flights is reported as several (`keyframes.flights`).

`report.json` gains `flight_validation` (status, `single_pass`, path length, displacement, efficiency,
dominant axis, reversals, revisit score, trajectory start/end, telemetry coverage, warnings and the
thresholds used) and `quality`, which keeps six claims apart — reconstruction success, single-pass
compliance, georeferencing, point-cloud quality, AI classification coverage, gap-filling amount. Where a
metric cannot be established (no independent reference, stage skipped) it is `null`/`not_run` with the
reason; no accuracy number is invented.

## Problem statement coverage

| SIH26158 requirement | Where it is handled |
|---|---|
| Single-pass video (1080p/4K) + GPS + flight metadata | `telemetry.py`, `frames.py` |
| Optional IMU / barometer / camera intrinsics / RTK | barometer and attitude read; intrinsics via `--calib`; RTK accuracy flows through fusion |
| Georeferenced, metrically accurate | GPS-anchored reconstruction; `evaluate.py` measures scale and offset |
| Terrain, structures, facades, rooftops, roads, vegetation, obstacles | AI classes on the point cloud + textured mesh |
| Textured meshes or point clouds | OpenDroneMap mesh + classified, filled LAZ |
| Limited viewing angles, occluded surfaces | AI gap filling, marked as estimated |
| Motion blur and compression artefacts | sharpness-based keyframes, deblocking |
| Variable illumination and shadows | exposure normalisation |
| Dynamic objects | AI masks |
| GPS inaccuracies and sensor noise | fusion with outlier rejection; reconstruction refines positions |
| Real-time or near-real-time | `--mode preview` |
| Metric accuracy without extensive GCPs | no ground control points used anywhere; accuracy measured against independent references |

## Results on real data (Zurich AGZ, 100 s of 30 fps 1080p street flight, consumer GPS)

Independent references: swisstopo swissSURFACE3D LiDAR and swissBUILDINGS3D 3.0 official buildings; Pix4D poses.
No ground control points were used.

**One command, everything on** (`--mode preview`, run `agz_final_preview`):

| Stage | Result |
|---|---|
| Total time | 3.1 min (1.8 min pipeline + 1.3 min accuracy checks) |
| Keyframes | 53 from 3000 frames, 0 cuts, lens undistorted, exposure normalised |
| AI masks | 72 cars, 4 people excluded from dense reconstruction |
| AI classes | 100% of 290,442 points; building 85%, high vegetation 14%, road 1%; median 10 cameras per point |
| AI gap filling | 43,566 estimated points (13%, capped); depth fit error 2.0% |
| Roll correction | −9.8°; exactly vertical wall patches 16,104 → 59,081 |
| Walls vs official buildings (after alignment) | median 0.79 m, 37% within 0.5 m |
| Georeferencing offset (GPS only) | 4.8 m horizontal, 0.6 m vertical |
| Camera positions vs reference, worst case | raw GPS 24.7 m → reconstruction 12.5 m |

**Full quality** (`--mode full`): 6.7 min, 3.8 million points, road 7.6% of points.

**Roll about the flight line** (still needed to match official walls):

| Run | Before | Correction | After |
|---|---|---|---|
| No AI, preview | 19.3° | −12.1° | 7.9° |
| AI, preview | 8.0° | −12.5° | −3.6° |
| AI, full quality | ≥19.1° (lower bound) | −36.1° | 4.0° |
| Final one-command run, preview | 16.3° | −9.8° | 7.2° |

The correction reduces the roll in every run (average about 16° → 5.7° over the four measured runs) but under-corrects
some runs by up to ~7°; heading and along-street slope stay within ~1.5°.

**Plain DJI-style `.SRT`, no calibration** (`--camera-lens fisheye`): 1.8 min, one model with 52/53 cameras.

## Honest limits

- "Real-time" on a MacBook without an NVIDIA GPU means a preview in minutes, not live processing during flight.
- Consumer GPS in a street canyon has slowly varying multi-metre errors; smoothing removes jumps, not that bias.
  Absolute position is only as good as the GPS unless RTK or reference data is available; shape and scale are much better.
- AI-filled surfaces are estimates. They are always flagged and shown in a separate colour. On Zurich, filled wall
  points matched official walls about as well as measured ones (1.05 m vs 1.08 m median); filled roof points were worse
  (1.74 m vs 1.34 m).
- Wide lenses need a calibration (`--calib`) or `--camera-lens fisheye`. On the Zurich GoPro video without either,
  OpenDroneMap's lens estimate was far off (k1 −0.08 vs −0.28 calibrated), the solve split into 3 pieces and stray
  points reached ~350 km; with `--camera-lens fisheye` and `--auto-boundary` it stayed one model (52/53 cameras).
- A single straight pass leaves the model's roll about the flight line unconstrained by GPS: three identical Zurich
  runs came out tilted about 8°, 19° and ~36° against official walls. See `level.py` for the correction and its status.
- The single-pass check judges *shape*, not accuracy: a pass can be perfectly single-pass and still be
  georeferenced only as well as its GPS. It also cannot judge a trajectory whose timestamps are not the flight
  clock (a photo sequence played as a slideshow, e.g. the Aukerman fixture) — that input is reported as
  `unknown` with the reason, and the run continues exactly as before, but no single-pass or metric-accuracy
  claim is made for it. Genuinely re-visited ground (orbits, repeated lanes, out-and-back flights) is caught
  and needs `--allow-multi-pass`.
