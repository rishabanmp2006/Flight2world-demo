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
| AI scene labels | `semantics.py` | SegFormer labels every keyframe: ground, low/high vegetation, building, water, road, vehicle, person, structure, sky |
| AI masks | `masks.py` | YOLO11-seg outlines moving objects; with sky/vehicle/person labels they are excluded from dense reconstruction |
| Reconstruction | `odm.py` | OpenDroneMap in Docker with per-frame GPS accuracy, frame-order matching, rolling-shutter correction, reconstruction limited to the area around the camera track (`--auto-boundary`), lens model choice (`--camera-lens`); threads sized to memory, automatic retry after out-of-memory |
| AI point classification | `semantics.py` | Projects every point into every keyframe (with occlusion test) and takes a majority vote of AI labels; stores how many cameras saw each point |
| AI gap filling | `holes.py` | Depth Anything V2, calibrated per frame against measured depth, fills surfaces the pass saw poorly; only extends measured surfaces (≤1.5 m), only from well-calibrated frames (≤10% depth error), capped at 15% extra points; filled points are flagged `inferred=1` |
| Tilt correction (experimental, `--level`) | `level.py` | Uses vertical AI-classified walls and level ground to remove tilt a straight single pass leaves; unreliable in narrow streets, so off by default |
| Accuracy | `evaluate.py`, `citygml.py` | Camera positions vs reference poses; cloud vs reference LiDAR (as georeferenced, after rigid alignment, scale); walls and roofs vs official LoD2 buildings |
| Viewer | `export.py`, `viewer/index.html` | Textured mesh or points coloured by photo, AI class, or confidence (AI-estimated points in blue); click-to-measure |

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
