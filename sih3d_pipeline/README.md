# sih3d_pipeline — single-pass drone video to a measurable 3D model

The `sih3d` pipeline for SIH26158 (NTRO): one command turns a drone video plus its telemetry into
a georeferenced, textured, AI-classified 3D model with per-point confidence, an accuracy report and a
browser viewer.

This folder holds **code only**. Datasets (about 62 GB in the original working copy) and model
weights are not in git — [DATA.md](DATA.md) explains how to fetch exactly the parts the pipeline uses.
The full pipeline description, measured results and honest limits are in
[sih3d/README.md](sih3d/README.md).

It is independent of the `core/` reconstruction package at the root of this repository: separate
code, separate dependencies, separate virtual environment.

## Layout

```
sih3d_pipeline/
├── sih3d/               the pipeline package  (python -m sih3d ...)
├── scripts/             dataset preparation, OpenDroneMap runner, accuracy scripts
├── viewer/index.html    browser viewer (points / textured mesh, click-to-measure)
├── download_data.py     fetches datasets and model weights into data/ and models/
├── requirements.txt
├── DATA.md              what data exists, where it comes from, licences, how to get it
└── README.md
```

Created locally and ignored by git (see `.gitignore`):

```
data/            datasets, prepared keyframes, OpenDroneMap projects, run reports
models/          yolo11n-seg.pt and the Hugging Face cache (models/hf)
viewer/data/     point buffers exported for the viewer
viewer/models.json
.venv/
```

## Setup

Run everything **from inside this folder**. The code resolves `data/`, `models/` and
`viewer/` relative to the working directory, and `sih3d/__main__.py` imports
`scripts.prepare_dataset`, so `sih3d/` and `scripts/` must be importable from where you run.

```bash
cd sih3d_pipeline
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# OpenDroneMap does the reconstruction, in Docker
open -a Docker            # macOS; any running Docker daemon works
docker pull opendronemap/odm

# model weights (otherwise they download on first run) and a dataset
.venv/bin/python download_data.py models aukerman osm
```

`ffmpeg` is also needed, but only by `scripts/images_to_video.py`.

## Run

```bash
# Any DJI video with its .SRT (turn on "video captions" on the drone)
.venv/bin/python -m sih3d run --video DJI_0001.MP4 --telemetry DJI_0001.SRT --focal35 24 --name flight1

# Near-real-time preview first, then full quality
.venv/bin/python -m sih3d run ... --mode preview
.venv/bin/python -m sih3d run ... --mode full

# View results: serve this folder, then open http://127.0.0.1:8765/viewer/
python3 -m http.server 8765 --bind 127.0.0.1
```

Each stage prints one summary line (telemetry fixes, keyframes selected, GPS-fusion accepted /
rejected / fallback / quality, single-pass flight verdict, Docker availability, reconstruction
result). `--segment` defaults to `all`: every detected shot contributes keyframes, so multi-shot or
photo-sequence surveys keep their full coverage — pass `--segment longest` to reconstruct only the
single longest shot, or `--segment single-pass` to keep the keyframes of the largest *logical flight*
(video cuts inside one continuous flight change nothing).
GPS fusion validates its own Kalman/RTS track against the raw fixes: when the filter rejects too
much of the flight or drifts away from the observations, the run falls back to the raw GPS track
and says so loudly (log warning + `gps_fusion` diagnostics in report.json). `keyframes.json` and
`report.json` are written before ODM starts, so a failed or Docker-less run still records exactly
how far it got.

## Single-pass flight assumption

The intended input is **one continuous forward drone pass** — the drone may curve, climb, slow down,
hover briefly or be measured by noisy GPS, but it must not orbit a building or fly several passes over
the same street. Every run measures the fused trajectory (`sih3d/flight.py`): path length, start-to-end
displacement, path efficiency, dominant axis and lateral spread, direction reversals, revisited ground,
and whether the telemetry is a continuous flight path at all. A *video shot* is not a *flight*: cuts
inside one continuous flight are grouped back into one flight from the telemetry, and a video holding
several flights is reported as several.

| verdict | meaning | default (`--flight-validation enforce`) |
|---|---|---|
| `valid` / `valid_with_warning` | one continuous forward pass (warning = short track or coarse evidence) | continue |
| `unknown` | telemetry cannot support *or* rule out a single pass (e.g. a photo-sequence slideshow) | continue, limitation stated in log + report |
| `suspicious` / `invalid` | repeated coverage, several turn-backs, closed loop or out-and-back | stop, unless overridden |

```bash
python -m sih3d run ... --flight-validation warn     # report only, never gate
python -m sih3d run ... --flight-validation off      # skip the check (report says so)
python -m sih3d run ... --allow-multi-pass           # research/testing: continue anyway
python -m sih3d run ... --segment single-pass        # keyframes of the largest logical flight only
```

Curved trajectories are explicitly allowed (a half-circle still scores 0.64 path efficiency against a
0.55 threshold) and nothing is straightened or warped to make a track look straight. Orbiting and
repeated passes are outside the intended input model. Isolated GPS spikes are removed before measuring
(never the first or last fix), a track whose spacing is still dominated by GPS noise can only ever come
out `unknown` rather than rejected, and if more than 5% of the keyframes fall outside the telemetry's
time range their positions are extrapolated, so the trajectory OpenDroneMap is given can no longer earn a
positive verdict — it becomes `unknown` with the reason (never a rejection). The verdict, the metrics, the
thresholds that produced it and every warning are written to `report.json → flight_validation`;
`report.json → quality` keeps reconstruction success, single-pass compliance, georeferencing, point-cloud
quality, AI classification coverage and gap filling apart, and leaves anything that cannot be established
as `null`/`not_run` with the reason instead of inventing a number.

Outputs go to `data/runs/<name>/report.json` and `data/odm_projects/<name>/` (textured mesh, point
clouds, orthophoto). [DATA.md](DATA.md) has the commands for each benchmark dataset.

## Storage baseline

A completed run writes `data/runs/<name>/` (keyframes, AI labels/masks, report.json),
`data/odm_projects/<name>/` (the ODM project plus the AI point clouds) and
`viewer/data/<name>/` (exported buffers). Measure a run — read-only, nothing is
deleted or modified — with:

```bash
cd sih3d_pipeline
.venv/bin/python -m sih3d disk-usage --name <run>
# defaults: --work data/runs --projects data/odm_projects --viewer viewer
```

Each row is one target's logical size, largest first. The footer separates the
**logical total** (sum of the rows — a hardlinked file appears in every row that
contains a name for it, e.g. frames/masks hard-linked into
`odm_projects/<run>/images/`) from the **unique total** (each file counted once
by inode: unique file bytes, so hardlinked copies count once and separate copies
count in full).

What a successful run can give back — dry-run only (without `--yes` nothing is
deleted):

```bash
.venv/bin/python -m sih3d cleanup --name <run>
```

ODM's map outputs (`odm_orthophoto/`, `odm_dem/`, `odm_report/`) are listed
here too and, like the other ODM intermediates, are part of the cleanup
dry-run for a successful run. For anything outside the listed targets use
`du -sh data/odm_projects/<run>/*`.

## Tested with

| Component | Version |
|---|---|
| Python | 3.14.7 |
| OpenDroneMap (Docker `opendronemap/odm`) | 3.6.2 |
| torch / torchvision | 2.14.0 / 0.29.0 |
| transformers | 5.17.0 |
| ultralytics | 8.4.150 |
| timm | 1.0.29 |
| numpy | 2.5.3 |
| opencv-python | 5.0.0.93 |
| laspy / lazrs | 2.7.0 / 0.8.2 |
| pyproj | 3.8.0 |
| scipy | 1.18.1 |
| pillow | 12.3.0 |
| piexif | 1.1.3 |

`requirements.txt` is unpinned, as in the original project. If a newer release breaks something,
pin to the versions above.

The `models` weights have their own licences — notably **YOLO11 is AGPL-3.0** — and several datasets
are non-commercial. Check [DATA.md](DATA.md) before any use beyond research.
