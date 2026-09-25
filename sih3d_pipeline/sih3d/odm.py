"""OpenDroneMap stage: project setup, memory-aware Docker run, automatic retry after running out of memory.

Project layout (data/odm_projects/<name>/):
  images/<frame>.jpg          keyframes with GPS in EXIF
  images/<frame>_mask.png     optional: black = ignore (moving objects, sky)
  geo.txt                     per-image position + horizontal/vertical accuracy from the GPS fusion
"""
import re
import shutil
import subprocess
from pathlib import Path

MODES = {
    # full quality: dense cloud at medium density, classified ground, textured 3D mesh
    "full": ["--pc-quality", "medium", "--feature-quality", "high", "--mesh-size", "300000"],
    # near-real-time preview: coarse cloud and mesh in a few minutes
    "preview": ["--pc-quality", "lowest", "--feature-quality", "medium", "--mesh-size", "60000",
                "--orthophoto-resolution", "10"],
}
STAGES = ["dataset", "split", "merge", "opensfm", "openmvs", "odm_filterpoints", "odm_meshing", "mvs_texturing",
          "odm_georeferencing", "odm_dem", "odm_orthophoto", "odm_report", "odm_postprocess"]


def _hardlink_or_copy(src, dst):
    """Hard-link src to dst so keyframe/mask data is not duplicated on disk.

    Falls back to a full metadata-preserving copy (shutil.copy2) where the filesystem
    refuses links (cross-device targets, container bind-mounts, permission quirks).
    """
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def setup_project(projects_root, name, frames_dir, records, fused, masks_dir=None):
    """records: keyframe dicts with 'name'; fused: arrays lat, lon, alt, h_std, v_std aligned with records."""
    project = Path(projects_root) / name
    images = project / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)
    for r in records:
        _hardlink_or_copy(Path(frames_dir) / r["name"], images / r["name"])
        if masks_dir:
            mask = Path(masks_dir) / (Path(r["name"]).stem + "_mask.png")
            if mask.exists():
                _hardlink_or_copy(mask, images / mask.name)
    with open(project / "geo.txt", "w") as f:
        f.write("EPSG:4326\n")
        for i, r in enumerate(records):
            f.write(f"{r['name']} {fused['lon'][i]:.9f} {fused['lat'][i]:.9f} {fused['alt'][i]:.3f} nan nan nan "
                    f"{max(fused['h_std'][i], 0.05):.3f} {max(fused['v_std'][i], 0.05):.3f}\n")
    return project


class DockerUnavailable(RuntimeError):
    """Docker cannot be used: the executable is missing from PATH or the daemon is unreachable.

    Raised instead of an unhandled FileNotFoundError / SystemExit so the run orchestration
    can record the failure in report.json (same failure structure as a failed ODM run).
    """


def docker_threads():
    try:
        out = subprocess.run(["docker", "info", "--format", "{{.MemTotal}} {{.NCPU}}"], capture_output=True, text=True)
    except FileNotFoundError as e:
        raise DockerUnavailable(
            "docker executable not found on PATH - install Docker Desktop, start it, then re-run") from e
    if out.returncode:
        detail = ((out.stderr or out.stdout or "").strip().splitlines() or ["no details"])[0][:160]
        raise DockerUnavailable(f"Docker is not running: open Docker Desktop and retry (docker info failed: {detail})")
    mem, cpus = out.stdout.split()
    gb = int(mem) / 2 ** 30
    return max(2, min(int(cpus), int(gb) - 3)), round(gb, 1)  # ODM peaks around 1 GB per thread


def _last_stage(log_text):
    started = re.findall(r"Running (\w+) stage", log_text)
    return started[-1] if started else "dataset"


def run(project, mode="full", extra=None, rolling_shutter=False, camera_lens="auto", max_retries=2, log_path=None):
    project = Path(project)
    threads, gb = docker_threads()
    log_path = Path(log_path or project / "odm_run.log")
    # --auto-boundary keeps the model to the area around the camera track. Without it, an uncalibrated wide lens left
    # a few points ~350 km away (AGZ via plain .SRT), and meshing tiled that whole extent.
    # --skip-report (all modes): ODM's PDF report (odm_report/) is never read by the pipeline –
    # Flight2World writes its own data/runs/<name>/report.json.  No --dsm: the DSM
    # (odm_dem/dsm.tif) is a QGIS export only and nothing downstream reads it either.
    base = ["--skip-report", "--pc-classify", "--auto-boundary", "--geo", "/datasets/%s/geo.txt" % project.name,
            "--matcher-order", "10", "--camera-lens", camera_lens, *MODES[mode]]
    if rolling_shutter:
        base.append("--rolling-shutter")
    rerun_from = None
    for attempt in range(max_retries + 1):
        cmd = ["docker", "run", "--rm", "-v", f"{project.parent.resolve()}:/datasets", "opendronemap/odm",
               "--project-path", "/datasets", project.name, *base, "--max-concurrency", str(threads), *(extra or [])]
        if rerun_from:
            cmd += ["--rerun-from", rerun_from]
        with open(log_path, "a") as log:
            log.write(f"\n===== sih3d: mode={mode} threads={threads} docker={gb}GB rerun_from={rerun_from} =====\n")
            log.flush()
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        text = log_path.read_text(errors="ignore")
        if code == 0:
            return {"ok": True, "threads": threads, "attempts": attempt + 1}
        if "out of memory" not in text.split("===== sih3d:")[-1] and "Child returned 137" not in text.split("===== sih3d:")[-1]:
            break
        rerun_from = _last_stage(text.split("===== sih3d:")[-1])
        threads = max(2, threads // 2)
    return {"ok": False, "threads": threads, "attempts": attempt + 1, "log": str(log_path)}


def outputs(project):
    p = Path(project)
    files = {
        "textured_mesh": p / "odm_texturing" / "odm_textured_model_geo.obj",
        "point_cloud": p / "odm_georeferencing" / "odm_georeferenced_model.laz",
        "orthophoto": p / "odm_orthophoto" / "odm_orthophoto.tif",
        "dsm": p / "odm_dem" / "dsm.tif",
        "report": p / "odm_report" / "report.pdf",
    }
    return {k: str(v) for k, v in files.items() if v.exists()}
