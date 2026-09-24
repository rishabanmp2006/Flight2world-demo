"""One command: single-pass drone video -> georeferenced, classified, textured 3D model with an accuracy report.

  python -m sih3d run --video DJI_0001.MP4 --telemetry DJI_0001.SRT --name flight1
  python -m sih3d run --video data/simulated/agz_zurich_seg59001.mp4 \
      --telemetry "data/agz_zurich_seg59001/Log Files" --agz-range 59001 62000 \
      --calib data/agz_zurich_seg59001/calibration_data.npz --name agz --mode preview

Stages: telemetry -> keyframes (cuts, blur, undistortion, exposure, deblocking) -> GPS+barometer fusion
-> AI scene labels (SegFormer) -> AI masks for moving objects and sky (YOLO-seg + labels) -> OpenDroneMap
-> AI-classified point cloud with per-point confidence -> report.json (+ accuracy vs reference when available).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from scripts.prepare_dataset import log, write_gps_exif
from sih3d import frames, fusion, odm, telemetry


class Timer:
    def __init__(self):
        self.times = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = time.time()
                log(f"== {name}")

            def __exit__(self, *exc):
                timer.times[name] = round(time.time() - self.t, 1)
        return _Ctx()


def ground_truth_checks(project, a, cloud_path=None):
    """Optional accuracy against reference LiDAR (--lidar) and official CityGML buildings (--citygml).
    Uses the final cloud when given; AI-estimated points are always excluded from the accuracy numbers."""
    from sih3d import citygml, evaluate
    out = {}
    cloud = Path(cloud_path or Path(project) / "odm_georeferencing" / "odm_georeferenced_model.laz")
    crs = evaluate.project_crs(project)
    if a.lidar:
        lid = evaluate.cloud_vs_lidar(project, a.lidar, lidar_crs=a.reference_crs, lidar_z_offset=a.reference_z_offset,
                                      cloud_path=cloud)
        out["accuracy_vs_lidar"] = lid
        if a.citygml and "rigid_alignment" in lid:
            al = (lid["rigid_alignment"]["rotation"], lid["rigid_alignment"]["translation"])
            out["accuracy_vs_buildings"] = {"as_georeferenced": citygml.check(cloud, crs, a.citygml),
                                            "after_alignment_to_lidar": citygml.check(cloud, crs, a.citygml, alignment=al)}
    elif a.citygml:
        out["accuracy_vs_buildings"] = {"as_georeferenced": citygml.check(cloud, crs, a.citygml)}
    return out


def cmd_run(a):
    work = Path(a.work) / a.name
    frames_dir = work / "frames"
    timer = Timer()
    report = {"name": a.name, "mode": a.mode, "inputs": {"video": str(a.video), "telemetry": str(a.telemetry),
                                                        "calibration": str(a.calib or "")}}

    with timer("telemetry"):
        tel = telemetry.load(a.telemetry, a.agz_range)
        report["telemetry"] = tel.summary()

    with timer("keyframes"):
        calib = frames.load_calibration(a.calib) if a.calib else None
        records, report["keyframes"] = frames.extract_keyframes(
            a.video, frames_dir, calib=calib, sample_fps=a.sample_fps, min_shift=a.min_shift, max_gap=a.max_gap,
            segment=a.segment, normalise=not a.no_normalise, deblock=not a.no_deblock, max_size=a.max_size)
    if len(records) < 5:
        raise SystemExit(f"Only {len(records)} keyframes: the video is too short or static for reconstruction")

    with timer("gps_fusion"):
        times = np.array([r["t"] for r in records])
        if a.agz_range:
            _, ft = telemetry.agz_frame_times(a.telemetry, *a.agz_range)
            times = ft[np.array([r["frame"] for r in records])]
        fused = fusion.fuse(tel, times)
        raw = {k: np.interp(times, tel.t, getattr(tel, k)) for k in ("lat", "lon", "alt")}
        focal35 = a.focal35 or report["keyframes"]["focal35"]
        for i, r in enumerate(records):
            r.update(t=float(times[i]), raw_lat=float(raw["lat"][i]), raw_lon=float(raw["lon"][i]),
                     raw_alt=float(raw["alt"][i]), lat=float(fused["lat"][i]), lon=float(fused["lon"][i]),
                     alt=float(fused["alt"][i]), h_std=float(fused["h_std"][i]), v_std=float(fused["v_std"][i]))
            write_gps_exif(frames_dir / r["name"], r["lat"], r["lon"], r["alt"], focal35)
        report["gps_fusion"] = fused["stats"]
    names = [r["name"] for r in records]

    labels_dir = masks_dir = None
    if not a.no_ai:
        from sih3d import masks, semantics
        with timer("ai_scene_labels"):
            labels_dir = work / "labels"
            report["ai_scene_labels"] = semantics.label_frames(frames_dir, names, labels_dir)
        with timer("ai_masks"):
            masks_dir = work / "masks"
            report["ai_masks"] = masks.make_masks(frames_dir, names, masks_dir, labels_dir=labels_dir)

    with timer(f"reconstruction_{a.mode}"):
        project = odm.setup_project(a.projects, a.name, frames_dir, records, fused, masks_dir)
        report["reconstruction"] = odm.run(project, mode=a.mode, rolling_shutter=a.rolling_shutter,
                                           camera_lens=a.camera_lens)
    (work / "keyframes.json").write_text(json.dumps(records, indent=1))
    if not report["reconstruction"]["ok"]:
        report["timings_s"] = timer.times
        (work / "report.json").write_text(json.dumps(report, indent=2, default=str))
        raise SystemExit(f"Reconstruction failed; see {report['reconstruction'].get('log')}")
    report["outputs"] = odm.outputs(project)

    final_cloud = project / "odm_georeferencing" / "odm_georeferenced_model.laz"
    if labels_dir:
        from sih3d import semantics
        with timer("ai_point_classification"):
            report["classified_cloud"] = semantics.classify_cloud(project, labels_dir, project / "sih3d_classified.laz")
            final_cloud = project / "sih3d_classified.laz"
            report["outputs"]["classified_point_cloud"] = str(final_cloud)
        if not a.no_fill:
            from sih3d import holes
            with timer("ai_gap_filling"):
                report["gap_filling"] = holes.fill_gaps(project, frames_dir, labels_dir, project / "sih3d_filled.laz")
                final_cloud = project / "sih3d_filled.laz"
                report["outputs"]["filled_point_cloud"] = str(final_cloud)
        if not a.no_level:
            from sih3d import level
            with timer("structure_levelling"):  # after gap filling, which needs the unrotated camera poses
                lv = level.level_cloud(final_cloud, project / "sih3d_final.laz", project)
                report["levelling"] = {k: v for k, v in lv.items() if k not in ("rotation", "centre")}
                if lv.get("applied"):
                    final_cloud = project / "sih3d_final.laz"
                    report["outputs"]["final_point_cloud"] = str(final_cloud)

    from sih3d import evaluate
    with timer("evaluation"):
        if a.agz_range:
            report["accuracy_camera_positions"] = evaluate.compare_agz(project, records, a.telemetry, a.agz_range[0])
        report.update(ground_truth_checks(project, a, cloud_path=final_cloud))

    with timer("viewer_export"):
        from sih3d import export
        mesh = project / "odm_texturing"
        report["viewer"] = export.export_points(
            a.name, final_cloud, evaluate.project_crs(project), title=a.name, note=f"{a.mode} · {len(records)} keyframes",
            mesh_dir=str(mesh) if (mesh / "odm_textured_model_geo.obj").exists() else None)

    report["timings_s"] = timer.times
    report["total_minutes"] = round(sum(timer.times.values()) / 60, 1)
    (work / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(json.dumps({k: report[k] for k in ("outputs", "timings_s", "total_minutes") if k in report}, indent=2))
    log(f"Report: {work / 'report.json'}")


def main():
    ap = argparse.ArgumentParser(prog="python -m sih3d", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="video + telemetry -> 3D model")
    r.add_argument("--video", required=True, type=Path)
    r.add_argument("--telemetry", required=True, type=Path, help="DJI .SRT, flight-log .csv, or AGZ 'Log Files' folder")
    r.add_argument("--agz-range", nargs=2, type=int, metavar=("FIRST", "LAST"), help="AGZ image ids of the video")
    r.add_argument("--calib", type=Path, help="lens calibration (.npz intrinsic_matrix/distCoeff or .json K/dist)")
    r.add_argument("--focal35", type=float, help="35 mm-equivalent focal length when there is no calibration")
    r.add_argument("--name", required=True)
    r.add_argument("--mode", choices=["preview", "full"], default="full")
    r.add_argument("--segment", default="longest")
    r.add_argument("--sample-fps", type=float, default=5.0)
    r.add_argument("--min-shift", type=float, default=0.10)
    r.add_argument("--max-gap", type=float, default=2.0)
    r.add_argument("--max-size", type=int, default=0)
    r.add_argument("--no-normalise", action="store_true")
    r.add_argument("--no-deblock", action="store_true")
    r.add_argument("--no-ai", action="store_true", help="skip AI labels, masks, classification and gap filling")
    r.add_argument("--no-fill", action="store_true", help="skip AI gap filling of poorly seen surfaces")
    r.add_argument("--no-level", action="store_true",
                   help="skip roll correction about the flight line from vertical AI-classified walls")
    r.add_argument("--rolling-shutter", action="store_true")
    r.add_argument("--camera-lens", default="auto",
                   choices=["auto", "perspective", "brown", "fisheye", "fisheye_opencv"],
                   help="lens model when there is no --calib (e.g. fisheye for GoPro-style wide lenses)")
    r.add_argument("--work", default="data/runs")
    r.add_argument("--projects", default="data/odm_projects")
    for p in (r, sub.add_parser("evaluate", help="accuracy of a finished run against reference data")):
        p.add_argument("--lidar", nargs="*", type=Path, help="reference LiDAR .las/.laz tiles")
        p.add_argument("--citygml", type=Path, help="official LoD2 buildings (CityGML .zip)")
        p.add_argument("--reference-crs", default="EPSG:2056", help="CRS of the reference data")
        p.add_argument("--reference-z-offset", type=float, default=-0.111,
                       help="added to reference heights to match the drone's altitude system")
    ev = sub.choices["evaluate"]
    ev.add_argument("--name", required=True)
    ev.add_argument("--projects", default="data/odm_projects")
    e = sub.add_parser("export", help="make a finished run viewable in viewer/index.html")
    e.add_argument("--name", required=True)
    e.add_argument("--title")
    e.add_argument("--note", default="")
    e.add_argument("--projects", default="data/odm_projects")
    c = sub.add_parser("cleanup", help="remove reproducible ODM intermediates for a completed run (dry-run without --yes)")
    c.add_argument("--name", required=True, help="run/project name (as passed to 'run')")
    c.add_argument("--work", default="data/runs", help="runs root (default: data/runs)")
    c.add_argument("--projects", default="data/odm_projects", help="projects root (default: data/odm_projects)")
    c.add_argument("--yes", action="store_true", help="actually delete; without it, dry-run only")
    d = sub.add_parser("disk-usage", help="show disk usage of major run/workspace directories (sorted, no deletion)")
    d.add_argument("--name", required=True, help="run/project name (as passed to 'run')")
    d.add_argument("--work", default="data/runs", help="runs root (default: data/runs)")
    d.add_argument("--projects", default="data/odm_projects", help="projects root (default: data/odm_projects)")
    d.add_argument("--viewer", default="viewer", help="viewer root (default: viewer, expects viewer/data/<name>)")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "evaluate":
        result = ground_truth_checks(Path(a.projects) / a.name, a)
        out = Path("data/runs") / a.name / "accuracy_ground_truth.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        log(json.dumps(result, indent=1, default=str))
    elif a.cmd == "export":
        from sih3d import evaluate, export
        project = Path(a.projects) / a.name
        cloud = next((c for c in (project / "sih3d_final.laz", project / "sih3d_filled.laz", project / "sih3d_classified.laz",
                                  project / "odm_georeferencing" / "odm_georeferenced_model.laz") if c.exists()))
        mesh = project / "odm_texturing"
        log(json.dumps(export.export_points(a.name, cloud, evaluate.project_crs(project), title=a.title, note=a.note,
                                            mesh_dir=str(mesh) if (mesh / "odm_textured_model_geo.obj").exists() else None)))
    elif a.cmd == "cleanup":
        from sih3d.cleanup import DELETABLE_SUBDIRS, _human_size, collect_cleanup_info, is_run_successful, run_cleanup
        import shutil as _shutil  # local alias to avoid shadowing

        work_dir = Path(a.work) / a.name
        project_dir = Path(a.projects) / a.name

        # 1. Verify existence
        if not work_dir.exists() and not project_dir.exists():
            raise SystemExit(f"Run/project '{a.name}' not found in {a.work} nor {a.projects}")
        if not project_dir.exists():
            raise SystemExit(f"Project directory not found: {project_dir}")

        # 2. Verify successful completion if report available
        successful, reason = is_run_successful(work_dir, project_dir)
        if not successful:
            raise SystemExit(
                f"Refusing cleanup: run '{a.name}' not marked successful ({reason}). "
                f"Check {work_dir / 'report.json'} or ensure final outputs exist."
            )

        # 3. Print what will be deleted and sizes
        _, total, sized = collect_cleanup_info(project_dir)
        if not sized:
            log(f"Nothing to clean for '{a.name}': no deletable intermediates found in {project_dir}")
            log(f"Deletable set is: {', '.join(DELETABLE_SUBDIRS)} (already missing)")
        else:
            log(f"Cleanup for '{a.name}' in {project_dir}:")
            for p, s in sized:
                log(f"  {p}  ({_human_size(s)})")
            log(f"Total reclaimable: {_human_size(total)} in {len(sized)} entries "
                f"(unique file bytes; hardlinked files counted once)")
            log(f"Protected (never deleted): video, telemetry, data/runs/<name>/frames, final LAZs, "
                f"odm_texturing/, viewer/data/, report.json, keyframes.json, coords.txt, reconstruction.json "
                f"(if required) – only {', '.join(DELETABLE_SUBDIRS)} are considered.")
            if not a.yes:
                log("Dry-run: pass --yes to actually delete.")
            else:
                # 7. Handle already-missing gracefully – run_cleanup does rmtree with exists check
                result = run_cleanup(a.work, a.projects, a.name, confirm=True)
                if result["deleted"]:
                    log(f"Deleted {len(result['deleted'])} directories, reclaimed {_human_size(total)}")
                else:
                    log("Nothing deleted (directories already missing).")
                if result.get("error"):
                    log(f"Warning: {result['error']}")
    elif a.cmd in ("disk-usage", "disk_usage"):
        from sih3d.disk_usage import collect_disk_usage, format_report
        work_dir = Path(a.work) / a.name
        project_dir = Path(a.projects) / a.name
        viewer_path = Path(a.viewer) / "data" / a.name
        # Gracefully handle missing – still report what exists
        if not work_dir.exists() and not project_dir.exists() and not viewer_path.exists():
            log(f"Run '{a.name}' not found in {a.work} nor {a.projects} nor {viewer_path} (all missing, handled gracefully)")
        items, total = collect_disk_usage(a.work, a.projects, a.viewer, a.name)
        for line in format_report(a.name, items, total):
            log(line)


if __name__ == "__main__":
    sys.exit(main())
