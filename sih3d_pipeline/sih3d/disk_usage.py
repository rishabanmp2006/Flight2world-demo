"""Disk-usage report for a Flight2World run/workspace.

Lists the size of major directories/files used by the pipeline, sorted
largest to smallest, with a total. Missing paths are ignored gracefully
and no files are ever modified.
"""

from __future__ import annotations

from pathlib import Path


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    return f"{n / 1024 / 1024 / 1024:.2f} GB"


def collect_disk_usage(
    work_root: Path | str,
    projects_root: Path | str,
    viewer_root: Path | str,
    name: str,
) -> tuple[list[tuple[str, Path, int]], int]:
    """Return (sorted_items, total_bytes).

    sorted_items is list of (label, path, size) for existing entries,
    sorted largest to smallest.  Missing paths are skipped gracefully.
    """
    work_root = Path(work_root)
    projects_root = Path(projects_root)
    viewer_root = Path(viewer_root)

    work = work_root / name
    project = projects_root / name
    # viewer/data/<name> – viewer_root is typically "viewer" inside sih3d_pipeline
    viewer_path = viewer_root / "data" / name
    # Handle case where caller passes viewer/data as viewer_root directly
    # (e.g. viewer_root already ends with "data"). Check alternative.
    alt_viewer = viewer_root / name
    # Prefer viewer_path, fallback to alt_viewer if viewer_path missing and alt exists.
    # We will check both but avoid duplicate.
    viewer_candidates = []
    if viewer_path.exists():
        viewer_candidates.append(("viewer/data/" + name, viewer_path))
    elif alt_viewer.exists() and alt_viewer != viewer_path:
        # caller passed viewer/data as root
        viewer_candidates.append((str(alt_viewer), alt_viewer))

    # Major directories – label is the relative path as described in the task
    # for readability in the report.
    dir_targets: list[tuple[str, Path]] = [
        (f"data/runs/{name}/frames", work / "frames"),
        (f"data/runs/{name}/labels", work / "labels"),
        (f"data/runs/{name}/masks", work / "masks"),
        (f"data/odm_projects/{name}/images", project / "images"),
        (f"data/odm_projects/{name}/opensfm", project / "opensfm"),
        (f"data/odm_projects/{name}/openmvs", project / "openmvs"),
        (f"data/odm_projects/{name}/odm_filterpoints", project / "odm_filterpoints"),
        (f"data/odm_projects/{name}/odm_meshing", project / "odm_meshing"),
        (f"data/odm_projects/{name}/odm_georeferencing", project / "odm_georeferencing"),
        (f"data/odm_projects/{name}/odm_texturing", project / "odm_texturing"),
    ]

    items: list[tuple[str, Path, int]] = []

    for label, p in dir_targets:
        if p.exists():
            # if it's a file (unexpected) use file size, else dir size
            if p.is_file():
                try:
                    sz = p.stat().st_size
                except OSError:
                    continue
            else:
                sz = _dir_size(p)
            items.append((label, p, sz))

    # Final LAZ files – check both project root and odm_georeferencing location
    laz_candidates: list[tuple[str, Path]] = [
        (f"data/odm_projects/{name}/odm_georeferencing/odm_georeferenced_model.laz", project / "odm_georeferencing" / "odm_georeferenced_model.laz"),
        (f"data/odm_projects/{name}/sih3d_classified.laz", project / "sih3d_classified.laz"),
        (f"data/odm_projects/{name}/sih3d_filled.laz", project / "sih3d_filled.laz"),
        (f"data/odm_projects/{name}/sih3d_final.laz", project / "sih3d_final.laz"),
    ]
    for label, p in laz_candidates:
        if p.exists() and p.is_file():
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            items.append((label, p, sz))

    # Viewer
    for label, p in viewer_candidates:
        if p.exists():
            sz = _dir_size(p) if p.is_dir() else p.stat().st_size
            items.append((label, p, sz))

    # Sort largest to smallest
    items.sort(key=lambda x: x[2], reverse=True)
    total = sum(sz for _, _, sz in items)
    return items, total


def format_report(name: str, items: list[tuple[str, Path, int]], total: int) -> list[str]:
    """Return human-readable report lines (without printing)."""
    lines: list[str] = []
    if not items:
        lines.append(f"No data found for run '{name}'. Checked work/projects/viewer locations (missing directories handled gracefully).")
        lines.append(f"Total: {_human_size(0)}")
        return lines
    lines.append(f"Disk usage for run '{name}':")
    for label, path, size in items:
        lines.append(f"  {_human_size(size):>10}  {label}")
        # Optionally also show absolute path for debugging – keep label as primary
        # lines.append(f"    -> {path}")
    lines.append(f"Total: {_human_size(total)} ({total} bytes) in {len(items)} entries")
    return lines
