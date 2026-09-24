"""Disk-usage report for a Flight2World run/workspace.

Lists the size of major directories/files used by the pipeline, sorted
largest to smallest, with two totals:

- Logical total: the sum of the displayed row sizes.  A hardlinked file
  has one name in each target it appears in, so its bytes are counted
  once per row.
- Unique total: each file counted once by (st_dev, st_ino).  Hardlinks
  (same inode) contribute only once; independent copies (different
  inodes, even with identical content) count in full.  This is unique
  file bytes, not exact filesystem allocation.

Missing paths are ignored gracefully and no files are ever modified.
"""

from __future__ import annotations

from pathlib import Path


def _iter_files(path: Path):
    """Yield (file_path, stat_result) for each regular file at/under *path*.

    *path* itself is yielded when it is a regular file; otherwise its
    contents are walked recursively.  Symlink handling is unchanged from
    the original implementation (file symlinks are followed via is_file,
    directory symlinks are not traversed by rglob).  Individual stat
    failures are skipped silently.
    """
    if path.is_file():
        try:
            yield path, path.stat()
        except OSError:
            return
        return
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    yield p, p.stat()
                except OSError:
                    pass
    except OSError:
        pass


def _dir_size(path: Path) -> int:
    """Logical size in bytes of everything at/under *path* (no dedup)."""
    total = 0
    for _, st in _iter_files(path):
        total += st.st_size
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
) -> tuple[list[tuple[str, Path, int]], tuple[int, int]]:
    """Return (sorted_items, (logical_total, unique_total)).

    sorted_items is list of (label, path, size) for existing entries,
    sorted largest to smallest.  Missing paths are skipped gracefully.
    logical_total is the sum of the displayed row sizes; unique_total
    counts each (st_dev, st_ino) once, so hardlinked files contribute
    their bytes a single time while separate copies count in full.
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
    seen_inodes: set[tuple[int, int]] = set()
    unique_total = 0

    def _record(st) -> None:
        """Add st_size to unique_total the first time an inode is seen.

        The first occurrence of a (st_dev, st_ino) contributes its full
        st_size; the same inode seen again (a hardlink in another target)
        contributes 0.  A different inode – e.g. the shutil.copy2 fallback
        of _hardlink_or_copy – counts in full even with identical content.
        """
        nonlocal unique_total
        key = (st.st_dev, st.st_ino)
        if key not in seen_inodes:
            seen_inodes.add(key)
            unique_total += st.st_size

    def _walk_size(p: Path) -> int:
        """Logical size of directory *p*, recording its files' inodes."""
        size = 0
        for _, st in _iter_files(p):
            size += st.st_size
            _record(st)
        return size

    for label, p in dir_targets:
        if p.exists():
            # if it's a file (unexpected) use file size, else dir size
            if p.is_file():
                try:
                    st = p.stat()
                except OSError:
                    continue
                items.append((label, p, st.st_size))
                _record(st)
            else:
                items.append((label, p, _walk_size(p)))

    # Final LAZ files at the project root.  odm_georeferenced_model.laz is
    # intentionally NOT listed here: it lives inside the recursively
    # counted odm_georeferencing/ directory target above, and a separate
    # row would double-count it.
    laz_candidates: list[tuple[str, Path]] = [
        (f"data/odm_projects/{name}/sih3d_classified.laz", project / "sih3d_classified.laz"),
        (f"data/odm_projects/{name}/sih3d_filled.laz", project / "sih3d_filled.laz"),
        (f"data/odm_projects/{name}/sih3d_final.laz", project / "sih3d_final.laz"),
    ]
    for label, p in laz_candidates:
        if p.exists() and p.is_file():
            try:
                st = p.stat()
            except OSError:
                continue
            items.append((label, p, st.st_size))
            _record(st)

    # Viewer
    for label, p in viewer_candidates:
        if p.exists():
            if p.is_dir():
                sz = _walk_size(p)
            else:
                st = p.stat()
                _record(st)
                sz = st.st_size
            items.append((label, p, sz))

    # Sort largest to smallest
    items.sort(key=lambda x: x[2], reverse=True)
    logical_total = sum(sz for _, _, sz in items)
    return items, (logical_total, unique_total)


def format_report(
    name: str,
    items: list[tuple[str, Path, int]],
    total: tuple[int, int],
) -> list[str]:
    """Return human-readable report lines (without printing).

    total is (logical_total, unique_total): logical_total is the sum of
    the displayed row sizes; unique_total counts each file once by
    (st_dev, st_ino) – unique file bytes, not filesystem allocation.
    """
    logical, unique = total
    lines: list[str] = []
    if not items:
        lines.append(f"No data found for run '{name}'. Checked work/projects/viewer locations (missing directories handled gracefully).")
        lines.append(f"Logical total: {_human_size(0)}")
        lines.append(f"Unique total:  {_human_size(0)}")
        return lines
    lines.append(f"Disk usage for run '{name}':")
    for label, path, size in items:
        lines.append(f"  {_human_size(size):>10}  {label}")
        # Optionally also show absolute path for debugging – keep label as primary
        # lines.append(f"    -> {path}")
    lines.append(f"Logical total: {_human_size(logical)} ({logical} bytes) in {len(items)} entries")
    lines.append(
        f"Unique total:  {_human_size(unique)} ({unique} bytes) "
        f"- unique file bytes, hardlinks counted once by inode (not filesystem allocation)"
    )
    return lines
