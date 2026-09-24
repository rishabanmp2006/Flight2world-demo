#!/usr/bin/env python3
"""Download the datasets and model weights that sih3d uses -- only the parts it needs.

Nothing fetched here is committed to git (data/ and models/ are ignored). Run from this folder:

    python download_data.py --list
    python download_data.py models osm aukerman      # model weights + the smallest dataset
    python download_data.py semcity                  # metric-accuracy test set
    python download_data.py agz zurich-gt            # Zurich video-rate set + its ground truth
    python download_data.py uavscenes                # primary development set
    python download_data.py all

AGZ.zip (27.8 GB) and the UAVScenes zips (33 GB) are never downloaded whole. Their zip central
directory is read with HTTP Range requests and only the members sih3d uses are fetched, so the
three sets need about 4 GB instead of 60 GB. Re-running skips files already present at the
expected size, so an interrupted download can simply be started again.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "sih3d-download/1.0"}
BLOCK = 16 * 1024 * 1024          # read-ahead window for ranged zip reads


# ----------------------------------------------------------------------------- transport

def _open(req, timeout=120, tries=5):
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403, 404, 410, 416):
                raise
            if attempt == tries - 1:
                raise
            wait = 2 ** attempt
            print(f"    network error ({e}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def _mb(n):
    return f"{n / 2**20:,.1f} MB"


def download(url, dest: Path, size=None):
    """Stream url to dest atomically. Skips when dest already exists at the expected size."""
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with _open(urllib.request.Request(url, headers=UA), timeout=300) as r, part.open("wb") as fh:
        total = size or int(r.headers.get("Content-Length") or 0)
        done, last = 0, time.time()
        while chunk := r.read(4 * 1024 * 1024):
            fh.write(chunk)
            done += len(chunk)
            if time.time() - last > 5:
                pct = f" {done / total:5.1%}" if total else ""
                print(f"    {dest.name}: {_mb(done)}{pct}", flush=True)
                last = time.time()
    if size is not None and part.stat().st_size != size:
        raise SystemExit(f"size mismatch for {dest.name}: got {part.stat().st_size}, expected {size}")
    part.replace(dest)
    return "downloaded"


class RemoteFile(io.RawIOBase):
    """Seekable read-only view of a URL over HTTP Range, with a read-ahead block cache.

    zipfile only needs seek/tell/read, so it reads the central directory and the chosen
    members without the rest of the archive ever being transferred. Signed CDN URLs
    (Hugging Face) expire, so a rejected range request re-resolves the original URL.
    """

    def __init__(self, url):
        self.origin = url
        self._resolve()
        self.pos = 0
        self._buf, self._buf_start = b"", 0

    def _resolve(self):
        with _open(urllib.request.Request(self.origin, method="HEAD", headers=UA), timeout=90) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.geturl()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _fetch(self, start, length):
        end = min(start + length, self.size) - 1
        for attempt in range(2):
            req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={start}-{end}"})
            try:
                with _open(req, timeout=300) as r:
                    if r.status != 206:
                        raise SystemExit(f"server ignored the Range request for {self.origin} "
                                         f"(HTTP {r.status}); refusing to pull the whole archive")
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code in (400, 403, 410) and attempt == 0:
                    self._resolve()          # signed URL expired
                    continue
                raise

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        buf_end = self._buf_start + len(self._buf)
        if not (self._buf_start <= self.pos and self.pos + n <= buf_end):
            self._buf = self._fetch(self.pos, max(n, BLOCK))
            self._buf_start = self.pos
        off = self.pos - self._buf_start
        data = self._buf[off:off + n]
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def extract_members(url, select, dest_root: Path, strip_prefix="", rename=None):
    """Extract the zip members at url for which select(name) is true, without the whole zip."""
    print(f"  reading archive index: {url}", flush=True)
    zf = zipfile.ZipFile(RemoteFile(url))
    chosen = sorted((i for i in zf.infolist() if not i.is_dir() and select(i.filename)),
                    key=lambda i: i.header_offset)   # archive order = sequential range reads
    if not chosen:
        raise SystemExit(f"no members matched in {url}")
    total = sum(i.file_size for i in chosen)
    print(f"  {len(chosen):,} members, {_mb(total)} of a {_mb(zf.fp.size)} archive", flush=True)
    done = skipped = 0
    last = time.time()
    for k, info in enumerate(chosen, 1):
        rel = info.filename[len(strip_prefix):] if info.filename.startswith(strip_prefix) else info.filename
        out = dest_root / (rename(rel) if rename else rel)
        if out.exists() and out.stat().st_size == info.file_size:
            skipped += 1
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.name + ".part")
            with zf.open(info) as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            part.replace(out)
        done += info.file_size
        if time.time() - last > 5 or k == len(chosen):
            print(f"    {k:,}/{len(chosen):,} files  {_mb(done)} / {_mb(total)}", flush=True)
            last = time.time()
    return len(chosen), skipped


# ----------------------------------------------------------------------------- datasets

HF = "https://huggingface.co"
SWISSTOPO = "https://data.geo.admin.ch"


def get_models(root: Path, args):
    """YOLO11-seg weights, plus the two Hugging Face models cached exactly as the pipeline loads them."""
    download("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
             root / "models" / "yolo11n-seg.pt", size=6_182_636)
    print("  yolo11n-seg.pt ready")
    try:
        from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                                  SegformerForSemanticSegmentation)
    except ImportError:
        raise SystemExit("transformers is not installed: pip install -r requirements.txt first")
    cache = str(root / "models" / "hf")
    for repo, model_cls in (("nvidia/segformer-b2-finetuned-ade-512-512", SegformerForSemanticSegmentation),
                            ("depth-anything/Depth-Anything-V2-Small-hf", AutoModelForDepthEstimation)):
        print(f"  caching {repo} -> models/hf", flush=True)
        AutoImageProcessor.from_pretrained(repo, cache_dir=cache)
        model_cls.from_pretrained(repo, cache_dir=cache)
    print("  Hugging Face models ready")


def get_osm(root: Path, args):
    """The four Aukerman building outlines used by scripts/measure_building.py."""
    dest = root / "data" / "ground_truth" / "aukerman_osm_buildings.json"
    if dest.exists():
        print("  skipped (present)")
        return
    query = "[out:json][timeout:60];way(id:530052613,749959908,749959909,749959910);out geom;"
    body = urllib.parse.urlencode({"data": query}).encode()
    for endpoint in ("https://overpass-api.de/api/interpreter",
                     "https://overpass.kumi.systems/api/interpreter"):
        try:
            with _open(urllib.request.Request(endpoint, data=body, headers=UA), timeout=120) as r:
                data = json.load(r)
            break
        except Exception as e:                      # noqa: BLE001 -- try the mirror
            print(f"    {endpoint}: {e}")
    else:
        raise SystemExit("no Overpass endpoint reachable")
    if len(data.get("elements", [])) != 4:
        raise SystemExit(f"expected 4 buildings from Overpass, got {len(data.get('elements', []))}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=1))
    print(f"  wrote {dest.relative_to(root)}")


def get_aukerman(root: Path, args):
    """OpenDroneMap's Aukerman sample: 77 nadir photos with EXIF GPS (CC0)."""
    out = root / "data" / "odm_aukerman_baseline"
    images = out / "odm_data_aukerman-master" / "images"
    if images.is_dir() and len(list(images.iterdir())) >= 77:
        print("  skipped (77 images present)")
        return
    archive = root / "data" / "_downloads" / "odm_data_aukerman-master.zip"
    download("https://github.com/OpenDroneMap/odm_data_aukerman/archive/refs/heads/master.zip", archive)
    print("  extracting", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out)
    if not args.keep_archives:
        archive.unlink()
    print(f"  {len(list(images.iterdir()))} images in {images.relative_to(root)}")


def get_semcity(root: Path, args):
    """SemCityLockeD: the 170-frame continuous nadir run (RTK GPS) + official LoD building models."""
    repo = "datasets/DevinEaston/SemCityLockeD"
    dest = root / "data" / "semcitylocked"

    def tree(path):
        with _open(urllib.request.Request(f"{HF}/api/{repo}/tree/main/{path}", headers=UA)) as r:
            return json.load(r)

    def size_of(entry):
        return (entry.get("lfs") or {}).get("size") or entry.get("size")

    # the run is DJI_20241217084340_0034_D.JPG .. DJI_20241217084919_0203_D.JPG, 170 frames
    frames = [e for e in tree("original_images")
              if "20241217084340" <= e["path"].split("/")[-1].split("_")[1] <= "20241217084919"]
    if len(frames) != 170:
        raise SystemExit(f"expected 170 frames in the SemCityLockeD run, found {len(frames)}")
    lod = [e for e in tree("LoD_model") if e["type"] == "file"
           and (args.with_lod3 or not e["path"].endswith("lod3.obj"))]
    lod += tree("LoD_model/semantic_obj")
    files = frames + [e for e in lod if e["type"] == "file"]
    files.append({"path": "README.md", "size": None})
    got = 0
    for k, e in enumerate(files, 1):
        status = download(f"{HF}/{repo}/resolve/main/{urllib.parse.quote(e['path'])}",
                          dest / e["path"], size=size_of(e))
        got += status == "downloaded"
        if k % 20 == 0 or k == len(files):
            print(f"    {k}/{len(files)} files", flush=True)
    print(f"  {got} downloaded, {len(files) - got} already present")


def get_uavscenes(root: Path, args):
    """UAVScenes AMtown03: 1,120 nadir frames + RTK CSV + poses, every Nth LiDAR scan, reference mesh."""
    dest = root / "data" / "uavscenes_amtown03"
    prefix = "interval5_CAM_LIDAR/interval5_AMtown03/"
    lidar_every = max(1, args.lidar_every)

    zf = zipfile.ZipFile(RemoteFile(f"{HF}/datasets/sijieaaa/UAVScenes/resolve/main/interval5_CAM_LIDAR.zip"))
    lidar = sorted(i.filename for i in zf.infolist() if i.filename.startswith(prefix + "interval5_LIDAR/")
                   and not i.is_dir())
    keep_lidar = set(lidar[lidar_every - 1::lidar_every])     # the Nth, 2Nth, ... scan, as in the original
    zf.close()
    print(f"  LiDAR: keeping every {lidar_every}th scan -> {len(keep_lidar)} of {len(lidar)}")

    def select(name):
        if not name.startswith(prefix):
            return False
        if "/interval5_LIDAR/" in name:
            return name in keep_lidar
        return True

    extract_members(f"{HF}/datasets/sijieaaa/UAVScenes/resolve/main/interval5_CAM_LIDAR.zip",
                    select, dest, strip_prefix="interval5_CAM_LIDAR/")
    mesh_members = {"terra_3dmap_pointcloud_mesh/AMtown/Mesh.ply",
                    "terra_3dmap_pointcloud_mesh/AMtown/terra_ply/metadata.xml"}
    extract_members(f"{HF}/datasets/sijieaaa/UAVScenes/resolve/main/terra_3dmap_pointcloud_mesh.zip",
                    lambda n: n in mesh_members, dest / "reference_mesh",
                    strip_prefix="terra_3dmap_pointcloud_mesh/")


def get_agz(root: Path, args):
    """Zurich AGZ: a continuous 30 fps street flight (frames 59001-62000 by default) + logs + calibration."""
    first, last = args.agz_range
    dest = root / "data" / f"agz_zurich_seg{first}"
    extras = {"AGZ/calibration_data.npz", "AGZ/readme.txt", "AGZ/loadGroundTruthAGL.m",
              "AGZ/plotPath.m", "AGZ/write_ros_bag.py"}

    def select(name):
        if name in extras or name.startswith(("AGZ/Log Files/", "AGZ/MAV Images Calib/")):
            return True
        if name.startswith("AGZ/MAV Images/") and name.endswith(".jpg"):
            stem = name[len("AGZ/MAV Images/"):-4]
            return stem.isdigit() and first <= int(stem) <= last
        return False

    extract_members("https://download.ifi.uzh.ch/rpg/AGZ_data/AGZ.zip", select, dest, strip_prefix="AGZ/")


def get_zurich_gt(root: Path, args):
    """swisstopo ground truth for the AGZ flight: official LoD2 buildings, LiDAR, 10 cm orthophoto."""
    dest = root / "data" / "ground_truth" / "zurich"
    # the pipeline reads the CityGML straight from its zip, so it stays zipped
    download(f"{SWISSTOPO}/ch.swisstopo.swissbuildings3d_3_0/swissbuildings3d_3_0_2019_1091-23/"
             "swissbuildings3d_3_0_2019_1091-23_2056_5728.citygml.zip",
             dest / "swissbuildings3d_3_0_2019_1091-23_2056_5728.citygml.zip", size=190_870_064)
    for tile in ("2683-1248", "2683-1249"):
        name = f"swisssurface3d_2018_{tile}_2056_5728.las.zip"
        url = f"{SWISSTOPO}/ch.swisstopo.swisssurface3d/swisssurface3d_2018_{tile}/{name}"
        if args.keep_archives:
            download(url, dest / name)
        las = tile.replace("-", "_") + ".las"
        # --lidar expects the extracted .las; pull that member without storing the zip
        extract_members(url, lambda n, las=las: n == las, dest)
    for tile, size in (("2683-1248", 61_148_414), ("2683-1249", 62_959_521)):
        name = f"swissimage-dop10_2019_{tile}_0.1_2056.tif"
        download(f"{SWISSTOPO}/ch.swisstopo.swissimage-dop10/swissimage-dop10_2019_{tile}/{name}",
                 dest / name, size=size)
    print(f"  ground truth ready in {dest.relative_to(root)}")


TARGETS = {
    "models":    (get_models,    "~0.3 GB", "YOLO11n-seg (AGPL-3.0), SegFormer-B2 ADE20k, Depth Anything V2 Small"),
    "osm":       (get_osm,       "<1 MB",   "Aukerman building outlines, OpenStreetMap (ODbL)"),
    "aukerman":  (get_aukerman,  "~0.5 GB", "OpenDroneMap Aukerman sample, 77 photos (CC0)"),
    "semcity":   (get_semcity,   "~1.0 GB", "SemCityLockeD 170-frame RTK run + LoD models (CC BY-NC 4.0)"),
    "agz":       (get_agz,       "~1.1 GB", "Zurich AGZ frames 59001-62000 + logs (academic research)"),
    "zurich-gt": (get_zurich_gt, "~1.2 GB", "swisstopo LiDAR, buildings, orthophoto (free geodata, attribution)"),
    "uavscenes": (get_uavscenes, "~2.1 GB", "UAVScenes AMtown03 + reference mesh (CC BY-NC-SA 4.0)"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help=f"one or more of: {', '.join(TARGETS)}, all")
    ap.add_argument("--list", action="store_true", help="show what each target downloads")
    ap.add_argument("--root", type=Path, default=HERE,
                    help="folder that receives data/ and models/ (default: this folder)")
    ap.add_argument("--lidar-every", type=int, default=10,
                    help="UAVScenes: keep every Nth LiDAR scan (default 10; 1 = all 1,120)")
    ap.add_argument("--agz-range", type=int, nargs=2, default=(59001, 62000), metavar=("FIRST", "LAST"),
                    help="AGZ image ids to extract (default 59001 62000)")
    ap.add_argument("--with-lod3", action="store_true", help="SemCityLockeD: also fetch the 108 MB LoD3 model")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep downloaded zips (Aukerman, swissSURFACE3D) instead of deleting them")
    args = ap.parse_args()

    if args.list or not args.targets:
        print("targets (sizes are what lands on disk):")
        for name, (_, size, what) in TARGETS.items():
            print(f"  {name:10s} {size:>8s}  {what}")
        print("\nLicences differ per dataset; several are non-commercial. See DATA.md before using the data.")
        return 0 if args.list else 1

    wanted = list(TARGETS) if "all" in args.targets else args.targets
    unknown = [t for t in wanted if t not in TARGETS]
    if unknown:
        ap.error(f"unknown target(s): {', '.join(unknown)}")
    root = args.root.resolve()
    for name in wanted:
        fn, size, what = TARGETS[name]
        print(f"\n== {name}: {what} ({size})", flush=True)
        t0 = time.time()
        fn(root, args)
        print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
