#!/usr/bin/env python3
"""Turn a GPS-tagged photo sequence into a drone-style video plus a DJI-style .SRT.

Most open drone datasets ship photos, while SIH26158 input is video + telemetry. This rebuilds that
input (with real H.264 compression) so the video path of prepare_dataset.py can be tested end to end.
The .SRT carries only what the photos really contain: time, latitude, longitude, altitude.

  python scripts/images_to_video.py --images data/odm_aukerman_baseline/images --out data/simulated/aukerman.mp4
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import order_images  # noqa: E402


def srt_time(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output .mp4; the .SRT is written next to it")
    ap.add_argument("--seconds-per-image", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920, help="output width (1920 = 1080p-class, 3840 = 4K)")
    ap.add_argument("--crf", type=int, default=23, help="H.264 quality; higher = more compression artefacts")
    args = ap.parse_args()

    items = [it for it in order_images(args.images) if it[1]]
    if len(items) < 2:
        sys.exit(f"{args.images}: need at least 2 images with EXIF GPS")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dt = args.seconds_per_image

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p, _, _ in items:
            f.write(f"file '{p.resolve()}'\nduration {dt}\n")
        f.write(f"file '{items[-1][0].resolve()}'\n")  # concat demuxer ignores the last duration otherwise
        listing = f.name

    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listing,
           "-vf", f"scale={args.width}:-2:flags=lanczos,format=yuv420p", "-r", str(args.fps),
           "-c:v", "libx264", "-crf", str(args.crf), "-preset", "medium", str(args.out)]
    print("Encoding", len(items), "images ->", args.out, file=sys.stderr)
    subprocess.run(cmd, check=True)
    Path(listing).unlink(missing_ok=True)

    alt0 = items[0][1][2]
    blocks = []
    for k, (p, (lat, lon, alt), _) in enumerate(items):
        blocks.append(f"{k + 1}\n{srt_time(k * dt)} --> {srt_time((k + 1) * dt)}\n"
                      f"FrameCnt: {k + 1}, source: {p.name}\n"
                      f"[latitude: {lat:.7f}] [longitude: {lon:.7f}] [rel_alt: {alt - alt0:.3f} abs_alt: {alt:.3f}]\n")
    srt = args.out.with_suffix(".SRT")
    srt.write_text("\n".join(blocks))
    print("Wrote", srt, file=sys.stderr)


if __name__ == "__main__":
    main()
