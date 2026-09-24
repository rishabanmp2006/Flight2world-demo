#!/usr/bin/env bash
# Build a georeferenced, textured 3D model with OpenDroneMap (Docker, native on Apple Silicon).
# Usage: scripts/run_odm.sh <project-name> [extra ODM flags]
# Expects data/odm_projects/<project-name>/images/ (e.g. hard-linked from data/processed/<name>/images).
#
# Main outputs in data/odm_projects/<project-name>/:
#   odm_texturing/odm_textured_model_geo.obj        textured mesh     -> MeshLab / Blender
#   odm_georeferencing/odm_georeferenced_model.laz  point cloud (UTM) -> CloudCompare
#   odm_orthophoto/odm_orthophoto.tif               map image         -> QGIS
#   odm_dem/dsm.tif                                 surface heights   -> QGIS
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?usage: scripts/run_odm.sh <project-name> [extra ODM flags]}"
shift
PROJECT="$ROOT/data/odm_projects/$NAME"
[ -d "$PROJECT/images" ] || { echo "Missing $PROJECT/images" >&2; exit 1; }

# ODM peaks at ~1 GB per thread, so size threads to Docker's memory limit, not the CPU count
# (12 threads in an 8 GB Docker VM ran out of memory during texturing). Extra flags override this.
CPUS="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
DOCKER_GB="$(( $(docker info --format '{{.MemTotal}}') / 1073741824 ))"
THREADS="$(( DOCKER_GB - 3 < CPUS ? DOCKER_GB - 3 : CPUS ))"
[ "$THREADS" -ge 2 ] || THREADS=2
echo "Docker memory ${DOCKER_GB} GB -> ${THREADS} threads" | tee -a "$PROJECT/odm_run.log"

docker run --rm -v "$ROOT/data/odm_projects:/datasets" opendronemap/odm \
  --project-path /datasets "$NAME" \
  --dsm \
  --pc-quality medium \
  --mesh-size 300000 \
  --max-concurrency "$THREADS" \
  "$@" 2>&1 | tee -a "$PROJECT/odm_run.log"
