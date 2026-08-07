#!/bin/bash
# Builds the ffs image.
# Usage:
#   ./build_image.sh       Python + TensorRT image (docker/dockerfile)       -> "ffs"
#   ./build_image.sh cpp   Adds TensorRT/OpenCV C++ dev headers (docker/dockerfile_cpp) -> "ffs-cpp"
set -e

cd "$(dirname "$0")"

DOCKERFILE="dockerfile"
IMAGE_NAME="ffs"
if [ "$1" = "cpp" ]; then
  DOCKERFILE="dockerfile_cpp"
  IMAGE_NAME="ffs-cpp"
fi

docker build --network host \
  --build-arg USERNAME=$(id -un) \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -t "$IMAGE_NAME" \
  -f "$DOCKERFILE" \
  ..
