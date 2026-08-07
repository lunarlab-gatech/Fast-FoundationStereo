# Usage:
#   ./enter_container.sh       enters the "ffs" container
#   ./enter_container.sh cpp   enters the "ffs-cpp" container
CONTAINER_NAME="ffs"
if [ "$1" = "cpp" ]; then
  CONTAINER_NAME="ffs-cpp"
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != "true" ]; then
    docker start "$CONTAINER_NAME" 2>/dev/null || (echo "Container not found. Run run_container.sh first." && exit 1)
fi

docker exec -it "$CONTAINER_NAME" /bin/bash
