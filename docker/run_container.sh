# Usage:
#   ./run_container.sh       runs the "ffs" image (docker/dockerfile)
#   ./run_container.sh cpp   runs the "ffs-cpp" image (docker/dockerfile_cpp)
IMAGE_NAME="ffs"
CONTAINER_NAME="ffs"
if [ "$1" = "cpp" ]; then
  IMAGE_NAME="ffs-cpp"
  CONTAINER_NAME="ffs-cpp"
fi

USERNAME="$(id -un)"
PROJ_DIR="/home/${USERNAME}/Research/Fast-FoundationStereo"
DATA_DIR="/media/${USERNAME}/T73"
XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

# Build X11 args only when a display and auth file are available
X11_ARGS=()
if [ -n "$DISPLAY" ]; then
    X11_ARGS+=(--env="DISPLAY=$DISPLAY")
    X11_ARGS+=(--env="QT_X11_NO_MITSHM=1")
    [ -S /tmp/.X11-unix ] && X11_ARGS+=(--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw")
    if [ -f "$XAUTHORITY" ]; then
        X11_ARGS+=(--env="XAUTHORITY=/tmp/.Xauthority")
        X11_ARGS+=(--volume="$XAUTHORITY:/tmp/.Xauthority:ro")
    fi
fi

docker run --init -it \
    --name="$CONTAINER_NAME" \
    --gpus=all \
    --runtime=nvidia \
    --env="NVIDIA_DISABLE_REQUIRE=1" \
    --net="host" \
    --ipc="host" \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --workdir="/home/$USERNAME/Fast-FoundationStereo" \
    --env="XDG_RUNTIME_DIR=/tmp/runtime-$USERNAME" \
    --env="USER_ID=$(id -u)" \
    --env="GROUP_ID=$(id -g)" \
    "${X11_ARGS[@]}" \
    --volume="$PROJ_DIR:/home/$USERNAME/Fast-FoundationStereo" \
    --volume="$DATA_DIR:/home/$USERNAME/data" \
    --volume="/home/$USERNAME/.bash_aliases:/home/$USERNAME/.bash_aliases" \
    --volume="/home/$USERNAME/.ssh:/home/$USERNAME/.ssh:ro" \
    --volume="/etc/localtime:/etc/localtime:ro" \
    --volume="/etc/timezone:/etc/timezone:ro" \
    --volume /tmp/runtime-$USERNAME:/tmp/runtime-$USERNAME \
    "$IMAGE_NAME" \
    /bin/bash
