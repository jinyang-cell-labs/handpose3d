#!/bin/bash
# Download the MediaPipe HandLandmarker model bundle used by handpose_estimation.
set -e

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ros2_ws/src/handpose_estimation/models"
DEST="${DEST_DIR}/hand_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

mkdir -p "${DEST_DIR}"

if [ -f "${DEST}" ]; then
    echo "Model already present at ${DEST}"
    exit 0
fi

echo "Downloading hand_landmarker.task -> ${DEST}"
curl -L -o "${DEST}" "${URL}"
echo "Done."
