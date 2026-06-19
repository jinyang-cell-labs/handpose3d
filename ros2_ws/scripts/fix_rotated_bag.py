#!/usr/bin/env python3
"""Rewrite a rosbag, de-rotating the camera image topics and renaming them.

The ``xinyang_demo`` bag was recorded with the two camera streams rotated 90
degrees and published on the non-standard topics ``/cameraN_rot/image_rotated``.
This produces a fresh bag whose images are rotated back (clockwise 90) and
published on the conventional ``/cameraN/image_raw`` topics. ``camera_info``
(and any other) topics are copied through byte-for-byte, untouched.

Must run inside the ROS 2 (jazzy) container, e.g.:

    docker compose -f docker/docker-compose.yaml exec handpose3d bash
    source /opt/ros/jazzy/setup.bash
    python3 /workspace/ros2_ws/scripts/fix_rotated_bag.py \
        /workspace/ros2_ws/recordings/xinyang_demo \
        /workspace/ros2_ws/recordings/xinyang_demo_fixed
"""
import argparse

import numpy as np
from rclpy.serialization import deserialize_message, serialize_message
from rosbag2_py import (
    ConverterOptions,
    ReadOrder,
    ReadOrderSortBy,
    SequentialReader,
    SequentialWriter,
    StorageOptions,
    TopicMetadata,
)
from sensor_msgs.msg import Image

# Bytes per pixel for the encodings we expect in this bag. Extend if needed.
_CHANNELS = {
    "mono8": 1,
    "bgr8": 3,
    "rgb8": 3,
    "bgra8": 4,
    "rgba8": 4,
}

# Image topics to de-rotate and rename: old name -> new name.
RENAME = {
    "/camera0_rot/image_rotated": "/camera0/image_raw",
    "/camera1_rot/image_rotated": "/camera1/image_raw",
}


# np.rot90 k: positive is counter-clockwise. Map directions to k.
_ROT_K = {"cw": -1, "ccw": 1, "180": 2}


def rotate_image(msg: Image, direction: str) -> Image:
    """Return a new Image rotated by the given direction (cw/ccw/180)."""
    ch = _CHANNELS.get(msg.encoding)
    if ch is None:
        raise ValueError(f"unsupported encoding {msg.encoding!r}")

    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, ch)
    rot = np.ascontiguousarray(np.rot90(arr, k=_ROT_K[direction]))
    new_h, new_w = rot.shape[:2]

    out = Image()
    out.header = msg.header  # preserve timestamp + frame_id
    out.height = new_h
    out.width = new_w
    out.encoding = msg.encoding
    out.is_bigendian = msg.is_bigendian
    out.step = new_w * ch
    out.data = rot.tobytes()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="input bag directory")
    ap.add_argument("output", help="output bag directory (must not exist)")
    ap.add_argument("--storage", default="mcap", help="storage id (default: mcap)")
    ap.add_argument("--rotate", default="cw", choices=["cw", "ccw", "180"],
                    help="image rotation to apply (default: cw, to match camera_info)")
    args = ap.parse_args()

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.input, storage_id=args.storage),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"),
    )
    # The source mcap lacks a message index, so the default timestamp-ordered
    # read bails after the first chunk. Read in stored file order instead.
    reader.set_read_order(ReadOrder(ReadOrderSortBy.File, False))

    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=args.output, storage_id=args.storage),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"),
    )

    # Register output topics, applying the rename map. Keep types/QoS as-is.
    type_by_topic = {}
    for t in reader.get_all_topics_and_types():
        name = RENAME.get(t.name, t.name)
        type_by_topic[t.name] = t.type
        writer.create_topic(
            TopicMetadata(
                id=t.id,
                name=name,
                type=t.type,
                serialization_format=t.serialization_format,
                offered_qos_profiles=t.offered_qos_profiles,
                type_description_hash=t.type_description_hash,
            )
        )

    n_img = n_copy = 0
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic in RENAME:
            msg = deserialize_message(data, Image)
            out = rotate_image(msg, args.rotate)
            writer.write(RENAME[topic], serialize_message(out), stamp)
            n_img += 1
        else:
            # Pass through untouched (same serialized bytes, same timestamp).
            writer.write(topic, data, stamp)
            n_copy += 1

    print(f"done: rotated+renamed {n_img} image msgs, copied {n_copy} others")


if __name__ == "__main__":
    main()
