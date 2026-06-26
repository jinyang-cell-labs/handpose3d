#!/usr/bin/env python3
"""One-shot AprilGrid detection diagnostic.

Grabs a single frame from each camera topic, saves it to /tmp/detect_debug_<cam>.png,
and reports how many AprilTag markers OpenCV's aruco detector finds — trying every
apriltag family so we can tell a family mismatch from a "detects nothing" problem.

    python3 src/calibration_multi_cam/scripts/detect_debug.py
    python3 src/calibration_multi_cam/scripts/detect_debug.py /camera0/image_raw
"""
import os
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

FAMILIES = {
    "36h11": "DICT_APRILTAG_36h11",
    "25h9": "DICT_APRILTAG_25h9",
    "16h5": "DICT_APRILTAG_16h5",
    # sanity: also try plain aruco in case the board is not apriltag at all
    "aruco_4x4": "DICT_4X4_50",
}


def build_detector(dict_name, refine):
    dict_id = getattr(cv2.aruco, dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = refine
    return cv2.aruco.ArucoDetector(dictionary, params)


def report(gray, tag):
    refine = getattr(cv2.aruco, "CORNER_REFINE_APRILTAG",
                     getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 0))
    print(f"  [{tag}] gray {gray.shape} dtype={gray.dtype} "
          f"min={gray.min()} max={gray.max()} mean={gray.mean():.1f}")
    for fam, dict_name in FAMILIES.items():
        if not hasattr(cv2.aruco, dict_name):
            print(f"    {fam:10s}: dictionary {dict_name} NOT in this OpenCV build")
            continue
        det = build_detector(dict_name, refine)
        corners, ids, _ = det.detectMarkers(gray)
        n = 0 if ids is None else len(ids)
        id_list = [] if ids is None else sorted(int(i) for i in ids.flatten())
        print(f"    {fam:10s}: {n} marker(s) ids={id_list[:20]}")


class Grab(Node):
    def __init__(self, topics):
        super().__init__("detect_debug")
        self.bridge = CvBridge()
        self.got = {}
        for t in topics:
            self.create_subscription(
                Image, t, lambda m, t=t: self._cb(m, t), qos_profile_sensor_data
            )
        self.topics = topics

    def _cb(self, msg, topic):
        if topic in self.got:
            return
        cam = topic.strip("/").split("/")[0]
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # save into the current working directory so it can be inspected
        path = os.path.join(os.getcwd(), f"detect_debug_{cam}.png")
        cv2.imwrite(path, bgr)
        print(f"\n{cam}  ({msg.width}x{msg.height}, encoding={msg.encoding}) "
              f"-> saved {path}")
        report(gray, cam)
        self.got[topic] = True


def main():
    topics = sys.argv[1:] or ["/camera0/image_raw", "/camera1/image_raw", "/camera2/image_raw"]
    rclpy.init()
    node = Grab(topics)
    print(f"OpenCV {cv2.__version__} — waiting for one frame on {topics} ...")
    while rclpy.ok() and len(node.got) < len(topics):
        rclpy.spin_once(node, timeout_sec=1.0)
    print("\ndone.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
