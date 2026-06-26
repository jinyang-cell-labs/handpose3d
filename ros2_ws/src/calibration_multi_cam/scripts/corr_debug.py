#!/usr/bin/env python3
"""Find the correct per-tag corner ordering from a live frame.

For a near-frontal planar target, the correct ordering maps object points to
detected image points through a single homography with sub-pixel residual. We
brute-force the 8 dihedral orderings of each tag's 4 corners and report the
homography residual of each: the winner (~<1 px) is the mapping target.py should
use; a 70+ px "best" means object geometry, not ordering, is wrong.

    python3 src/calibration_multi_cam/scripts/corr_debug.py /camera1/image_raw
"""
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from calibration_multi_cam.target import AprilGridTarget

# Candidate orderings: each maps aruco corner slot (0..3) -> kalibr corner
# index within the tag's [BL, BR, TR, TL] tuple. 4 rotations x 2 reflections.
CANDIDATES = {
    "rot0        (0,1,2,3)": (0, 1, 2, 3),
    "rot90       (1,2,3,0)": (1, 2, 3, 0),
    "rot180      (2,3,0,1)": (2, 3, 0, 1),
    "rot270      (3,0,1,2)": (3, 0, 1, 2),
    "current     (3,2,1,0)": (3, 2, 1, 0),
    "flip+rot90  (2,1,0,3)": (2, 1, 0, 3),
    "flip+rot180 (1,0,3,2)": (1, 0, 3, 2),
    "flip+rot270 (0,3,2,1)": (0, 3, 2, 1),
}


class Probe(Node):
    def __init__(self, topic):
        super().__init__("corr_debug")
        self.bridge = CvBridge()
        self.done = False
        self.t = AprilGridTarget(tag_rows=6, tag_cols=6, tag_size=0.03,
                                 tag_spacing=0.333)
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.markerBorderBits = self.t.border_bits
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(self.dictionary, params)
        self.create_subscription(Image, topic, self._cb, qos_profile_sensor_data)

    def _residual(self, perm, quads, ids):
        obj, img = [], []
        for quad, tag_id in zip(quads, ids):
            if tag_id < 0 or tag_id >= self.t.num_tags:
                continue
            pidx = self.t.tag_point_indices(tag_id)  # [BL, BR, TR, TL]
            q = np.asarray(quad, np.float32).reshape(4, 2)
            for slot in range(4):
                obj.append(self.t.object_points[pidx[perm[slot]]][:2])
                img.append(q[slot])
        obj = np.asarray(obj, np.float64)
        img = np.asarray(img, np.float64)
        H, _ = cv2.findHomography(obj, img, 0)
        proj = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), H).reshape(-1, 2)
        return np.linalg.norm(proj - img, axis=1).mean()

    def _cb(self, msg):
        if self.done:
            return
        gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None or len(ids) < 8:
            self.get_logger().warn("need a full board in view")
            return
        self.done = True
        ids = ids.flatten().tolist()
        print(f"\n{len(ids)} tags detected. mean homography residual per ordering:")
        scored = sorted(((self._residual(p, corners, ids), name, p)
                         for name, p in CANDIDATES.items()))
        for r, name, p in scored:
            mark = "  <-- BEST" if (r, name, p) == scored[0] else ""
            print(f"  {name}: {r:8.2f} px{mark}")


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "/camera1/image_raw"
    rclpy.init()
    node = Probe(topic)
    print(f"waiting for a full-board frame on {topic} ...")
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
