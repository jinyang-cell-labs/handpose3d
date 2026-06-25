"""Multi-camera intrinsic + extrinsic calibration for ROS 2.

Pipeline (reimplementation of ethz-asl/kalibr's kalibr_calibrate_cameras,
Path B: same algorithm on a modern OpenCV + scipy stack):

    collect  -> AprilGrid detection on synchronized multi-camera frames
    solve    -> per-camera intrinsics (OpenCV) + rig extrinsics (pairwise PnP
                + covisibility-graph chaining) + global bundle adjustment (scipy)
    publish  -> intrinsics-only sensor_msgs/CameraInfo + extrinsics over TF/Pose

The world frame is aligned with the first camera (cameras[0]).
"""
