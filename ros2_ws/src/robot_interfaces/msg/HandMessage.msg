# Hand message - 25 poses for each joint in order (WRIST=0, THUMB_METACARPAL=1, ..., PINKY_FINGER_TIP=24)
# Based on WebXR XRHand API: https://developer.mozilla.org/en-US/docs/Web/API/XRHand
# Always contains exactly 25 poses, even if some joints are not tracked
# Joints that are not tracked should have NaN values in position/orientation
geometry_msgs/Pose[] joints
