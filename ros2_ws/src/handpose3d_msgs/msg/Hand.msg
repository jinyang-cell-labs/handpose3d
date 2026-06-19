# One hand detected by MediaPipe HandLandmarker.

string handedness                      # "Left" or "Right"
float32 score                          # per-hand handedness/detection confidence in [0, 1].
                                       # The Tasks API exposes no per-landmark
                                       # confidence (visibility/presence are
                                       # always 0), so this is the only one.

# 21 image-space landmarks (MediaPipe hand_landmarks): x, y in pixels, z is the
# model's relative depth (origin ~ wrist, roughly the same scale as normalized x;
# NOT metric). Index order follows MediaPipe's 21-landmark hand model.
geometry_msgs/Point[] landmarks_image

# 21 world-space landmarks (MediaPipe hand_world_landmarks): x, y, z in metres,
# in a hand-local frame (origin ~ the hand's geometric center). Single-view
# estimate, no camera calibration. Empty when 3D estimation is disabled.
geometry_msgs/Point[] landmarks_world
