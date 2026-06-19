# All hands detected in one image frame.

std_msgs/Header header     # stamp + frame_id copied from the source image
string source_topic        # input image topic these detections came from
Hand[] hands               # one entry per detected hand (may be empty)
