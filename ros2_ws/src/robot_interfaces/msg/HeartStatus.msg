# HeartStatus.msg - Status message for Heart power board
# Matches HeartStatusData struct

std_msgs/Header header

bool[12] status       # 12 status flags
bool[4] iso_status    # 4 isolation status flags
