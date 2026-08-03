cmake_minimum_required(VERSION 3.16)
project(body_cam_teleop)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(handpose3d_msgs REQUIRED)
find_package(robot_interfaces REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core calib3d imgproc)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(hand_pose_node src/hand_pose_node.cpp)
target_compile_features(hand_pose_node PUBLIC cxx_std_17)
target_link_libraries(hand_pose_node
  Ceres::ceres
  Eigen3::Eigen
  yaml-cpp
  ${OpenCV_LIBS}
)
ament_target_dependencies(hand_pose_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2_ros
  cv_bridge
  handpose3d_msgs
  robot_interfaces
)

add_executable(hand_gesture_mapping_node src/hand_gesture_mapping_node.cpp)
target_compile_features(hand_gesture_mapping_node PUBLIC cxx_std_17)
target_link_libraries(hand_gesture_mapping_node yaml-cpp)
ament_target_dependencies(hand_gesture_mapping_node
  rclcpp
  ament_index_cpp
  std_msgs
  std_srvs
  geometry_msgs
  handpose3d_msgs
  robot_interfaces
)

install(TARGETS hand_pose_node hand_gesture_mapping_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS
  scripts/hand_label_tracker.py
  scripts/hand_landmarks_node.py
  scripts/mediapipe_detection_debug_node.py
  scripts/teleop_mux_node.py
  scripts/hand_scale_calib_node.py
  scripts/perf_monitor_node.py
  scripts/perf_report.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch models DESTINATION share/${PROJECT_NAME})

ament_package()
