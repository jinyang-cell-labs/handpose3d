cmake_minimum_required(VERSION 3.8)
project(robot_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(msg_files
  "msg/ActuatorCommand.msg"
  "msg/ActuatorConfig.msg"
  "msg/ActuatorFeedback.msg"
  "msg/JointCommand.msg"
  "msg/JointFeedback.msg"
  "msg/HeartStatus.msg"
  "msg/HeartMeasurement.msg"
  "msg/ControllerStatus.msg"
  "msg/ControllerAnalytics.msg"
  "msg/PolicyAnalytics.msg"
  "msg/SpineAnalytics.msg"
  "msg/ArmManipulationAnalytics.msg"
  "msg/TeleopMessage.msg"
  "msg/ControllerMessage.msg"
  "msg/Emergency.msg"
  "msg/HandMessage.msg"
  "msg/GripperCommand.msg"
  "msg/SpineImu.msg"
  "msg/VelocityConfig.msg"
)

set(srv_files
  "srv/ActionScaleGet.srv"
  "srv/ActionScaleSet.srv"
  "srv/ActuatorConfigGet.srv"
  "srv/ActuatorConfigSet.srv"
  "srv/ActuatorElectricalAlignment.srv"
  "srv/ActuatorFirmwareVersionGet.srv"
  "srv/ActuatorGainsGet.srv"
  "srv/ActuatorGainsSet.srv"
  "srv/ActuatorProvisioning.srv"
  "srv/ActuatorReset.srv"
  "srv/ActuatorStartStop.srv"
  "srv/ActuatorTareAngle.srv"
  "srv/ActuatorZeroing.srv"
  "srv/FirmwareUpload.srv"
  "srv/ControllerModeGet.srv"
  "srv/ControllerModeSet.srv"
  "srv/ControllerTypeGet.srv"
  "srv/ControllerTypeList.srv"
  "srv/ControllerTypeSet.srv"
  "srv/ControllerSubsystem.srv"
  "srv/HeartEnable.srv"
  "srv/SmoothingCutoffGet.srv"
  "srv/SmoothingCutoffSet.srv"
  "srv/UploadMap.srv"
  "srv/VelocityConfigGet.srv"
  "srv/VelocityConfigSet.srv"
)

set(action_files
  "action/EndstopAlignment.action"
  "action/FirmwareFlash.action"
  "action/GraspObjectAction.action"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
