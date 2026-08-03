// hand_gesture_mapping_node - finger curl / thumb gesture extraction and
// direct actuator mapping.
//
// Subscribes to ONE camera's HandLandmarks stream (which camera is picked in
// body_cam_teleop.yaml via camera_namespace) and, per detected hand, reduces the
// 21 MediaPipe world landmarks to 2 normalized gesture values published as a
// std_msgs/Float32MultiArray:
//
//   [finger_curl, thumb_rotate]
//
// Each joint's "curl" is 180 deg minus the interior angle at that joint (two
// adjacent skeleton segments), so 0 = segments collinear (straight) and 180 =
// fully folded back. A finger's curl is the average over its three joints
// (MCP/PIP/DIP), the thumb's over its two distal joints (MCP/IP); thumb
// rotate is the single CMC joint (segments 0-1 / 1-2). finger_curl is the
// average of the index/middle/ring/pinky curls, and thumb_rotate the average
// of the thumb curl and thumb rotate values. Values are normalized by
// 180 deg into [0, 1].
//
// Unlike hands_node (boolean open/close + internal min-jerk interpolation),
// this node maps the measured gesture values straight into actuator space:
// per hand and per mechanism the vision value is first run through a one-pole
// temporal low-pass (gesture_filter_k, resets to the raw measurement when the
// hand was unseen for gesture_filter_timeout_sec so re-detection snaps) and
// clamped to the vision-side range (gesture_reprojection.*.vision_min/max,
// defaulting to the [open_vision, close_vision] envelope); then a linear
// reprojection defined by two calibration points
// (gesture_reprojection.{left,right}.{open,close}_{vision,actuator})
// converts it to an actuator position, which is finally clamped to the
// joint's [endstop_position_min, endstop_position_max] from the robot
// model's actuator_config.yaml. The gesture debug topics publish the RAW
// (unfiltered, unclamped) values — use them to calibrate the vision points. Per-joint kp/kd come from the model's
// controller_config.yaml (stiffness/damping maps), resolved the same way
// hands_node does it: robot_config.yaml `robot_model` -> humanoid_model share
// dir; joints missing from those maps fall back to the hand_stiffness /
// hand_damping parameters.
//
// One ActuatorCommand (finger_curl/thumb_rotate x left/right) is published
// per received landmarks frame — landmarks arrive at camera rate even with no
// hands in view, so the command stays inside the controller's freshness
// window while the camera runs; each hand holds its last commanded pose
// (open pose before its first detection). Nothing is published until the
// first hand is detected. Left and right gesture arrays still publish on
// separate topics; a hand absent from a message (or below min_score) simply
// publishes no gesture that frame.
//
// This node REPLACES hands_node as the controller's "hands" subsystem (don't
// run both — they publish the same topic): it serves the same hands/start /
// hands/stop / hands/status Trigger services the controller dispatches to
// when /controller/subsystem_start_stop is called with name "hands", and the
// actuator command is only published while started. Gesture extraction and
// the debug gesture topics keep running while stopped.

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <yaml-cpp/yaml.h>

#include <handpose3d_msgs/msg/hand.hpp>
#include <handpose3d_msgs/msg/hand_landmarks.hpp>
#include <robot_interfaces/msg/actuator_command.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace
{

constexpr size_t kNumLandmarks = 21;

// One joint = interior angle at landmark `joint` between the segments to its
// two skeleton neighbours `prev` and `next`.
struct Joint
{
  int prev, joint, next;
};

// MediaPipe hand model indices: wrist=0, thumb=1-4, index=5-8, middle=9-12,
// ring=13-16, pinky=17-20.
constexpr std::array<std::array<Joint, 3>, 4> kFingerJoints = {{
  {{{0, 5, 6}, {5, 6, 7}, {6, 7, 8}}},        // index
  {{{0, 9, 10}, {9, 10, 11}, {10, 11, 12}}},  // middle
  {{{0, 13, 14}, {13, 14, 15}, {14, 15, 16}}},// ring
  {{{0, 17, 18}, {17, 18, 19}, {18, 19, 20}}},// pinky
}};
constexpr std::array<Joint, 2> kThumbCurlJoints = {{{1, 2, 3}, {2, 3, 4}}};
constexpr Joint kThumbRotateJoint = {0, 1, 2};

// Hand actuator mechanisms driven by this node, in gesture-vector order.
constexpr std::array<const char *, 2> kMechanisms = {"finger_curl", "thumb_rotate"};
// Hand order for state/actuator indexing (matches hands_node).
constexpr std::array<const char *, 2> kHands = {"left", "right"};
constexpr size_t kNumMechanisms = kMechanisms.size();
constexpr size_t kNumHands = kHands.size();

constexpr char kDefaultRobotConfigPath[] = "/workspace/robot/ros2_ws/config/robot_config.yaml";

// Normalized curl in [0, 1] at one joint: (180 deg - interior angle) / 180.
// Returns -1 if a segment is degenerate (coincident landmarks).
double joint_curl(const std::vector<geometry_msgs::msg::Point> & lm, const Joint & j)
{
  const auto & p = lm[j.prev];
  const auto & c = lm[j.joint];
  const auto & n = lm[j.next];
  const double ax = p.x - c.x, ay = p.y - c.y, az = p.z - c.z;
  const double bx = n.x - c.x, by = n.y - c.y, bz = n.z - c.z;
  const double na = std::sqrt(ax * ax + ay * ay + az * az);
  const double nb = std::sqrt(bx * bx + by * by + bz * bz);
  if (na < 1e-9 || nb < 1e-9) {
    return -1.0;
  }
  const double cosang =
    std::clamp((ax * bx + ay * by + az * bz) / (na * nb), -1.0, 1.0);
  return (M_PI - std::acos(cosang)) / M_PI;
}

// Average curl over a set of joints, ignoring degenerate ones.
template<typename Joints>
float average_curl(const std::vector<geometry_msgs::msg::Point> & lm, const Joints & joints)
{
  double sum = 0.0;
  int n = 0;
  for (const auto & j : joints) {
    const double c = joint_curl(lm, j);
    if (c >= 0.0) {
      sum += c;
      ++n;
    }
  }
  return n > 0 ? static_cast<float>(sum / n) : 0.0f;
}

// Linear reprojection through two calibration points (open/close), clamped to
// the joint's endstop range.
struct LinearMap
{
  double vision_open, vision_close, actuator_open, actuator_close, pos_min, pos_max;

  double operator()(double vision) const
  {
    const double s = (vision - vision_open) / (vision_close - vision_open);
    return std::clamp(
      actuator_open + s * (actuator_close - actuator_open), pos_min, pos_max);
  }
};

// Resolve robot_model from robot_config.yaml -> humanoid_model model dir.
// Throws on any missing file/key — a silently wrong gain/limit set is worse
// than a loud boot failure (same policy as hands_node).
std::string resolve_model_dir(const std::string & robot_config_path)
{
  const YAML::Node cfg = YAML::LoadFile(robot_config_path);
  if (!cfg["robot_model"]) {
    throw std::runtime_error(robot_config_path + " has no robot_model key");
  }
  return ament_index_cpp::get_package_share_directory("humanoid_model") +
         "/models/" + cfg["robot_model"].as<std::string>();
}

}  // namespace

class HandGestureMappingNode : public rclcpp::Node
{
public:
  HandGestureMappingNode()
  : Node("hand_gesture_mapping_node")
  {
    const auto camera_ns = declare_parameter<std::string>("camera_namespace", "cam0");
    const auto landmarks_topic =
      declare_parameter<std::string>("landmarks_topic", "body_cam_teleop/landmarks");
    const auto left_topic =
      declare_parameter<std::string>("left_output_topic", "hand_gestures/left");
    const auto right_topic =
      declare_parameter<std::string>("right_output_topic", "hand_gestures/right");
    min_score_ = declare_parameter<double>("min_score", 0.5);

    const auto actuator_topic =
      declare_parameter<std::string>("actuator_command_topic", "cmd/hands/actuator");
    const auto robot_config_path =
      declare_parameter<std::string>("robot_config_path", kDefaultRobotConfigPath);
    // Fallback gains for joints absent from the model's stiffness/damping maps.
    const double fallback_kp = declare_parameter<double>("hand_stiffness", 0.2);
    const double fallback_kd = declare_parameter<double>("hand_damping", 0.001);

    // One-pole low-pass on the vision values feeding the actuator path,
    // gain in (0, 1]: filt += k * (raw - filt) per frame; 1.0 disables.
    filter_k_ = declare_parameter<double>("gesture_filter_k", 0.3);
    if (filter_k_ <= 0.0 || filter_k_ > 1.0) {
      throw std::runtime_error("gesture_filter_k must be in (0, 1]");
    }
    // A hand unseen for longer than this snaps the filter to the next
    // measurement instead of dragging from the stale state.
    filter_timeout_sec_ = declare_parameter<double>("gesture_filter_timeout_sec", 0.5);

    // Two calibration points per hand, each a [finger_curl, thumb_rotate]
    // pair in vision space and in actuator space.
    auto vec2 = [this](const std::string & name, std::vector<double> def) {
        const auto v = declare_parameter<std::vector<double>>(name, std::move(def));
        if (v.size() != kNumMechanisms) {
          throw std::runtime_error(name + " must have 2 entries [finger_curl, thumb_rotate]");
        }
        return std::array<double, 2>{v[0], v[1]};
      };

    const std::string model_dir = resolve_model_dir(robot_config_path);
    const YAML::Node ctrl_cfg = YAML::LoadFile(model_dir + "/controller_config.yaml");
    const YAML::Node act_cfg = YAML::LoadFile(model_dir + "/actuator_config.yaml");

    std::string gains_log;
    for (size_t h = 0; h < kNumHands; ++h) {
      const std::string base = std::string("gesture_reprojection.") + kHands[h] + ".";
      const auto open_vision = vec2(base + "open_vision", {0.1, 0.08});
      const auto open_actuator = vec2(base + "open_actuator", {-0.3491, 0.0});
      const auto close_vision = vec2(base + "close_vision", {0.3, 0.16});
      const auto close_actuator = vec2(base + "close_actuator", {1.0, -0.9});

      // Vision-side clamp applied after the temporal filter, before the
      // reprojection; defaults to the open/close_vision envelope.
      const auto vision_min = vec2(
        base + "vision_min",
        {std::min(open_vision[0], close_vision[0]), std::min(open_vision[1], close_vision[1])});
      const auto vision_max = vec2(
        base + "vision_max",
        {std::max(open_vision[0], close_vision[0]), std::max(open_vision[1], close_vision[1])});
      for (size_t m = 0; m < kNumMechanisms; ++m) {
        if (vision_min[m] > vision_max[m]) {
          throw std::runtime_error(base + "vision_min exceeds vision_max");
        }
        vision_min_[h][m] = vision_min[m];
        vision_max_[h][m] = vision_max[m];
      }

      for (size_t m = 0; m < kNumMechanisms; ++m) {
        const std::string joint = std::string(kMechanisms[m]) + "_" + kHands[h];
        joint_names_[h * kNumMechanisms + m] = joint;

        if (std::abs(close_vision[m] - open_vision[m]) < 1e-9) {
          throw std::runtime_error(
                  base + "*: open_vision and close_vision coincide for " + joint +
                  " — linear reprojection is undefined");
        }

        const YAML::Node act_joint = act_cfg[joint];
        if (!act_joint || !act_joint["endstop_position_min"] ||
          !act_joint["endstop_position_max"])
        {
          throw std::runtime_error(
                  model_dir + "/actuator_config.yaml has no endstop_position_min/max for " +
                  joint);
        }
        const double pos_min = act_joint["endstop_position_min"].as<double>();
        const double pos_max = act_joint["endstop_position_max"].as<double>();

        maps_[h][m] = LinearMap{
          open_vision[m], close_vision[m], open_actuator[m], close_actuator[m],
          pos_min, pos_max};
        // Hold the open pose until this hand is first seen.
        cmd_[h][m] = std::clamp(open_actuator[m], pos_min, pos_max);

        const YAML::Node kp_node = ctrl_cfg["stiffness"][joint];
        const YAML::Node kd_node = ctrl_cfg["damping"][joint];
        if (!kp_node || !kd_node) {
          RCLCPP_WARN(
            get_logger(), "model config %s has no gains for %s; falling back to "
            "hand_stiffness=%.3f, hand_damping=%.3f",
            model_dir.c_str(), joint.c_str(), fallback_kp, fallback_kd);
        }
        kp_[h * kNumMechanisms + m] = kp_node ? kp_node.as<double>() : fallback_kp;
        kd_[h * kNumMechanisms + m] = kd_node ? kd_node.as<double>() : fallback_kd;

        gains_log += joint + ": kp=" + std::to_string(kp_[h * kNumMechanisms + m]) +
          " kd=" + std::to_string(kd_[h * kNumMechanisms + m]) +
          " range=[" + std::to_string(pos_min) + ", " + std::to_string(pos_max) + "] ";
      }
    }

    const std::string full_topic = "/" + camera_ns + "/" + landmarks_topic;

    left_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(left_topic, rclcpp::QoS(5));
    right_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(right_topic, rclcpp::QoS(5));
    // Default (RELIABLE, VOLATILE, KEEP_LAST 10) — matches the controller's
    // add_source subscription (same as hands_node).
    actuator_pub_ =
      create_publisher<robot_interfaces::msg::ActuatorCommand>(actuator_topic, rclcpp::QoS(10));
    // Matches hand_landmarks_node's reliable depth-5 publisher.
    landmarks_sub_ = create_subscription<handpose3d_msgs::msg::HandLandmarks>(
      full_topic, rclcpp::QoS(5),
      [this](const handpose3d_msgs::msg::HandLandmarks::SharedPtr msg) {on_landmarks(*msg);});

    // Subsystem dispatch contract (same as hands_node): the controller calls
    // /hands/start and /hands/stop when /controller/subsystem_start_stop is
    // invoked with name "hands", and only forwards cmd/hands/actuator while
    // the subsystem is active. Relative names — this node runs in the root
    // namespace, so they resolve to /hands/*.
    using Trigger = std_srvs::srv::Trigger;
    start_srv_ = create_service<Trigger>(
      "hands/start",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr res) {
        running_ = true;
        res->success = true;
        res->message = "hands started";
        RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
      });
    stop_srv_ = create_service<Trigger>(
      "hands/stop",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr res) {
        running_ = false;
        res->success = true;
        res->message = "hands stopped";
        RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
      });
    status_srv_ = create_service<Trigger>(
      "hands/status",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr res) {
        res->success = true;
        res->message = std::string("running=") + (running_ ? "true" : "false") +
        ", hand_seen=" + (hand_seen_ ? "true" : "false");
      });

    RCLCPP_INFO(
      get_logger(),
      "listening on %s -> left: %s, right: %s, actuator: %s; gains/limits from %s (%s)",
      full_topic.c_str(), left_topic.c_str(), right_topic.c_str(), actuator_topic.c_str(),
      model_dir.c_str(), gains_log.c_str());
  }

private:
  void on_landmarks(const handpose3d_msgs::msg::HandLandmarks & msg)
  {
    for (const auto & hand : msg.hands) {
      if (hand.score < min_score_ || hand.landmarks_world.size() != kNumLandmarks) {
        continue;
      }
      const size_t h = hand.handedness == "Left" ? 0 : 1;

      float fingers_sum = 0.0f;
      for (const auto & finger : kFingerJoints) {
        fingers_sum += average_curl(hand.landmarks_world, finger);
      }
      const float thumb_curl = average_curl(hand.landmarks_world, kThumbCurlJoints);
      const float thumb_rot =
        average_curl(hand.landmarks_world, std::array<Joint, 1>{kThumbRotateJoint});

      const std::array<float, 2> gestures = {
        fingers_sum / kFingerJoints.size(), (thumb_curl + thumb_rot) / 2.0f};

      std_msgs::msg::Float32MultiArray out;
      out.layout.dim.resize(1);
      out.layout.dim[0].label = "finger_curl,thumb_rotate";
      out.layout.dim[0].size = kNumMechanisms;
      out.layout.dim[0].stride = kNumMechanisms;
      out.data.assign(gestures.begin(), gestures.end());
      (h == 0 ? left_pub_ : right_pub_)->publish(out);

      // Actuator path: temporal filter -> vision clamp -> reprojection.
      // The filter snaps to the measurement on the first sighting or after a
      // detection gap (stale state must not drag a re-detected hand).
      const rclcpp::Time t = now();
      const bool reset = last_gesture_time_[h].nanoseconds() == 0 ||
        (t - last_gesture_time_[h]).seconds() > filter_timeout_sec_;
      last_gesture_time_[h] = t;
      for (size_t m = 0; m < kNumMechanisms; ++m) {
        filt_[h][m] = reset ? gestures[m] :
          filt_[h][m] + filter_k_ * (gestures[m] - filt_[h][m]);
        cmd_[h][m] = maps_[h][m](
          std::clamp(filt_[h][m], vision_min_[h][m], vision_max_[h][m]));
      }
      hand_seen_ = true;
    }

    // Gated by hands/start (like hands_node: when stopped, publish nothing so
    // the controller damps the hands). Landmarks arrive at camera rate even
    // with no hands in view, so while running this keeps the command fresh
    // (hands hold their last pose). Also silent until the first detection so
    // an idle camera moves nothing.
    if (!running_ || !hand_seen_) {
      return;
    }
    robot_interfaces::msg::ActuatorCommand cmd;
    cmd.header.stamp = now();
    cmd.name.assign(joint_names_.begin(), joint_names_.end());
    for (size_t h = 0; h < kNumHands; ++h) {
      for (size_t m = 0; m < kNumMechanisms; ++m) {
        cmd.position.push_back(cmd_[h][m]);
      }
    }
    cmd.velocity.assign(joint_names_.size(), 0.0);
    cmd.effort.assign(joint_names_.size(), 0.0);
    cmd.stiffness.assign(kp_.begin(), kp_.end());
    cmd.damping.assign(kd_.begin(), kd_.end());
    actuator_pub_->publish(cmd);
  }

  double min_score_{0.5};
  // Indexed [hand][mechanism] / [hand * kNumMechanisms + mechanism], hand
  // order = kHands, mechanism order = kMechanisms.
  std::array<std::array<LinearMap, kNumMechanisms>, kNumHands> maps_{};
  std::array<std::array<double, kNumMechanisms>, kNumHands> cmd_{};
  // Vision-side clamp bounds and one-pole filter state for the actuator path.
  std::array<std::array<double, kNumMechanisms>, kNumHands> vision_min_{}, vision_max_{};
  std::array<std::array<double, kNumMechanisms>, kNumHands> filt_{};
  std::array<rclcpp::Time, kNumHands> last_gesture_time_;
  double filter_k_{0.3};
  double filter_timeout_sec_{0.5};
  std::array<std::string, kNumHands * kNumMechanisms> joint_names_;
  std::array<double, kNumHands * kNumMechanisms> kp_{}, kd_{};
  bool hand_seen_{false};
  bool running_{false};  // toggled by hands/start, hands/stop

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr left_pub_, right_pub_;
  rclcpp::Publisher<robot_interfaces::msg::ActuatorCommand>::SharedPtr actuator_pub_;
  rclcpp::Subscription<handpose3d_msgs::msg::HandLandmarks>::SharedPtr landmarks_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_srv_, stop_srv_, status_srv_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HandGestureMappingNode>());
  rclcpp::shutdown();
  return 0;
}
