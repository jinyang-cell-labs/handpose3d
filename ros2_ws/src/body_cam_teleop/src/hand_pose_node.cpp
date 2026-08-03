// hand_pose_node: monocular hand-pose estimation + teleop bridge (C++).
//
// Consumes handpose3d_msgs/HandLandmarks from hand_landmarks_node (MediaPipe:
// 21 image pixels + hand-local metric model per hand), and per detected hand:
//
//   1. undistorts the 21 detected pixels (points only — the full frame is
//      never remapped),
//   2. scales the hand-local model by hand_size_scaling_factor,
//   3. solves the rigid 6-DoF T_cam_hand with Ceres (reprojection cost + a
//      one-sided cheirality penalty that breaks the front/back mirror),
//      warm-started from the previous frame's pose,
//   4. rebuilds the pose as an anatomical palm frame (origin=wrist, y=palm
//      normal, z=finger->wrist bisector, x=y×z),
//   5. transforms it into the operator_body frame: the camera is mounted at
//      the operator body center, so camera -> operator_body is just the fixed
//      operator_body_position/rotation offset from the parameters (identity
//      by default — no board detection, no extrinsics),
//   6. publishes robot_interfaces/TeleopMessage on /teleop_converted at
//      publish_hz, holding the trigger button while the hand pose is fresh.
//
// Diagnostics: this node owns the last three gates a detection must pass to
// reach the arm controller — the min_score handedness gate, the Ceres solve,
// and the pose_timeout_sec freshness gate that holds the trigger button. Each
// has its own opt-in log flag (log_input, log_score_gate, log_solve, log_pose,
// log_trigger) and every reject line names the gate, the measured value and the
// threshold that rejected it. log_gate_summary adds a periodic funnel count,
// and log_reproj_warn_px warns when a solve converges to a fit that does not
// actually match the detected landmarks. All flags are runtime-settable
// (`ros2 param set <node> log_solve true`).
//
// With enable_reprojection=true it additionally subscribes to the camera
// image, reprojects the placed 3D joints back onto it (full distortion model,
// raw image), publishes the overlay, and publishes RViz skeleton markers +
// per-hand TF in the operator_body frame. Debug TF/marker frames get the node
// namespace as a prefix so parallel instances don't fight over the same TF
// names.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <handpose3d_msgs/msg/hand_landmarks.hpp>
#include <robot_interfaces/msg/teleop_message.hpp>

namespace
{

constexpr int kNumLandmarks = 21;
constexpr int kNumHandJoints = 25;  // robot_interfaces/HandMessage contract
constexpr int kWristIdx = 0;
constexpr int kIndexMcpIdx = 5;
constexpr int kPinkyMcpIdx = 17;

const std::array<std::pair<int, int>, 21> kHandConnections = {{
    {0, 1},  {1, 2},   {2, 3},   {3, 4},    // thumb
    {0, 5},  {5, 6},   {6, 7},   {7, 8},    // index
    {5, 9},  {9, 10},  {10, 11}, {11, 12},  // middle
    {9, 13}, {13, 14}, {14, 15}, {15, 16},  // ring
    {13, 17}, {17, 18}, {18, 19}, {19, 20},  // pinky
    {0, 17},                                 // palm base
}};

// Intrinsic XYZ euler (degrees) -> rotation matrix; matches
// calibration_multi_cam se3.euler_deg_to_R (R = Rx * Ry * Rz).
Eigen::Matrix3d eulerDegToR(double rx, double ry, double rz)
{
  const double ax = rx * M_PI / 180.0, ay = ry * M_PI / 180.0, az = rz * M_PI / 180.0;
  return (Eigen::AngleAxisd(ax, Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(ay, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(az, Eigen::Vector3d::UnitZ()))
    .toRotationMatrix();
}

geometry_msgs::msg::Pose matrixToPose(const Eigen::Matrix4d & T)
{
  geometry_msgs::msg::Pose p;
  const Eigen::Quaterniond q(Eigen::Matrix3d(T.block<3, 3>(0, 0)));
  p.position.x = T(0, 3);
  p.position.y = T(1, 3);
  p.position.z = T(2, 3);
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  p.orientation.w = q.w();
  return p;
}

geometry_msgs::msg::TransformStamped matrixToTf(
  const Eigen::Matrix4d & T, const std::string & parent, const std::string & child,
  const builtin_interfaces::msg::Time & stamp)
{
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = stamp;
  tf.header.frame_id = parent;
  tf.child_frame_id = child;
  const Eigen::Quaterniond q(Eigen::Matrix3d(T.block<3, 3>(0, 0)));
  tf.transform.translation.x = T(0, 3);
  tf.transform.translation.y = T(1, 3);
  tf.transform.translation.z = T(2, 3);
  tf.transform.rotation.x = q.x();
  tf.transform.rotation.y = q.y();
  tf.transform.rotation.z = q.z();
  tf.transform.rotation.w = q.w();
  return tf;
}

// Reprojection (2 residuals) + one-sided cheirality penalty (1 residual) for
// one joint. Parameters: [angle-axis (3), translation (3)] = T_cam_hand.
struct JointResidual
{
  JointResidual(
    const Eigen::Vector3d & x_hand, const Eigen::Vector2d & uv, double fx, double fy, double cx,
    double cy, double cheir_margin, double cheir_weight)
  : x_hand_(x_hand), uv_(uv), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
    cheir_margin_(cheir_margin), cheir_weight_(cheir_weight)
  {
  }

  template <typename T>
  bool operator()(const T * pose, T * residuals) const
  {
    const T pt[3] = {T(x_hand_.x()), T(x_hand_.y()), T(x_hand_.z())};
    T pc[3];
    ceres::AngleAxisRotatePoint(pose, pt, pc);
    pc[0] += pose[3];
    pc[1] += pose[4];
    pc[2] += pose[5];
    residuals[0] = T(fx_) * pc[0] / pc[2] + T(cx_) - T(uv_.x());
    residuals[1] = T(fy_) * pc[1] / pc[2] + T(cy_) - T(uv_.y());
    // relu(margin - z): zero once the joint is in front of the camera.
    const T viol = T(cheir_margin_) - pc[2];
    residuals[2] = viol > T(0.0) ? T(cheir_weight_) * viol : T(0.0);
    return true;
  }

  Eigen::Vector3d x_hand_;
  Eigen::Vector2d uv_;
  double fx_, fy_, cx_, cy_, cheir_margin_, cheir_weight_;
};

struct SolveResult
{
  bool success{false};
  Eigen::Matrix4d T_cam_hand{Eigen::Matrix4d::Identity()};
  double reproj_rms_px{0.0};
  int num_iterations{0};
  // Diagnostics (log_solve), filled on both the success and the failure path.
  int num_residual_blocks{0};   // joints that entered the problem (of 21)
  int num_skipped{0};           // joints dropped as non-finite
  double initial_cost{0.0};
  double final_cost{0.0};
  bool warm_started{false};
  double seed_depth_m{0.0};     // depth of the seed the solve started from
  std::string termination;      // Ceres termination, or the pre-solve reject
};

// Per-stage perf accumulator (enable_perf). All writers run on the default
// single-threaded executor (callbacks + perf timer are serialized), so no
// locking. Values are usually wall milliseconds; the stage name carries the
// unit (_ms / _px / plain count).
struct StageStats
{
  int64_t n{0};
  double sum{0.0};
  double max{0.0};
  void add(double v)
  {
    ++n;
    sum += v;
    max = std::max(max, v);
  }
};

using PerfClock = std::chrono::steady_clock;

double msSince(const PerfClock::time_point & t0)
{
  return std::chrono::duration<double, std::milli>(PerfClock::now() - t0).count();
}

// Project 3D points with the full distortion model, dispatching on the
// camera model: pinhole-radtan ([k1,k2,p1,p2,k3]) or pinhole-equi
// (fisheye/equidistant, [k1,k2,k3,k4]).
void projectPointsModel(
  const std::vector<cv::Point3d> & objp, const cv::Vec3d & rvec, const cv::Vec3d & tvec,
  const cv::Mat & K, const cv::Mat & dist, bool fisheye_model, std::vector<cv::Point2d> & proj)
{
  if (fisheye_model) {
    cv::fisheye::projectPoints(objp, proj, rvec, tvec, K, dist);
  } else {
    cv::projectPoints(objp, rvec, tvec, K, dist, proj);
  }
}

}  // namespace

class HandPoseNode : public rclcpp::Node
{
public:
  HandPoseNode()
  : Node("hand_pose_node")
  {
    // ---- parameters ------------------------------------------------------
    camera_name_ = declare_parameter<std::string>("camera_name", "camera0");
    // Explicit *_file params win; empty ones default into calibration_dir
    // (injected by the launch file as this package's share/config directory).
    const auto config_dir = declare_parameter<std::string>("calibration_dir", "");
    intrinsics_file_ = declare_parameter<std::string>("intrinsics_file", "");
    if (intrinsics_file_.empty()) {intrinsics_file_ = config_dir + "/intrinsics.yaml";}
    body_frame_ = declare_parameter<std::string>("body_frame", "operator_body");
    // Camera -> operator_body rigid offset: the camera is mounted at the
    // operator body center, so this is identity by default; the rotation is
    // there to re-align the camera optical axes with the body convention if
    // needed. Position [m] in the camera frame, intrinsic XYZ euler [deg].
    operator_body_position_ = declare_parameter<std::vector<double>>(
      "operator_body_position", {0.0, 0.0, 0.0});
    operator_body_rotation_ = declare_parameter<std::vector<double>>(
      "operator_body_rotation", {0.0, 0.0, 0.0});

    landmarks_topic_ =
      declare_parameter<std::string>("landmarks_topic", "body_cam_teleop/landmarks");
    output_topic_ = declare_parameter<std::string>("output_topic", "/teleop_converted");
    publish_hz_ = declare_parameter<double>("publish_hz", 50.0);
    trigger_button_index_ = declare_parameter<int>("trigger_button_index", 5);
    num_joy_buttons_ = declare_parameter<int>("num_joy_buttons", 16);
    pose_timeout_sec_ = declare_parameter<double>("pose_timeout_sec", 0.5);
    min_score_ = declare_parameter<double>("min_score", 0.5);
    hand_size_scaling_factor_ = declare_parameter<double>("hand_size_scaling_factor", 1.3);
    cheirality_margin_ = declare_parameter<double>("cheirality_margin", 0.05);
    cheirality_weight_ = declare_parameter<double>("cheirality_weight", 1000.0);
    seed_depth_m_ = declare_parameter<double>("seed_depth_m", 0.5);
    warm_start_ = declare_parameter<bool>("warm_start", true);
    max_solver_iterations_ = declare_parameter<int>("max_solver_iterations", 50);
    anatomical_hand_frame_ = declare_parameter<bool>("anatomical_hand_frame", true);
    ray_filter_k_ = declare_parameter<double>("ray_filter_k", 1.0);
    lateral_filter_k_ = declare_parameter<double>("lateral_filter_k", 1.0);
    filter_ref_dist_m_ = declare_parameter<double>("filter_ref_dist_m", 0.0);
    filter_min_scale_ = declare_parameter<double>("filter_min_scale", 0.2);
    const auto offset_rpy = declare_parameter<std::vector<double>>(
      "hand_orientation_offset_rpy", {0.0, 0.0, 0.0});

    enable_reprojection_ = declare_parameter<bool>("enable_reprojection", false);
    image_topic_ = declare_parameter<std::string>("image_topic", "body_cam_teleop/image_raw");
    reprojected_topic_ =
      declare_parameter<std::string>("reprojected_topic", "body_cam_teleop/image_reprojected");
    markers_topic_ = declare_parameter<std::string>("markers_topic", "body_cam_teleop/markers");
    image_buffer_size_ = declare_parameter<int>("image_buffer_size", 15);
    image_match_tol_sec_ = declare_parameter<double>("image_match_tol", 0.05);
    joint_size_ = declare_parameter<double>("joint_size", 0.012);
    line_width_ = declare_parameter<double>("line_width", 0.006);

    // ---- diagnostics (per-stage, opt-in, runtime-settable) ----------------
    // One flag per gate; see the header comment. Detail lines are throttled
    // per (stage, reason, hand) rather than per call site, so a Left reject
    // never masks a Right one.
    log_input_ = declare_parameter<bool>("log_input", false);
    log_score_gate_ = declare_parameter<bool>("log_score_gate", false);
    log_solve_ = declare_parameter<bool>("log_solve", false);
    log_pose_ = declare_parameter<bool>("log_pose", false);
    log_trigger_ = declare_parameter<bool>("log_trigger", false);
    log_gate_summary_ = declare_parameter<bool>("log_gate_summary", false);
    log_throttle_sec_ = declare_parameter<double>("log_throttle_sec", 2.0);
    log_summary_period_sec_ = declare_parameter<double>("log_summary_period_sec", 5.0);
    // Warn when a converged solve still misses the detected landmarks by more
    // than this [px]. Nothing is gated on it — the pose is still published.
    // 0 disables.
    log_reproj_warn_px_ = declare_parameter<double>("log_reproj_warn_px", 30.0);

    if (publish_hz_ <= 0.0) {
      throw std::invalid_argument("publish_hz must be > 0");
    }
    if (num_joy_buttons_ <= trigger_button_index_) {
      throw std::invalid_argument("num_joy_buttons must exceed trigger_button_index");
    }
    if (offset_rpy.size() != 3) {
      throw std::invalid_argument("hand_orientation_offset_rpy must be length 3");
    }
    if (ray_filter_k_ <= 0.0 || ray_filter_k_ > 1.0) {
      throw std::invalid_argument("ray_filter_k must be in (0, 1]");
    }
    if (lateral_filter_k_ <= 0.0 || lateral_filter_k_ > 1.0) {
      throw std::invalid_argument("lateral_filter_k must be in (0, 1]");
    }
    if (filter_min_scale_ <= 0.0 || filter_min_scale_ > 1.0) {
      throw std::invalid_argument("filter_min_scale must be in (0, 1]");
    }
    if (filter_ref_dist_m_ < 0.0) {
      throw std::invalid_argument("filter_ref_dist_m must be >= 0");
    }
    if (lateral_filter_k_ < ray_filter_k_) {
      RCLCPP_WARN(
        get_logger(),
        "lateral_filter_k (%.2f) < ray_filter_k (%.2f): lateral directions are "
        "better observed than depth and normally filtered less",
        lateral_filter_k_, ray_filter_k_);
    }
    pose_filter_enabled_ =
      ray_filter_k_ < 1.0 || lateral_filter_k_ < 1.0 || filter_ref_dist_m_ > 0.0;
    if (std::abs(offset_rpy[0]) > 1e-9 || std::abs(offset_rpy[1]) > 1e-9 ||
        std::abs(offset_rpy[2]) > 1e-9)
    {
      offset_quat_ = Eigen::Quaterniond(
        eulerDegToR(offset_rpy[0], offset_rpy[1], offset_rpy[2]));
    }
    loadCalibration();

    // Debug TF and marker frames carry the node namespace as a prefix (e.g.
    // cam0/operator_body) so per-camera instances publish disjoint TF trees;
    // the TeleopMessage header keeps the plain body_frame name.
    const std::string ns = get_namespace();
    tf_prefix_ = ns == "/" ? "" : ns.substr(1) + "/";
    tf_body_frame_ = tf_prefix_ + body_frame_;

    // ---- pubs / subs -----------------------------------------------------
    // /teleop_converted is best-effort KEEP_LAST(1): matches the arm
    // controller's sensor-data subscriber.
    teleop_pub_ = create_publisher<robot_interfaces::msg::TeleopMessage>(
      output_topic_, rclcpp::QoS(1).best_effort());

    landmarks_sub_ = create_subscription<handpose3d_msgs::msg::HandLandmarks>(
      landmarks_topic_, rclcpp::QoS(5),
      [this](handpose3d_msgs::msg::HandLandmarks::ConstSharedPtr msg) {
        onLandmarks(*msg);
      });

    // The reprojection overlay consumes the raw frame (hand_landmarks_node
    // publishes it when the flag is on).
    if (enable_reprojection_) {
      image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {onImage(msg);});
      reprojected_pub_ =
        create_publisher<sensor_msgs::msg::Image>(reprojected_topic_, rclcpp::SensorDataQoS());
      markers_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>(markers_topic_, rclcpp::QoS(5));
      hand_tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    // Static TF (operator_body -> camera, the fixed mount offset) so RViz can
    // display everything; cheap (published once, latched).
    static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    broadcastStaticTfs();

    // hand_size_scaling_factor is tunable at runtime (hand_scale_calib_node
    // closes a loop over it); rclcpp does not sync member copies of
    // parameters, so mirror accepted updates into the member each solve reads.
    param_cb_handle_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & params) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto & p : params) {
          const auto & name = p.get_name();
          if (name == "hand_size_scaling_factor") {
            const double v = p.as_double();
            if (v <= 0.0) {
              result.successful = false;
              result.reason = "hand_size_scaling_factor must be > 0";
              break;
            }
            hand_size_scaling_factor_ = v;
            RCLCPP_INFO(get_logger(), "hand_size_scaling_factor set to %.4f", v);
          } else if (name == "log_input") {
            log_input_ = p.as_bool();
          } else if (name == "log_score_gate") {
            log_score_gate_ = p.as_bool();
          } else if (name == "log_solve") {
            log_solve_ = p.as_bool();
          } else if (name == "log_pose") {
            log_pose_ = p.as_bool();
          } else if (name == "log_trigger") {
            log_trigger_ = p.as_bool();
          } else if (name == "log_gate_summary") {
            log_gate_summary_ = p.as_bool();
          } else if (name == "log_reproj_warn_px") {
            log_reproj_warn_px_ = p.as_double();
          } else if (name == "log_throttle_sec") {
            const double v = p.as_double();
            if (v < 0.0) {
              result.successful = false;
              result.reason = "log_throttle_sec must be >= 0";
              break;
            }
            log_throttle_sec_ = v;
          } else if (name == "min_score") {
            min_score_ = p.as_double();
            RCLCPP_INFO(get_logger(), "min_score set to %.3f", min_score_);
          } else if (name == "pose_timeout_sec") {
            const double v = p.as_double();
            if (v <= 0.0) {
              result.successful = false;
              result.reason = "pose_timeout_sec must be > 0";
              break;
            }
            pose_timeout_sec_ = v;
            RCLCPP_INFO(get_logger(), "pose_timeout_sec set to %.3f", pose_timeout_sec_);
          }
        }
        return result;
      });

    // Per-stage timings, published once a second as a JSON std_msgs/String
    // (same schema as hand_landmarks_node's) for perf_monitor_node to record.
    enable_perf_ = declare_parameter<bool>("enable_perf", true);
    if (enable_perf_) {
      perf_pub_ = create_publisher<std_msgs::msg::String>(
        declare_parameter<std::string>("perf_topic", "body_cam_teleop/perf"), 5);
      perf_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {publishPerf();});
    }

    // Drains the funnel counters regardless of log_gate_summary, so the flag
    // can be flipped at runtime without reporting a stale backlog.
    gate_window_start_ = PerfClock::now();
    gate_timer_ = create_wall_timer(
      std::chrono::duration<double>(std::max(log_summary_period_sec_, 0.1)),
      [this]() {publishGateSummary();});

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_hz_), [this]() {tick();});

    RCLCPP_INFO(
      get_logger(),
      "hand_pose_node up: camera=%s, landmarks=%s -> %s (%s frame) @ %.0f Hz; "
      "hand_size_scaling=%.2f, reprojection=%s",
      camera_name_.c_str(), landmarks_topic_.c_str(), output_topic_.c_str(),
      body_frame_.c_str(), publish_hz_, hand_size_scaling_factor_,
      enable_reprojection_ ? "on" : "off");
    {
      std::string enabled;
      const std::pair<const char *, bool> flags[] = {
        {"input", log_input_}, {"score_gate", log_score_gate_}, {"solve", log_solve_},
        {"pose", log_pose_}, {"trigger", log_trigger_},
        {"gate_summary", log_gate_summary_}};
      for (const auto & [name, on] : flags) {
        if (on) {enabled += (enabled.empty() ? "" : ", ") + std::string(name);}
      }
      RCLCPP_INFO(
        get_logger(),
        "stage logging: %s (throttle %.1fs, reproj warn %.0f px; toggle with "
        "`ros2 param set <node> log_<stage> true`)",
        enabled.empty() ? "none" : enabled.c_str(), log_throttle_sec_,
        log_reproj_warn_px_);
    }
  }

private:
  struct HandState
  {
    geometry_msgs::msg::Pose pose;   // body-frame pose
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    bool valid{false};
    // Warm start for the next solve: [angle-axis, translation] T_cam_hand.
    std::array<double, 6> last_params{};
    bool has_last_params{false};
    // Ray-noise filter memory (camera frame, onLandmarks thread only).
    Eigen::Vector3d filt_pos_cam{Eigen::Vector3d::Zero()};
    rclcpp::Time filt_stamp{0, 0, RCL_ROS_TIME};
    bool has_filt{false};
  };

  // Monocular PnP noise is dominated by the camera->hand ray direction (the
  // solve is scale/depth ambiguous), and every direction degrades with
  // distance (pixel sensitivity ~ 1/L). Split the frame-to-frame position
  // step into its ray-aligned and lateral parts, apply ray_filter_k /
  // lateral_filter_k respectively, and scale both gains by
  // f(L) = clamp(filter_ref_dist_m / L, filter_min_scale, 1): pass-through
  // inside the reference distance, ~1/L beyond it, never below the floor.
  // Camera frame only: the ray is normalize(p) there (camera at origin).
  // Returns the filtered position; resets to raw when the previous sample is
  // stale.
  Eigen::Vector3d rayFilter(HandState & state, const Eigen::Vector3d & p_raw)
  {
    const auto stamp = now();
    Eigen::Vector3d p_filt = p_raw;
    const bool fresh =
      state.has_filt && (stamp - state.filt_stamp).seconds() <= pose_timeout_sec_;
    const double dist = state.filt_pos_cam.norm();
    if (fresh && dist > 1e-6) {
      const double f = filter_ref_dist_m_ > 0.0 ?
        std::clamp(filter_ref_dist_m_ / dist, filter_min_scale_, 1.0) : 1.0;
      const Eigen::Vector3d ray = state.filt_pos_cam / dist;
      const Eigen::Vector3d dp = p_raw - state.filt_pos_cam;
      const Eigen::Vector3d dp_par = dp.dot(ray) * ray;
      p_filt = state.filt_pos_cam +
        f * (lateral_filter_k_ * (dp - dp_par) + ray_filter_k_ * dp_par);
    }
    state.filt_pos_cam = p_filt;
    state.filt_stamp = stamp;
    state.has_filt = true;
    return p_filt;
  }

  // ---- calibration -------------------------------------------------------
  void loadCalibration()
  {
    // Intrinsics for the selected camera.
    const YAML::Node intr = YAML::LoadFile(intrinsics_file_);
    const YAML::Node cam_intr = intr["cameras"][camera_name_];
    if (!cam_intr) {
      throw std::runtime_error(
              "camera '" + camera_name_ + "' not found in " + intrinsics_file_);
    }
    const auto ivals = cam_intr["intrinsics"].as<std::vector<double>>();
    fx_ = ivals[0];
    fy_ = ivals[1];
    cx_ = ivals[2];
    cy_ = ivals[3];
    K_ = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
    const std::string model = cam_intr["model"] ?
      cam_intr["model"].as<std::string>() : "pinhole-radtan";
    if (model == "pinhole-equi" || model == "pinhole-equidistant") {
      fisheye_ = true;
    } else if (model == "pinhole-radtan" || model == "pinhole" || model == "plumb_bob") {
      fisheye_ = false;
    } else {
      throw std::runtime_error(
              "camera '" + camera_name_ + "': unsupported model '" + model +
              "' (expected pinhole-radtan or pinhole-equi)");
    }
    auto dvals = cam_intr["distortion"].as<std::vector<double>>();
    if (fisheye_) {
      // equi is exactly [k1,k2,k3,k4]; anything else means the yaml and the
      // model tag disagree — refuse rather than misinterpret coefficients.
      if (dvals.size() != 4) {
        throw std::runtime_error(
                "camera '" + camera_name_ + "': pinhole-equi needs 4 distortion "
                "coefficients [k1,k2,k3,k4], got " + std::to_string(dvals.size()));
      }
    } else {
      // radtan gives [k1,k2,p1,p2]; OpenCV plumb_bob wants [k1,k2,p1,p2,k3].
      while (dvals.size() < 5) {
        dvals.push_back(0.0);
      }
    }
    dist_ = cv::Mat(dvals, true);

    // Camera -> operator_body: the camera is mounted at the operator body
    // center, so the transform is the fixed parameter offset (identity by
    // default). T_body_cam maps camera-frame points into the body frame.
    Eigen::Matrix4d T_cam_body = Eigen::Matrix4d::Identity();
    T_cam_body.block<3, 3>(0, 0) = eulerDegToR(
      operator_body_rotation_[0], operator_body_rotation_[1], operator_body_rotation_[2]);
    T_cam_body(0, 3) = operator_body_position_[0];
    T_cam_body(1, 3) = operator_body_position_[1];
    T_cam_body(2, 3) = operator_body_position_[2];
    T_body_cam_ = T_cam_body.inverse();

    RCLCPP_INFO(
      get_logger(),
      "calibration loaded: %s fx=%.1f fy=%.1f cx=%.1f cy=%.1f; %s -> %s fixed mount offset",
      fisheye_ ? "pinhole-equi" : "pinhole-radtan", fx_, fy_, cx_, cy_,
      camera_name_.c_str(), body_frame_.c_str());
  }

  void broadcastStaticTfs()
  {
    // Root the debug tree at the body frame (body -> camera). The per-camera
    // <ns>/operator_body frames coincide by construction, so the launch file
    // can bridge them with a static identity TF and RViz renders every camera
    // in one scene.
    static_tf_broadcaster_->sendTransform(
      matrixToTf(T_body_cam_, tf_body_frame_, camera_name_, now()));
  }

  // ---- diagnostics ---------------------------------------------------------
  // Throttle keyed by `key` instead of by call site: the RCLCPP_*_THROTTLE
  // macros keep their state in a static at the call site, which would let one
  // reject reason (or one hand) mask another emitted from the same line.
  // Executor-thread only, like perf_ — no locking.
  bool diagReady(const std::string & key)
  {
    const auto t = PerfClock::now();
    const auto it = diag_last_.find(key);
    if (it != diag_last_.end() &&
      std::chrono::duration<double>(t - it->second).count() < log_throttle_sec_)
    {
      return false;
    }
    diag_last_[key] = t;
    return true;
  }

  void gateCount(const std::string & key, int64_t inc = 1) {gate_[key] += inc;}

  int64_t gateGet(const std::string & key) const
  {
    const auto it = gate_.find(key);
    return it == gate_.end() ? 0 : it->second;
  }

  // The funnel: how many hands each gate consumed over the window.
  void publishGateSummary()
  {
    const auto now_tp = PerfClock::now();
    const double window =
      std::chrono::duration<double>(now_tp - gate_window_start_).count();
    gate_window_start_ = now_tp;
    if (log_gate_summary_) {
      RCLCPP_INFO(
        get_logger(),
        "[gate funnel %.1fs] msgs=%ld hands=%ld -> gate3 rejected: "
        "score=%ld landmark_count=%ld (dup=%ld) -> gate4 rejected: solve=%ld "
        "-> posed=%ld | gate5 trigger held: Left %ld/%ld ticks, Right %ld/%ld",
        window, gateGet("msgs"), gateGet("hands"), gateGet("rej_score"),
        gateGet("rej_landmark_count"), gateGet("dup_label"), gateGet("rej_solve"),
        gateGet("posed"), gateGet("trigger_Left"), gateGet("ticks"),
        gateGet("trigger_Right"), gateGet("ticks"));
    }
    gate_.clear();
  }

  // ---- perf ----------------------------------------------------------------
  void perfAdd(const std::string & stage, double value)
  {
    if (enable_perf_) {
      perf_[stage].add(value);
    }
  }

  void publishPerf()
  {
    const auto now_tp = PerfClock::now();
    const double window_sec =
      std::chrono::duration<double>(now_tp - perf_window_start_).count();
    perf_window_start_ = now_tp;

    std::ostringstream os;
    os.setf(std::ios::fixed);
    os << std::setprecision(3);
    os << "{\"node\":\"" << get_fully_qualified_name() << "\","
       << "\"window_sec\":" << window_sec << ",\"stages\":{";
    bool first = true;
    for (const auto & [name, s] : perf_) {
      os << (first ? "" : ",") << "\"" << name << "\":{\"n\":" << s.n
         << ",\"mean_ms\":" << (s.n ? s.sum / s.n : 0.0)
         << ",\"max_ms\":" << s.max << "}";
      first = false;
    }
    os << "},\"counters\":{}}";
    perf_.clear();

    std_msgs::msg::String msg;
    msg.data = os.str();
    perf_pub_->publish(msg);
  }

  // ---- landmark processing ------------------------------------------------
  void onLandmarks(const handpose3d_msgs::msg::HandLandmarks & msg)
  {
    const auto t_cb = PerfClock::now();
    // Age of the landmarks at arrival = camera capture + MediaPipe + DDS hop
    // (both stamps come from this host's clock).
    const double age_ms = (now() - rclcpp::Time(msg.header.stamp)).seconds() * 1e3;
    perfAdd("latency_capture_to_pose_ms", age_ms);
    std::map<std::string, Eigen::Matrix<double, kNumLandmarks, 3>> placed_joints;
    std::map<std::string, cv::Mat> detected_px;

    gateCount("msgs");
    gateCount("hands", static_cast<int64_t>(msg.hands.size()));
    if (log_input_) {
      std::string listing;
      for (const auto & hand : msg.hands) {
        listing += (listing.empty() ? "" : ", ") + hand.handedness + ":" +
          std::to_string(hand.score).substr(0, 4);
      }
      if (msg.hands.empty()) {
        if (diagReady("input_empty")) {
          RCLCPP_WARN(
            get_logger(),
            "[stage input] landmarks msg with 0 hands (age %.1f ms) — gate 1/2 "
            "upstream in hand_landmarks_node ate it; enable its log_detection / "
            "log_handedness",
            age_ms);
        }
      } else if (diagReady("input")) {
        RCLCPP_INFO(
          get_logger(), "[stage input] %zu hand(s) [%s], age %.1f ms",
          msg.hands.size(), listing.c_str(), age_ms);
      }
    }

    // Highest-score detection per handedness label.
    std::map<std::string, const handpose3d_msgs::msg::Hand *> best;
    for (const auto & hand : msg.hands) {
      // ---- gate 3: score and landmark-count contract ---------------------
      if (hand.score < min_score_) {
        gateCount("rej_score");
        if (log_score_gate_ && diagReady("rej_score_" + hand.handedness)) {
          RCLCPP_WARN(
            get_logger(),
            "[gate3 score] dropped '%s': score %.3f < min_score %.3f (this is "
            "MediaPipe's HANDEDNESS confidence, not detection confidence)",
            hand.handedness.c_str(), hand.score, min_score_);
        }
        continue;
      }
      if (hand.landmarks_image.size() != kNumLandmarks ||
        hand.landmarks_world.size() != kNumLandmarks)
      {
        gateCount("rej_landmark_count");
        if (log_score_gate_ && diagReady("rej_count_" + hand.handedness)) {
          RCLCPP_WARN(
            get_logger(),
            "[gate3 count] dropped '%s': got %zu image / %zu world landmarks, "
            "expected %d each",
            hand.handedness.c_str(), hand.landmarks_image.size(),
            hand.landmarks_world.size(), kNumLandmarks);
        }
        continue;
      }
      auto it = best.find(hand.handedness);
      if (it != best.end()) {
        // Two detections claiming the same hand: only the higher score is
        // posed. Common when MediaPipe mislabels a second hand in frame.
        gateCount("dup_label");
        if (log_score_gate_ && diagReady("dup_" + hand.handedness)) {
          RCLCPP_WARN(
            get_logger(),
            "[gate3 dup] two '%s' detections in one frame (scores %.3f, %.3f); "
            "keeping the higher one",
            hand.handedness.c_str(), it->second->score, hand.score);
        }
      }
      if (it == best.end() || hand.score > it->second->score) {
        best[hand.handedness] = &hand;
      }
    }
    if (log_score_gate_ && !msg.hands.empty() && best.empty() &&
      diagReady("gate3_all_rejected"))
    {
      RCLCPP_WARN(
        get_logger(), "[gate3] all %zu detection(s) rejected — nothing to solve",
        msg.hands.size());
    }

    for (const auto & [label, hand] : best) {
      // Undistort the 21 detected pixels (points only, same K) so the solver
      // cost can be pure pinhole.
      const auto t_undistort = PerfClock::now();
      cv::Mat raw_px(kNumLandmarks, 1, CV_64FC2), undist_px;
      for (int i = 0; i < kNumLandmarks; ++i) {
        raw_px.at<cv::Vec2d>(i) = {hand->landmarks_image[i].x, hand->landmarks_image[i].y};
      }
      if (fisheye_) {
        cv::fisheye::undistortPoints(raw_px, undist_px, K_, dist_, cv::noArray(), K_);
      } else {
        cv::undistortPoints(raw_px, undist_px, K_, dist_, cv::noArray(), K_);
      }

      // Hand-local metric model, rescaled to the operator's real hand size.
      Eigen::Matrix<double, kNumLandmarks, 3> x_hand;
      for (int i = 0; i < kNumLandmarks; ++i) {
        x_hand.row(i) = hand_size_scaling_factor_ *
          Eigen::Vector3d(
          hand->landmarks_world[i].x, hand->landmarks_world[i].y, hand->landmarks_world[i].z);
      }
      perfAdd("undistort_ms", msSince(t_undistort));

      auto & state = states_[label];
      const auto t_solve = PerfClock::now();
      const SolveResult res = solvePose(x_hand, undist_px, state);
      perfAdd("solve_ms", msSince(t_solve));
      // ---- gate 4: did the Ceres solve produce a usable pose? -------------
      if (!res.success) {
        perfAdd("solve_failed", 1.0);
        gateCount("rej_solve");
        if (log_solve_ && diagReady("solve_fail_" + label)) {
          RCLCPP_WARN(
            get_logger(),
            "[gate4 solve] '%s' FAILED: %s (residual blocks %d of %d joints, "
            "%d non-finite; cost %.3g -> %.3g after %d iters; warm_start=%s)",
            label.c_str(), res.termination.c_str(), res.num_residual_blocks,
            kNumLandmarks, res.num_skipped, res.initial_cost, res.final_cost,
            res.num_iterations, res.warm_started ? "yes" : "no");
        }
        continue;
      }
      perfAdd("solve_iters", static_cast<double>(res.num_iterations));
      perfAdd("solve_reproj_rms_px", res.reproj_rms_px);
      if (log_solve_ && diagReady("solve_" + label)) {
        RCLCPP_INFO(
          get_logger(),
          "[gate4 solve] '%s' ok: reproj rms %.1f px, %d iters, %d/%d joints, "
          "cost %.3g -> %.3g, term=%s, seed %s (depth %.3f m)",
          label.c_str(), res.reproj_rms_px, res.num_iterations,
          res.num_residual_blocks, kNumLandmarks, res.initial_cost, res.final_cost,
          res.termination.c_str(), res.warm_started ? "warm" : "cold",
          res.seed_depth_m);
      }
      // Converged, but to a fit that does not match the pixels: the pose is
      // still published (nothing gates on this), so say so loudly.
      if (log_reproj_warn_px_ > 0.0 && res.reproj_rms_px > log_reproj_warn_px_ &&
        diagReady("reproj_warn_" + label))
      {
        RCLCPP_WARN(
          get_logger(),
          "[gate4 fit] '%s' reproj rms %.1f px > log_reproj_warn_px %.1f: the "
          "placed model does not match the detected landmarks, so the published "
          "pose is unreliable. Check hand_size_scaling_factor (%.2f) and the "
          "'%s' intrinsics (fx=%.1f fy=%.1f cx=%.1f cy=%.1f)",
          label.c_str(), res.reproj_rms_px, log_reproj_warn_px_,
          hand_size_scaling_factor_, camera_name_.c_str(), fx_, fy_, cx_, cy_);
      }

      // Place the model in the camera frame; derive the published pose.
      const Eigen::Matrix3d R = res.T_cam_hand.block<3, 3>(0, 0);
      const Eigen::Vector3d t = res.T_cam_hand.block<3, 1>(0, 3);
      Eigen::Matrix<double, kNumLandmarks, 3> joints_cam =
        (x_hand * R.transpose()).rowwise() + t.transpose();

      Eigen::Matrix4d T_cam_pose = res.T_cam_hand;
      bool anat_ok = true;
      if (anatomical_hand_frame_) {
        if (const auto anat = anatomicalFrame(joints_cam)) {
          T_cam_pose = *anat;
        } else {
          // Degenerate palm triangle: falls back to MediaPipe's arbitrary
          // local frame, so the published orientation convention silently
          // changes for this frame.
          anat_ok = false;
          if (log_pose_ && diagReady("anat_fail_" + label)) {
            RCLCPP_WARN(
              get_logger(),
              "[stage pose] '%s': anatomical palm frame degenerate (wrist/index/"
              "pinky MCP collinear or non-finite); falling back to MediaPipe's "
              "local frame for this sample",
              label.c_str());
          }
        }
      }
      double filter_step_mm = 0.0, raw_step_mm = 0.0;
      if (pose_filter_enabled_) {
        const Eigen::Vector3d p_raw = T_cam_pose.block<3, 1>(0, 3);
        const Eigen::Vector3d p_prev = state.filt_pos_cam;
        const bool had_filt = state.has_filt;
        const Eigen::Vector3d p_filt = rayFilter(state, p_raw);
        T_cam_pose.block<3, 1>(0, 3) = p_filt;
        // Shift the placed joints too so markers/TF/reprojection show what is
        // actually published.
        joints_cam.rowwise() += (p_filt - p_raw).transpose();
        if (had_filt) {
          raw_step_mm = (p_raw - p_prev).norm() * 1e3;
          filter_step_mm = (p_filt - p_prev).norm() * 1e3;
        }
      }
      const Eigen::Matrix4d T_body_pose = T_body_cam_ * T_cam_pose;
      if (log_pose_ && diagReady("pose_" + label)) {
        const Eigen::Vector3d p_cam = T_cam_pose.block<3, 1>(0, 3);
        const Eigen::Vector3d p_body = T_body_pose.block<3, 1>(0, 3);
        RCLCPP_INFO(
          get_logger(),
          "[stage pose] '%s': cam (%.3f %.3f %.3f) dist %.3f m -> %s "
          "(%.3f %.3f %.3f); filter step %.1f -> %.1f mm; frame=%s",
          label.c_str(), p_cam.x(), p_cam.y(), p_cam.z(), p_cam.norm(),
          body_frame_.c_str(), p_body.x(), p_body.y(), p_body.z(), raw_step_mm,
          filter_step_mm, anat_ok && anatomical_hand_frame_ ? "anatomical" : "mediapipe");
      }

      geometry_msgs::msg::Pose pose = matrixToPose(T_body_pose);
      if (offset_quat_) {
        // Right-multiply: fixed rotation in the hand's own frame.
        Eigen::Quaterniond q(
          pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        q = q * (*offset_quat_);
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();
      }

      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state.pose = pose;
        state.stamp = now();
        state.valid = true;
      }
      gateCount("posed");

      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000, "[%s] posed (reproj rms %.1f px)", label.c_str(),
        res.reproj_rms_px);

      if (enable_reprojection_) {
        placed_joints[label] = joints_cam;
        detected_px[label] = raw_px;
        broadcastHandTf(label, T_body_pose, msg.header.stamp);
      }
    }

    if (enable_reprojection_) {
      const auto t_markers = PerfClock::now();
      publishMarkers(placed_joints, msg.header.stamp);
      perfAdd("markers_ms", msSince(t_markers));
      const auto t_reproject = PerfClock::now();
      publishReprojection(placed_joints, detected_px, msg.header.stamp);
      perfAdd("reproject_ms", msSince(t_reproject));
    }
    perfAdd("landmarks_cb_ms", msSince(t_cb));
  }

  SolveResult solvePose(
    const Eigen::Matrix<double, kNumLandmarks, 3> & x_hand, const cv::Mat & undist_px,
    HandState & state)
  {
    // Seed: previous pose (warm start) or hand seed_depth_m in front of the
    // camera with the hand frame aligned to the camera frame.
    std::array<double, 6> params{};
    SolveResult res;
    res.warm_started = warm_start_ && state.has_last_params;
    if (res.warm_started) {
      params = state.last_params;
    } else {
      params = {0.0, 0.0, 0.0, 0.0, 0.0, seed_depth_m_};
    }
    res.seed_depth_m = params[5];

    ceres::Problem problem;
    for (int i = 0; i < kNumLandmarks; ++i) {
      const auto & uv = undist_px.at<cv::Vec2d>(i);
      if (!std::isfinite(uv[0]) || !std::isfinite(uv[1]) ||
        !x_hand.row(i).allFinite())
      {
        ++res.num_skipped;
        continue;
      }
      problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<JointResidual, 3, 6>(
          new JointResidual(
            x_hand.row(i), Eigen::Vector2d(uv[0], uv[1]), fx_, fy_, cx_, cy_,
            cheirality_margin_, cheirality_weight_)),
        nullptr, params.data());
    }
    res.num_residual_blocks = static_cast<int>(problem.NumResidualBlocks());
    if (res.num_residual_blocks < 4) {
      res.termination = "too few finite landmarks (need >= 4 of " +
        std::to_string(kNumLandmarks) + ")";
      return res;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = max_solver_iterations_;
    options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    res.num_iterations = static_cast<int>(summary.iterations.size());
    res.initial_cost = summary.initial_cost;
    res.final_cost = summary.final_cost;
    res.termination = ceres::TerminationTypeToString(summary.termination_type);
    if (!summary.IsSolutionUsable()) {
      // Drop the warm start: seeding the next solve from an unusable pose
      // would keep it stuck in the same bad basin.
      state.has_last_params = false;
      res.termination += ": " + summary.message;
      return res;
    }
    state.last_params = params;
    state.has_last_params = true;

    res.success = true;
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(
      params.data(), ceres::ColumnMajorAdapter3x3(R.data()));
    res.T_cam_hand.block<3, 3>(0, 0) = R;
    res.T_cam_hand.block<3, 1>(0, 3) = Eigen::Vector3d(params[3], params[4], params[5]);
    // 2 reprojection residuals per joint; final cost = 0.5 * sum(r^2).
    res.reproj_rms_px =
      std::sqrt(2.0 * summary.final_cost / (2.0 * problem.NumResidualBlocks()));
    return res;
  }

  // Anatomical palm frame from placed joints (any frame): origin = wrist,
  // y = palm normal ((P5-P0) x (P17-P0)), z = -(unit(P5-P0)+unit(P17-P0))
  // (fingers -> wrist bisector), x = y x z.
  static std::optional<Eigen::Matrix4d> anatomicalFrame(
    const Eigen::Matrix<double, kNumLandmarks, 3> & joints)
  {
    const Eigen::Vector3d p0 = joints.row(kWristIdx);
    const Eigen::Vector3d v5 = Eigen::Vector3d(joints.row(kIndexMcpIdx)) - p0;
    const Eigen::Vector3d v17 = Eigen::Vector3d(joints.row(kPinkyMcpIdx)) - p0;
    if (!p0.allFinite() || !v5.allFinite() || !v17.allFinite()) {
      return std::nullopt;
    }
    const double n5 = v5.norm(), n17 = v17.norm();
    Eigen::Vector3d y = v5.cross(v17);
    const double ny = y.norm();
    if (n5 < 1e-9 || n17 < 1e-9 || ny < 1e-9) {
      return std::nullopt;
    }
    y /= ny;
    Eigen::Vector3d z = -(v5 / n5 + v17 / n17);
    const double nz = z.norm();
    if (nz < 1e-9) {
      return std::nullopt;
    }
    z /= nz;
    const Eigen::Vector3d x = y.cross(z);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 1>(0, 0) = x;
    T.block<3, 1>(0, 1) = y;
    T.block<3, 1>(0, 2) = z;
    T.block<3, 1>(0, 3) = p0;
    return T;
  }

  // ---- teleop output -------------------------------------------------------
  void tick()
  {
    const auto stamp = now();
    robot_interfaces::msg::TeleopMessage msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = body_frame_;
    msg.head_pose.orientation.w = 1.0;  // unused downstream; keep quat valid

    gateCount("ticks");
    fillController(msg.left_controller, "Left", stamp);
    fillController(msg.right_controller, "Right", stamp);

    geometry_msgs::msg::Pose nan_pose;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    nan_pose.position.x = nan_pose.position.y = nan_pose.position.z = nan;
    nan_pose.orientation.x = nan_pose.orientation.y = nan_pose.orientation.z =
      nan_pose.orientation.w = nan;
    msg.left_hand.joints.assign(kNumHandJoints, nan_pose);
    msg.right_hand.joints.assign(kNumHandJoints, nan_pose);

    teleop_pub_->publish(msg);
  }

  void fillController(
    robot_interfaces::msg::ControllerMessage & controller, const std::string & label,
    const rclcpp::Time & stamp)
  {
    controller.joy.buttons.assign(num_joy_buttons_, 0);
    bool fresh = false;
    bool ever_posed = false;
    double age = 0.0;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      auto it = states_.find(label);
      if (it != states_.end() && it->second.valid) {
        ever_posed = true;
        age = (stamp - it->second.stamp).seconds();
        if (age <= pose_timeout_sec_) {
          controller.pose = it->second.pose;
          fresh = true;
        }
      }
    }
    if (fresh) {
      // Hold the trigger so the arm controller's gate stays engaged.
      controller.joy.buttons[trigger_button_index_] = 1;
      gateCount("trigger_" + label);
    } else {
      controller.pose = geometry_msgs::msg::Pose();
      controller.pose.orientation.w = 1.0;
    }

    // ---- gate 5 diagnostics: this is what teleop_mux_node sees ------------
    // Transitions are the answer to "why did the mux drop to None", so they are
    // logged unthrottled; the waiting-state line is throttled.
    const auto prev = trigger_held_.find(label);
    const bool changed = prev == trigger_held_.end() || prev->second != fresh;
    trigger_held_[label] = fresh;
    if (!log_trigger_) {
      return;
    }
    if (changed && fresh) {
      RCLCPP_INFO(
        get_logger(),
        "[gate5 trigger] '%s' ENGAGED: pose age %.3f s <= pose_timeout_sec %.3f "
        "(button %d held -> mux can select this camera)",
        label.c_str(), age, pose_timeout_sec_, trigger_button_index_);
    } else if (changed && !fresh) {
      if (ever_posed) {
        RCLCPP_WARN(
          get_logger(),
          "[gate5 trigger] '%s' RELEASED: last pose was %.3f s ago > "
          "pose_timeout_sec %.3f — no detection survived gates 1-4 for that "
          "long (button %d cleared -> mux goes to None)",
          label.c_str(), age, pose_timeout_sec_, trigger_button_index_);
      } else {
        RCLCPP_WARN(
          get_logger(),
          "[gate5 trigger] '%s' never posed since startup (button %d cleared)",
          label.c_str(), trigger_button_index_);
      }
    } else if (!fresh && diagReady("waiting_" + label)) {
      if (ever_posed) {
        RCLCPP_INFO(
          get_logger(), "[gate5 trigger] '%s' still disengaged, last pose %.1f s ago",
          label.c_str(), age);
      } else {
        RCLCPP_INFO(
          get_logger(), "[gate5 trigger] '%s' still disengaged, never posed",
          label.c_str());
      }
    }
  }

  // ---- image input (reprojection overlay) -----------------------------------
  void onImage(sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    const auto t_cb = PerfClock::now();
    image_buffer_.emplace_back(rclcpp::Time(msg->header.stamp).nanoseconds(), msg);
    while (image_buffer_.size() > static_cast<size_t>(image_buffer_size_)) {
      image_buffer_.pop_front();
    }
    perfAdd("image_cb_ms", msSince(t_cb));
  }

  void broadcastHandTf(
    const std::string & label, const Eigen::Matrix4d & T_body_pose,
    const builtin_interfaces::msg::Time & stamp)
  {
    hand_tf_broadcaster_->sendTransform(
      matrixToTf(T_body_pose, tf_body_frame_, tf_prefix_ + "hand_" + label, stamp));
  }

  void publishMarkers(
    const std::map<std::string, Eigen::Matrix<double, kNumLandmarks, 3>> & placed_joints,
    const builtin_interfaces::msg::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray arr;
    int hand_idx = 0;
    for (const auto label : {"Left", "Right"}) {
      visualization_msgs::msg::Marker joints, bones;
      joints.header.frame_id = tf_body_frame_;
      joints.header.stamp = stamp;
      joints.ns = std::string("hand_") + label + "_joints";
      joints.id = hand_idx * 2;
      joints.type = visualization_msgs::msg::Marker::SPHERE_LIST;
      joints.action = visualization_msgs::msg::Marker::ADD;
      joints.scale.x = joints.scale.y = joints.scale.z = joint_size_;
      joints.color.a = 1.0;
      joints.color.r = hand_idx == 0 ? 0.2f : 1.0f;
      joints.color.g = hand_idx == 0 ? 0.6f : 0.5f;
      joints.color.b = hand_idx == 0 ? 1.0f : 0.2f;
      joints.lifetime = rclcpp::Duration(std::chrono::milliseconds(300));
      joints.pose.orientation.w = 1.0;
      bones = joints;
      bones.ns = std::string("hand_") + label + "_bones";
      bones.id = hand_idx * 2 + 1;
      bones.type = visualization_msgs::msg::Marker::LINE_LIST;
      bones.scale.x = line_width_;
      bones.scale.y = bones.scale.z = 0.0;

      const auto it = placed_joints.find(label);
      if (it != placed_joints.end()) {
        // Joints were placed in the camera frame; markers live in body_frame.
        const Eigen::Matrix3d R = T_body_cam_.block<3, 3>(0, 0);
        const Eigen::Vector3d t = T_body_cam_.block<3, 1>(0, 3);
        std::array<geometry_msgs::msg::Point, kNumLandmarks> pts;
        for (int i = 0; i < kNumLandmarks; ++i) {
          const Eigen::Vector3d p = R * Eigen::Vector3d(it->second.row(i)) + t;
          pts[i].x = p.x();
          pts[i].y = p.y();
          pts[i].z = p.z();
          joints.points.push_back(pts[i]);
        }
        for (const auto & [a, b] : kHandConnections) {
          bones.points.push_back(pts[a]);
          bones.points.push_back(pts[b]);
        }
      }
      arr.markers.push_back(joints);
      arr.markers.push_back(bones);
      ++hand_idx;
    }
    markers_pub_->publish(arr);
  }

  void publishReprojection(
    const std::map<std::string, Eigen::Matrix<double, kNumLandmarks, 3>> & placed_joints,
    const std::map<std::string, cv::Mat> & detected_px,
    const builtin_interfaces::msg::Time & stamp)
  {
    // Match the landmark stamp against the buffered raw frames.
    const int64_t target = rclcpp::Time(stamp).nanoseconds();
    sensor_msgs::msg::Image::ConstSharedPtr best;
    int64_t best_dt = 0;
    for (const auto & [t_ns, img] : image_buffer_) {
      const int64_t dt = std::llabs(t_ns - target);
      if (!best || dt < best_dt) {
        best = img;
        best_dt = dt;
      }
    }
    if (!best || best_dt > static_cast<int64_t>(image_match_tol_sec_ * 1e9)) {
      return;
    }

    cv_bridge::CvImagePtr cv_img;
    try {
      cv_img = cv_bridge::toCvCopy(best, "bgr8");
    } catch (const std::exception & exc) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "cv_bridge failed: %s", exc.what());
      return;
    }
    cv::Mat & frame = cv_img->image;

    std::vector<double> errors;
    for (const auto & [label, joints_cam] : placed_joints) {
      // Full distortion model back onto the RAW image (landmarks were
      // detected on the raw frame).
      std::vector<cv::Point3d> pts3(kNumLandmarks);
      for (int i = 0; i < kNumLandmarks; ++i) {
        pts3[i] = {joints_cam(i, 0), joints_cam(i, 1), joints_cam(i, 2)};
      }
      std::vector<cv::Point2d> proj;
      projectPointsModel(pts3, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K_, dist_, fisheye_, proj);

      const cv::Scalar color = label == "Left" ? cv::Scalar(255, 150, 50)
                                               : cv::Scalar(50, 150, 255);
      for (const auto & [a, b] : kHandConnections) {
        cv::line(frame, proj[a], proj[b], color, 2);
      }
      for (const auto & p : proj) {
        cv::circle(frame, p, 4, color, -1);
      }
      const auto det_it = detected_px.find(label);
      if (det_it != detected_px.end()) {
        for (int i = 0; i < kNumLandmarks; ++i) {
          const auto & d = det_it->second.at<cv::Vec2d>(i);
          cv::circle(frame, cv::Point2d(d[0], d[1]), 6, cv::Scalar(60, 220, 60), 1);
          errors.push_back(cv::norm(cv::Point2d(d[0], d[1]) - cv::Point2d(proj[i])));
        }
      }
    }
    if (!errors.empty()) {
      const double mean =
        std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
      cv::putText(
        frame, cv::format("reproj err: %.1fpx (n=%zu)", mean, errors.size()),
        {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
    }
    reprojected_pub_->publish(*cv_img->toImageMsg());
  }

  // ---- members -------------------------------------------------------------
  std::string camera_name_, intrinsics_file_;
  std::string body_frame_;
  std::string tf_prefix_, tf_body_frame_;
  std::string landmarks_topic_, output_topic_, image_topic_, reprojected_topic_, markers_topic_;
  std::vector<double> operator_body_position_, operator_body_rotation_;
  double publish_hz_{50.0}, pose_timeout_sec_{0.5}, min_score_{0.5};
  double hand_size_scaling_factor_{1.3}, cheirality_margin_{0.05}, cheirality_weight_{1000.0};
  double seed_depth_m_{0.5}, image_match_tol_sec_{0.05}, joint_size_{0.012}, line_width_{0.006};
  int trigger_button_index_{5}, num_joy_buttons_{16}, image_buffer_size_{15};
  int max_solver_iterations_{50};
  double ray_filter_k_{1.0}, lateral_filter_k_{1.0};
  double filter_ref_dist_m_{0.0}, filter_min_scale_{0.2};
  bool pose_filter_enabled_{false};
  bool warm_start_{true}, anatomical_hand_frame_{true}, enable_reprojection_{false};
  std::optional<Eigen::Quaterniond> offset_quat_;

  double fx_{0}, fy_{0}, cx_{0}, cy_{0};
  cv::Mat K_, dist_;
  bool fisheye_{false};  // distortion model: pinhole-equi vs pinhole-radtan
  Eigen::Matrix4d T_body_cam_{Eigen::Matrix4d::Identity()};

  // Perf instrumentation (single-threaded executor: no locking needed).
  bool enable_perf_{true};
  std::map<std::string, StageStats> perf_;
  PerfClock::time_point perf_window_start_{PerfClock::now()};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr perf_pub_;
  rclcpp::TimerBase::SharedPtr perf_timer_;

  // Per-stage diagnostics. Same threading assumption as perf_: every writer
  // (landmarks callback, publish timer, gate timer) runs on the default
  // single-threaded executor. The log_* flags are also written by the parameter
  // callback, which the same executor serves.
  bool log_input_{false}, log_score_gate_{false}, log_solve_{false};
  bool log_pose_{false}, log_trigger_{false}, log_gate_summary_{false};
  double log_throttle_sec_{2.0}, log_summary_period_sec_{5.0};
  double log_reproj_warn_px_{30.0};
  std::map<std::string, PerfClock::time_point> diag_last_;
  std::map<std::string, int64_t> gate_;
  PerfClock::time_point gate_window_start_{PerfClock::now()};
  rclcpp::TimerBase::SharedPtr gate_timer_;
  std::map<std::string, bool> trigger_held_;  // last trigger state per hand

  std::mutex state_mutex_;
  std::map<std::string, HandState> states_;
  std::deque<std::pair<int64_t, sensor_msgs::msg::Image::ConstSharedPtr>> image_buffer_;

  rclcpp::Publisher<robot_interfaces::msg::TeleopMessage>::SharedPtr teleop_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr reprojected_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
  rclcpp::Subscription<handpose3d_msgs::msg::HandLandmarks>::SharedPtr landmarks_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> hand_tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HandPoseNode>());
  rclcpp::shutdown();
  return 0;
}
