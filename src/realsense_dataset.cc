#include "realsense_dataset.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace {
// Upper bound on buffered stereo pairs. If AirSLAM falls behind the
// camera's frame rate, we drop the oldest unconsumed frame rather
// than let the queue (and its held cv::Mat buffers) grow without
// bound and exhaust memory.
constexpr size_t kMaxFrameQueueSize = 10;
}  // namespace

RealSenseDataset::RealSenseDataset(int width, int height, int fps,
                                    bool enable_emitter)
    : width_(width), height_(height), fps_(fps),
      enable_emitter_(enable_emitter) {
  cfg_.enable_stream(RS2_STREAM_INFRARED, 1, width_, height_,
                      RS2_FORMAT_Y8, fps_);
  cfg_.enable_stream(RS2_STREAM_INFRARED, 2, width_, height_,
                      RS2_FORMAT_Y8, fps_);
  // D435i motion module: gyro/accel arrive on independent, uncorrelated
  // rates (typically ~200Hz gyro, ~63Hz accel). Request each explicitly.
  cfg_.enable_stream(RS2_STREAM_GYRO,  RS2_FORMAT_MOTION_XYZ32F);
  cfg_.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);

  // Set running_ before starting the pipeline so frames delivered on
  // the internal callback thread immediately after start() are never
  // dropped by a false "not running yet" check.
  running_ = true;

  // Callback-based start: librealsense delivers every frame (video or
  // motion) on an internal thread as soon as it's ready, so IMU and
  // image timestamps interleave naturally without us polling.
  rs2::pipeline_profile profile =
      pipe_.start(cfg_, [this](const rs2::frame& frame) {
        FrameCallback(frame);
      });

  if (!enable_emitter_) {
    auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
    if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
      depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f);
    }
  }

  // Debug visibility: confirm which clock librealsense is timestamping
  // frames with (e.g. RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK vs
  // SYSTEM_TIME). This matters for IMU/frame sync correctness and is
  // useful to see once at startup rather than dug up during debugging.
  {
    auto ir_stream = profile.get_stream(RS2_STREAM_INFRARED, 1);
    rs2::video_stream_profile video_profile = ir_stream.as<rs2::video_stream_profile>();
    std::cout << "[RealSenseDataset] Note: timestamp domain is reported "
                 "per-frame by librealsense; first frame's domain will be "
                 "logged once received." << std::endl;
  }
}

RealSenseDataset::~RealSenseDataset() {
  running_ = false;
  frame_cv_.notify_all();
  pipe_.stop();
}

void RealSenseDataset::FrameCallback(const rs2::frame& frame) {
  // --- IMU samples: buffer independently, GetData() drains them ---
  if (auto motion = frame.as<rs2::motion_frame>()) {
    auto stream_type = motion.get_profile().stream_type();
    if (stream_type != RS2_STREAM_GYRO && stream_type != RS2_STREAM_ACCEL) {
      return;
    }

    rs2_vector v = motion.get_motion_data();
    double ts = ToSeconds(motion.get_timestamp());

    std::lock_guard<std::mutex> lock(imu_mutex_);

    if (stream_type == RS2_STREAM_GYRO) {
      // Gyro is the higher-rate stream; pair each gyro sample with the
      // most recently observed accel reading (nearest-previous), the
      // standard pragmatic approach for D435i's un-synchronized IMU.
      ImuData imu;
      imu.timestamp = ts;
      imu.gyr = Eigen::Vector3d(v.x, v.y, v.z);
      imu.acc = last_accel_;
      imu_queue_.push_back(imu);
    } else {
      last_accel_ = Eigen::Vector3d(v.x, v.y, v.z);
    }
    return;
  }

  // --- Stereo IR frameset: only handle it once both IR frames land ---
  if (auto fs = frame.as<rs2::frameset>()) {
    rs2::video_frame ir_left  = fs.get_infrared_frame(1);
    rs2::video_frame ir_right = fs.get_infrared_frame(2);
    if (!ir_left || !ir_right) return;

    static std::once_flag timestamp_domain_logged;
    std::call_once(timestamp_domain_logged, [&ir_left] {
      std::cout << "[RealSenseDataset] Timestamp domain: "
                << rs2_timestamp_domain_to_string(ir_left.get_frame_timestamp_domain())
                << std::endl;
    });

    cv::Mat left(cv::Size(ir_left.get_width(), ir_left.get_height()),
                 CV_8UC1, (void*)ir_left.get_data(), cv::Mat::AUTO_STEP);
    cv::Mat right(cv::Size(ir_right.get_width(), ir_right.get_height()),
                  CV_8UC1, (void*)ir_right.get_data(), cv::Mat::AUTO_STEP);

    StereoFrame stereo_frame;
    stereo_frame.left  = left.clone();
    stereo_frame.right = right.clone();
    stereo_frame.timestamp = ToSeconds(ir_left.get_timestamp());

    bool should_notify = false;
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (frame_queue_.size() >= kMaxFrameQueueSize) {
        // Consumer is falling behind; drop the oldest buffered frame
        // rather than growing unbounded.
        frame_queue_.pop_front();
      }
      frame_queue_.push_back(std::move(stereo_frame));
      should_notify = true;
    }
    // Notify outside the lock so the woken waiter doesn't immediately
    // block again trying to reacquire a mutex we're still holding.
    if (should_notify) {
      frame_cv_.notify_one();
    }
  }
}

bool RealSenseDataset::GetData(cv::Mat& left_image, cv::Mat& right_image,
                                ImuDataList& batch_imu_data,
                                double& timestamp) {
  if (!running_) return false;

  StereoFrame stereo_frame;

  // Block until the callback thread has produced at least one stereo
  // pair, then pop the oldest one so frames are consumed in capture
  // order without being skipped.
  {
    std::unique_lock<std::mutex> lock(frame_mutex_);
    frame_cv_.wait(lock, [this] { return !frame_queue_.empty() || !running_; });
    if (!running_ && frame_queue_.empty()) return false;

    stereo_frame = std::move(frame_queue_.front());
    frame_queue_.pop_front();
  }

  left_image  = stereo_frame.left;
  right_image = stereo_frame.right;
  timestamp   = stereo_frame.timestamp;

  // Drain every IMU sample accumulated since the previous frame,
  // matching the EuRoC Dataset's "batch since last call" contract.
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    batch_imu_data.clear();
    batch_imu_data.reserve(imu_queue_.size());  // ImuDataList assumed std::vector — remove if not

    while (!imu_queue_.empty() && imu_queue_.front().timestamp <= timestamp) {
      batch_imu_data.push_back(imu_queue_.front());
      imu_queue_.pop_front();
    }
  }

  return true;
}
