#ifndef REALSENSE_DATASET_H_
#define REALSENSE_DATASET_H_

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <atomic>
#include <limits>
#include <string>

#include "imu.h"        // ImuData, ImuDataList already defined here

// Live Intel RealSense D435i source that exposes the same duck-typed
// API the offline Dataset class provides:
//   RealSenseDataset(...)
//   size_t GetDatasetLength()
//   bool GetData(cv::Mat&, cv::Mat&, ImuDataList&, double&)
//
// No inheritance, no changes to MapBuilder or Dataset required.
class RealSenseDataset {
 public:
  // fps: shared frame rate for the two IR streams.
  // width/height: IR stream resolution (D435i native infra modes).
  // enable_emitter: keep false for stereo VO — the IR projector dot
  // pattern corrupts stereo matching if left on.
  explicit RealSenseDataset(int width = 848, int height = 480, int fps = 30,
                             bool enable_emitter = false);
  ~RealSenseDataset();

  // Live stream has no fixed length. Return the max value so any
  // existing "for (i = 0; i < dataset.GetDatasetLength(); ++i)" loop
  // in visual_odometry.cpp runs until externally stopped (Ctrl+C /
  // ros::ok() / whatever guards the demo loop already).
  size_t GetDatasetLength() const {
    return std::numeric_limits<size_t>::max();
  }

  // Blocks until the next synchronized stereo IR pair is available,
  // fills left_image/right_image, and returns every IMU sample
  // accumulated since the previous call in batch_imu_data (sorted by
  // timestamp, ascending). timestamp is the stereo frame's capture
  // time in seconds, same units/convention as the EuRoC Dataset.
  //
  // Returns false only on pipeline error / stream teardown.
  bool GetData(cv::Mat& left_image, cv::Mat& right_image,
               ImuDataList& batch_imu_data, double& timestamp);

 private:
  void FrameCallback(const rs2::frame& frame);

  static double ToSeconds(double rs_timestamp_ms) {
    return rs_timestamp_ms * 1.0e-3;
  }

  rs2::pipeline pipe_;
  rs2::config cfg_;

  int width_;
  int height_;
  int fps_;
  bool enable_emitter_;

  // IMU samples pushed by the callback thread, drained by GetData().
std::mutex imu_mutex_;
std::deque<ImuData> imu_queue_;

// Latest accel measurement. Each gyro sample is paired with the most
// recent accel sample (nearest-previous), matching the D435i's
// asynchronous IMU streams.
Eigen::Vector3d last_accel_ = Eigen::Vector3d::Zero();

// Stereo frame container.
struct StereoFrame {
    cv::Mat left;
    cv::Mat right;
    double timestamp;
};

// Queue of stereo frames waiting to be consumed by AirSLAM.
std::mutex frame_mutex_;
std::condition_variable frame_cv_;
std::deque<StereoFrame> frame_queue_;

std::atomic<bool> running_{false};
};

#endif  // REALSENSE_DATASET_H_
