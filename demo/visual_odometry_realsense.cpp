// demo/visual_odometry_realsense.cpp
//
// Live-camera counterpart to demo/visual_odometry.cpp. Everything from
// dataset construction onward mirrors the original as closely as
// possible; the only differences are:
//   - RealSenseDataset instead of Dataset (no dataroot, no index arg
//     to GetData(), no fixed dataset_length)
//   - an unbounded ros::ok() loop instead of a fixed-length for loop,
//     since a live camera stream never ends on its own
//
// MapBuilder and Dataset are untouched.

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>
#include <thread>
#include "read_configs.h"
#include "realsense_dataset.h"
#include "map_builder.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "air_slam_realsense");
  std::string config_path, model_dir;
  ros::param::get("~config_path", config_path);
  ros::param::get("~model_dir", model_dir);
  VisualOdometryConfigs configs(config_path, model_dir);
  std::cout << "config done" << std::endl;
  ros::param::get("~camera_config_path", configs.camera_config_path);
  ros::param::get("~saving_dir", configs.saving_dir);
  ros::NodeHandle nh;
  MapBuilder map_builder(configs, nh);
  std::cout << "map_builder done" << std::endl;

  // RealSenseDataset instead of Dataset. No dataroot — the live camera
  // is the data source. Duck-typed drop-in: same GetData()/
  // GetDatasetLength() contract as Dataset, so MapBuilder is unaffected.
  std::cout << "Initializing RealSense..." << std::endl;
  RealSenseDataset dataset;
  std::cout << "Waiting for RealSense frames..." << std::endl;

  double sum_time = 0;
  int image_num = 0;

  // Live camera has no fixed length — loop until ROS shuts down
  // instead of iterating dataset_length times.
  size_t i = 0;
  while (ros::ok()) {
    std::cout << "i ====== " << i << std::endl;
    cv::Mat image_left, image_right;
    double timestamp;
    ImuDataList batch_imu_data;

    // GetData() — no index argument; RealSenseDataset always returns
    // the next available frame rather than a specific dataset index.
    if (!dataset.GetData(image_left, image_right, batch_imu_data, timestamp)) continue;

    auto data = std::make_shared<InputData>();
    data->index = i;
    data->time = timestamp;
    data->image_left = image_left;
    data->image_right = image_right;
    data->batch_imu_data = batch_imu_data;

    auto before_infer = std::chrono::high_resolution_clock::now();
    map_builder.AddInput(data);
    auto after_infer = std::chrono::high_resolution_clock::now();
    auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(after_infer - before_infer).count();
    sum_time += (double)cost_time;
    image_num++;
    std::cout << "One Frame Processing Time: " << cost_time << " ms." << std::endl;

    i++;
  }

  if(sum_time > 0)
    std::cout << "Average FPS = "
              << image_num / (sum_time / 1000.0)
              << std::endl;
  map_builder.Stop();
  while (!map_builder.IsStopped()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::cout << "Map building has been stopped" << std::endl;
  std::string trajectory_path = ConcatenateFolderAndFileName(configs.saving_dir, "trajectory_v0.txt");
  map_builder.SaveTrajectory(trajectory_path);
  map_builder.SaveMap(configs.saving_dir);
  ros::shutdown();
  return 0;
}
