#include "airslam_interface.h"

#include <ros/ros.h>

#include <atomic>
#include <iostream>
#include <mutex>

#include "map_builder.h"
#include "read_configs.h"

class AirSLAMInterface::Impl {
public:
    std::unique_ptr<VisualOdometryConfigs> configs;
    std::unique_ptr<MapBuilder> map_builder;

    std::atomic<size_t> frame_index{0};
    std::atomic<bool> stopped{false};

    Impl(const std::string& config_path,
         const std::string& model_dir) {

        // MapBuilder currently uses ROS internally.
        // Initialize ROS only if the host application has not done so.
        if (!ros::isInitialized()) {
            int argc = 1;
            char arg0[] = "airslam_interface";
            char* argv[] = {arg0, nullptr};

            ros::init(argc, argv,
                      "airslam_interface",
                      ros::init_options::AnonymousName);
        }

        configs = std::make_unique<VisualOdometryConfigs>(
            config_path, model_dir);

        configs->camera_config_path =
            "/workspace/catkin_ws/src/AirSLAM/configs/camera/realsense_848_480.yaml";

        configs->saving_dir =
            "/workspace/results";

        ros::NodeHandle nh;

        map_builder =
            std::make_unique<MapBuilder>(*configs, nh);
    }

    void Stop() {
        bool expected = false;

        if (stopped.compare_exchange_strong(expected, true)) {
            if (map_builder) {
                map_builder->Stop();
            }
        }
    }

    ~Impl() {
        Stop();
    }
};

AirSLAMInterface::AirSLAMInterface(
    const std::string& config_path,
    const std::string& model_dir)
    : _impl(std::make_unique<Impl>(config_path, model_dir)) {
}

AirSLAMInterface::~AirSLAMInterface() = default;

bool AirSLAMInterface::ProcessStereoFrame(
    const cv::Mat& left,
    const cv::Mat& right,
    double timestamp) {

    if (!_impl || !_impl->map_builder) {
        return false;
    }

    if (_impl->stopped) {
        return false;
    }

    if (left.empty() || right.empty()) {
        return false;
    }

    auto data = std::make_shared<InputData>();

    data->index = _impl->frame_index++;
    data->time = timestamp;

    data->image_left = left.clone();
    data->image_right = right.clone();

    _impl->map_builder->AddInput(data);

    return true;
}

Eigen::Matrix4d AirSLAMInterface::GetCurrentPose() const {
    if (!_impl || !_impl->map_builder) {
        return Eigen::Matrix4d::Identity();
    }

    return _impl->map_builder->GetCurrentPose();
}

bool AirSLAMInterface::SaveTrajectory(
    const std::string& path) {

    if (!_impl || !_impl->map_builder || _impl->stopped) {
        return false;
    }

    _impl->map_builder->SaveTrajectory(path);
    return true;
}

bool AirSLAMInterface::SaveMap(
    const std::string& path) {

    if (!_impl || !_impl->map_builder || _impl->stopped) {
        return false;
    }

    _impl->map_builder->SaveMap(path);
    return true;
}

void AirSLAMInterface::Stop() {
    if (_impl) {
        _impl->Stop();
    }
}
