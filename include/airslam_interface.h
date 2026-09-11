#ifndef AIRSLAM_INTERFACE_H_
#define AIRSLAM_INTERFACE_H_

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>

class AirSLAMInterface {
public:
    AirSLAMInterface(const std::string& config_path,
                     const std::string& model_dir);

    ~AirSLAMInterface();

    // Feed one stereo frame to AirSLAM.
    //
    // left:
    //   Left infrared camera image.
    //
    // right:
    //   Right infrared camera image.
    //
    // timestamp:
    //   Capture timestamp in seconds.
    //
    // Returns false if the frame cannot be processed.
    bool ProcessStereoFrame(const cv::Mat& left,
                            const cv::Mat& right,
                            double timestamp);

    // Current camera pose.
    Eigen::Matrix4d GetCurrentPose() const;

    // Save the current trajectory.
    bool SaveTrajectory(const std::string& path);

    // Save the current map.
    bool SaveMap(const std::string& path);

    // Stop AirSLAM cleanly.
    void Stop();

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

#endif  // AIRSLAM_INTERFACE_H_
