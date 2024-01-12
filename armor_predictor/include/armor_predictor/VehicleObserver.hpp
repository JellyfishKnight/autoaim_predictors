// created by liuhan on 2023/9/15
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

#include <rclcpp/rclcpp.hpp>
#include <angles/angles.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "autoaim_utilities/ExtendedKalmanFilter.hpp"
#include "autoaim_utilities/Armor.hpp"

#include "sensor_msgs/msg/camera_info.hpp"
#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/target.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <utility>

namespace helios_cv {

typedef struct VehicleObserverParams{
    std::string target_frame;
    typedef struct EKFParams{
        double sigma2_q_xyz;
        double sigma2_q_yaw;
        double sigma2_q_r;
        double r_xyz_factor;
        double r_yaw;
    }EKFParams;
    EKFParams ekf_params;
    int max_lost;
    int max_detect;
    double max_match_distance;
    double max_match_yaw_diff;
    double lost_time_thresh;
}VOParams;

class VehicleObserver {
public:
    VehicleObserver(const VOParams& params);

    void init();

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt);

    std::vector<double> get_state() const;

    TargetType target_type_;
    int find_state_;
private:
    /**
     * @brief 
     * 
     * @param armors 
     */
    void armor_predict(autoaim_interfaces::msg::Armors armors);

    VOParams params_;

    // State Machine
    int lost_cnt_ = 0;
    int detect_cnt_ = 0;
    //
    std::string tracking_number_;
    // 识别到的目标点
    cv::Point2f target_point;
    // 上一次的装甲板状态
    double last_yaw_ = 0;
    double last_y_ = 0;
    double last_r_ = 0;
    double target_yaw_ = 0;
    std::string armor_type_;
    autoaim_interfaces::msg::Armor last_armor_;

    autoaim_interfaces::msg::Armor tracking_armor_;
    // 目标车辆状态
    Eigen::VectorXd target_state_;

    // kalman utilities
    double dz_;
    double dt_ = 0.008f;
    ExtendedKalmanFilter ekf_;

    void update_target_type(const autoaim_interfaces::msg::Armor& armor);

    double orientation2yaw(const geometry_msgs::msg::Quaternion& orientation);

    std::pair<bool, int> match_armor(autoaim_interfaces::msg::Armor& armor, const Eigen::VectorXd& prediction);

    void reset_kalman();

    void armor_jump(const autoaim_interfaces::msg::Armor tracking_armor);

    Eigen::Vector3d state2position(const Eigen::VectorXd& state);

    // Logger
    rclcpp::Logger logger_ = rclcpp::get_logger("VehicleObserver");
};

} // namespace helios_cv