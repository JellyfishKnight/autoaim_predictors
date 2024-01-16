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

typedef struct BaseObserverParams {
public:
    int max_lost;
    int max_detect;
    double max_match_distance;
    double max_match_yaw_diff;
    double lost_time_thresh;
    std::string target_frame;
}BaseObserverParams;

class BaseObserver {
public:
    BaseObserver() = default;

    virtual autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt) = 0;

    virtual void reset_kalman() = 0;

    TargetType target_type_;
protected:
    virtual void update_target_type(const autoaim_interfaces::msg::Armor& armor) {
        if (armor.type == static_cast<int>(ArmorType::LARGE) && (tracking_number_ == "3" || tracking_number_ == "4" || tracking_number_ == "5")) {
            target_type_ = TargetType::BALANCE;
        } else if (tracking_number_ == "outpost") {
            target_type_ = TargetType::OUTPOST;
        } else {
            target_type_ = TargetType::NORMAL;
        }
    }

    virtual void armor_jump(const autoaim_interfaces::msg::Armor same_id_armor) = 0;

    virtual Eigen::Vector3d state2position(const Eigen::VectorXd& state) = 0;

    virtual void track_armor(autoaim_interfaces::msg::Armors armors) = 0;

    virtual void init() = 0;

    int find_state_;

    autoaim_interfaces::msg::Armor tracking_armor_;
    autoaim_interfaces::msg::Armor last_armor_;

    std::string armor_type_;

    std::string tracking_number_;

    int lost_cnt_ = 0;
    int detect_cnt_ = 0;

    double dt_ = 0.008f;
    
    // 目标车辆状态
    Eigen::VectorXd target_state_;
};




} // namespace helios_cv