// created by liuhan on 2024/1/16
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

#include <rclcpp/rclcpp.hpp>


#include "BaseObserver.hpp"


namespace helios_cv {

typedef struct StandardObserverParams : public BaseObserverParams {
    typedef struct EKFParams {
        double sigma2_q_xyz;
        double sigma2_q_vxyz;
        double sigma2_q_yaw;
        double sigma2_r_xyz;
        double sigma2_r_vxyz;
        double sigma2_r_yaw;
    } EKFParams;
    EKFParams ekf_params;
    double max_match_yaw_diff;
}StandardObserverParams;


class StandardObserver : public BaseObserver {
public:
    StandardObserver(const StandardObserverParams& params);

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt) override;

    void reset_kalman() override;

private:
    StandardObserverParams params_;

    /**
     * @brief 
     * 
     * @param armors 
     */
    void track_armor(autoaim_interfaces::msg::Armors armors) override;

    // State Machine
    int lost_cnt_ = 0;
    int detect_cnt_ = 0;
    //
    std::string tracking_number_;
    // 识别到的目标点
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

    double orientation2yaw(const geometry_msgs::msg::Quaternion& orientation);

    void armor_jump(const autoaim_interfaces::msg::Armor same_id_armor) override;

    Eigen::Vector3d state2position(const Eigen::VectorXd& state) override;

    // Logger
    rclcpp::Logger logger_ = rclcpp::get_logger("StandardObserver");
};




} // namespace helios_cv