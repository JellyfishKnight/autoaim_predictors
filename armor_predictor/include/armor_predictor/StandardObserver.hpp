// created by liuhan on 2024/1/16
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>


#include "BaseObserver.hpp"


namespace helios_cv {

typedef struct StandardObserverParams : public BaseObserverParams {
    typedef struct DDMParams {
        double sigma2_q_xyz;
        double sigma2_q_yaw;
        double sigma2_q_r;
        double r_xyz_factor;
        double r_yaw;
    } DDMParams;
    DDMParams ekf_params;

    StandardObserverParams(
        int max_lost,
        int max_detect,
        double max_match_distance,
        double max_match_yaw_diff,
        double lost_time_thresh,
        std::string target_frame,
        DDMParams ekf_params
    ) : BaseObserverParams(max_lost, max_detect, max_match_distance, max_match_yaw_diff,
                             lost_time_thresh, std::move(target_frame)),
        ekf_params(ekf_params) {}
}StandardObserverParams;


class StandardObserver : public BaseObserver {
public:
    StandardObserver(const StandardObserverParams& params);

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt) override;

    void reset_kalman() override;

protected:
    StandardObserver() = default;

    void track_armor(autoaim_interfaces::msg::Armors armors) override;

    virtual double orientation2yaw(const geometry_msgs::msg::Quaternion& orientation);

    void armor_jump(const autoaim_interfaces::msg::Armor same_id_armor) override;

    Eigen::Vector3d state2position(const Eigen::VectorXd& state) override;

    void init() override;

    // kalman utilities

    ExtendedKalmanFilter ekf_;
    double last_yaw_ = 0;
    double last_y_ = 0;
    double last_r_ = 0;
    double target_yaw_ = 0;
    double dz_;
private:
    // Params
    std::shared_ptr<StandardObserverParams> params_;
    // Logger
    rclcpp::Logger logger_ = rclcpp::get_logger("StandardObserver");
};




} // namespace helios_cv