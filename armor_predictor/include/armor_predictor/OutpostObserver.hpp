// created by liuhan on 2024/1/16
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

#include <rclcpp/logger.hpp>
#include <rclcpp/rclcpp.hpp>

#include "BaseObserver.hpp"
#include "StandardObserver.hpp"

namespace helios_cv {

typedef struct OutpostObserverParams : public BaseObserverParams {
    typedef struct DCMParams {
        double sigma2_q_yaw;
        double sigma2_q_xyz;
        double r_xyz_factor;
        double r_yaw_factor;
    } DDMParams;
    DDMParams ekf_params;

    OutpostObserverParams(
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
}OutpostObserverParams;


class OutpostObserver : public StandardObserver {
public:
    OutpostObserver(const OutpostObserverParams& params);

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt) final;

    void reset_kalman() final;

    void set_params(void *params) final;

private:

    void track_armor(autoaim_interfaces::msg::Armors armors) final;

    void armor_jump(const autoaim_interfaces::msg::Armor same_id_armor) final;

    Eigen::Vector3d state2position(const Eigen::VectorXd& state) final;

    void init() final;

    double radius_ = 0.26;

    OutpostObserverParams params_;

    rclcpp::Logger logger_ = rclcpp::get_logger("OutpostObserver");
};

} // namespace helios_cv