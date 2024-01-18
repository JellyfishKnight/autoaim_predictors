// created by liuhan on 2024/1/16
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

#include <rclcpp/rclcpp.hpp>

#include "BaseObserver.hpp"
#include "StandardObserver.hpp"

namespace helios_cv {

typedef struct BalanceObserverParams : public BaseObserverParams {    
    typedef struct DDMParams {
        double sigma2_q_xyz;
        double sigma2_q_yaw;
        double sigma2_q_r;
        double r_xyz_factor;
        double r_yaw;
    } DDMParams;
    DDMParams ekf_params;
}BalanceObserverParams;

class BalanceObserver : public StandardObserver {
public:    
    BalanceObserver(const BalanceObserverParams& params);

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt) final;

    void reset_kalman() final;

protected:
    void track_armor(autoaim_interfaces::msg::Armors armors) final;

    void armor_jump(const autoaim_interfaces::msg::Armor same_id_armor) final;

    Eigen::Vector3d state2position(const Eigen::VectorXd& state) final;

    void init() final;
private:
    BalanceObserverParams params_;

    rclcpp::Logger logger_ = rclcpp::get_logger("BalanceObserver");
};



} // namespace helios_cv