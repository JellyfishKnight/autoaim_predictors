// created by liuhan on 2024/1/16
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

#include <rclcpp/rclcpp.hpp>

#include "BaseObserver.hpp"

namespace helios_cv {

class BalanceObserver : public BaseObserver {
public:
    BalanceObserver();

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt) override;

    void reset_kalman() override;


private:
    // 上一次的装甲板状态
    double last_yaw_ = 0;
    double last_y_ = 0;
    double last_r_ = 0;

};



} // namespace helios_cv