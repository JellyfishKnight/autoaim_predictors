// created by liuhan on 2024/1/16
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

#include <rclcpp/rclcpp.hpp>

#include "StandardObserver.hpp"

namespace helios_cv {

class BalanceObserver : public StandardObserver {
public:
    BalanceObserver();

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt) override;

    void reset_kalman() override;


private:

};



} // namespace helios_cv