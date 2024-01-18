// created by liuhan on 2023/1/16
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#include "BalanceObserver.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <angles/angles.h>
#include <autoaim_interfaces/msg/detail/armor__struct.hpp>
#include <autoaim_utilities/Armor.hpp>
#include <cfloat>
#include <cmath>
#include <rclcpp/logging.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>

namespace helios_cv {
BalanceObserver::BalanceObserver(const BalanceObserverParams& params) {
    params_ =  params;
    find_state_ = LOST;
    init();
}

void BalanceObserver::init() {
    // init kalman filter
    auto f = [this](const Eigen::VectorXd & x) {
        Eigen::VectorXd x_new = x;
        x_new(0) += x(4) * dt_;
        x_new(1) += x(5) * dt_;
        x_new(2) += x(6) * dt_;
        x_new(3) += x(7) * dt_;
        return x_new;
    };
    auto j_f = [this](const Eigen::VectorXd &) {
        Eigen::MatrixXd f(8, 8);
        auto X = target_state_;
        //  xc  yc v                   z  r  yaw                           vyaw
        f << 1, 0, cos(X(5, 0)) * dt_, 0, 0,-X(2, 0) * sin(X(5, 0)) * dt_, 0,
             0, 1, sin(X(5, 0)) * dt_, 0, 0, X(2, 0) * cos(X(5, 0)) * dt_, 0,
             0, 0, 1,                  0, 0, 0,                            0,
             0, 0, 0,                  1, 0, 0,                            0,
             0, 0, 0,                  0, 1, 0,                            0,
             0, 0, 0,                  0, 0, 1,                            dt_,
             0, 0, 0,                  0, 0, 0,                            1;
        return f;
    };
    auto h = [](const Eigen::VectorXd & X) {
        Eigen::VectorXd z(4);
        z(0) = X(0) + X(4) * cos(X(5) + i * M_PI)  
        z(1) = X(1) + X(4) * sin(X(5) + i * M_PI);  
        z(2) = X(3);               
        z(3) = X(5) + i * M_PI;                
        return z;
    };
    auto j_h = [](const Eigen::VectorXd & X) {
        Eigen::MatrixXd h(4, 7);
        //   xc yc v  zc r                     yaw                          vyaw
        h << 1, 0, 0, 0, cos(X(5) + i * M_PI),-X(4) * sin(X(5) + i * M_PI), 0,
             0, 1, 0, 0, sin(X(5) + i * M_PI), X(4) * cos(X(5) + i * M_PI), 0,
             0, 0, 0, 1, 0,                    0,                           0,
             0, 0, 0, 0, 0,                    1,                           0;
        return h;
    };
    // update_Q - process noise covariance matrix
    auto update_Q = [this]() -> Eigen::MatrixXd {
        double dd = params_.ekf_params.sigma2_q_xyz, 
                da = params_.ekf_params.sigma2_q_yaw, 
                dr = params_.ekf_params.sigma2_q_r;
        double t5 = pow(dt_, 5) / 5;
        double t4 = pow(dt_, 4) / 4;
        double t3 = pow(dt_, 3) / 3;
        double t2 = pow(dt_, 2) / 2;
        double theta = target_state_(5, 0);
        double v = target_state_(2, 0);
        Eigen::MatrixXd q(9, 9);
        double cos2 = pow(cos(theta), 2);
        double sin2 = pow(sin(theta), 2);
        double sincos = sin(theta) * cos(theta);
        //   xc                                                       yc                                                       v                     z         r          yaw                           vyaw
        q << t5 * da * pow(v, 2) * sin2 / 4 + t3 * dd * cos2,         sin(theta) * t3 * dd - pow(v, 2) * sincos * t5 * da / 4, t2 * dd * cos(theta), 0,        0,        -t4 * da * v * sin(theta) / 2,-t3 * da * v * sin(theta) / 2,
             sin(theta) * t3 * dd - pow(v, 2) * sincos * t5 * da / 4, t3 * dd * sin2 + t5 * da * pow(v, 2) * cos2 / 4,         t2 * dd * sin(theta), 0,        0,         t4 * da * v * cos(theta) / 2, t3 * da * v * cos(theta) / 2,
             t2 * dd * cos(theta),                                    t2 * dd * sin(theta),                                    dt_ * dd,            0,        0,         0,                            0,
             0,                                                       0,                                                       0,                    dt_ *dd, 0,         0,                            0,
             0,                                                       0,                                                       0,                    0,        dt_ * dr, 0,                            0,
             -t4 * da * v * sin(theta) / 2,                           t4 * da * v * cos(theta) / 2,                            0,                    0,        0,         t3 * da,                      t2 * da,
             -t3 * da * v * sin(theta) / 2,                           t3 * da * v * cos(theta) / 2,                            0,                    0,        0,         t2 * da,                      dt_ * da;
        return q;
    };
    // update_R - observation noise covariance matrix
    auto update_R = [this](const Eigen::VectorXd &z) -> Eigen::MatrixXd {
        Eigen::DiagonalMatrix<double, 4> r;
        double x = params_.ekf_params.r_xyz_factor;
        r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), params_.ekf_params.r_yaw;
        return r;
    };
    Eigen::DiagonalMatrix<double, 9> p0;
    p0.setIdentity();
    ekf_ = ExtendedKalmanFilter{f, h, j_f, j_h, update_Q,  update_R, p0};
}

autoaim_interfaces::msg::Target BalanceObserver::predict_target(autoaim_interfaces::msg::Armors armors, double dt) {
    dt_ = dt;
    if (dt_ > 0.1) {
        find_state_ = LOST;
    }
    autoaim_interfaces::msg::Target target;
    target.header.frame_id = params_.target_frame;
    target.header.stamp = armors.header.stamp;
    if (find_state_ == LOST) {
        if (armors.armors.empty()) {
            target.tracking = false;
            return target;
        }
        // Take the closet armor as target to init
        double min_distance = DBL_MAX;
        tracking_armor_ = armors.armors[0];
        for (auto armor : armors.armors) {
            if (armor.distance_to_image_center < min_distance && armor.type == 0) {
                min_distance = armor.distance_to_image_center;
                tracking_armor_ = armor;
            }
        }
        if (tracking_armor_.type == 0) {
            target.tracking = false;
            return target;
        }
        target_yaw_ = orientation2yaw(tracking_armor_.pose.orientation);
        armor_type_ = "LARGE";
        reset_kalman();
        tracking_number_ = tracking_armor_.number;
        find_state_ = DETECTING;
        update_target_type(tracking_armor_);
    } else {
        // get observation
        track_armor(armors);
        if (find_state_ == TRACKING || find_state_ == TEMP_LOST) {
            // Pack data
            target.position.x = target_state_(0);
            target.position.y = target_state_(1);
            target.position.z = target_state_(2);
            target.yaw = target_state_(3);
            target.velocity.x = target_state_(4);
            target.velocity.y = target_state_(5);
            target.velocity.z = target_state_(6);
            target.v_yaw = target_state_(7);
            target.radius_1 = target_state_(8);
            target.radius_2 = 0;
            target.dz = 0;
            target.id = tracking_number_;
            target.tracking = true;
            target.armor_type = "LARGE";
            target.armors_num = 2;
        } else {
            target.tracking = false;
        }
        // Update threshold of temp lost 
        params_.max_lost = static_cast<int>(params_.lost_time_thresh / dt_ * 4 / target.armors_num);
    }
    return target;
}

void BalanceObserver::track_armor(autoaim_interfaces::msg::Armors armors) {
    bool matched = false;
    Eigen::VectorXd prediction = ekf_.Predict();
    // Use KF prediction as default target state if no matched armor is found
    target_state_ = prediction;
    if (!armors.armors.empty()) {
        autoaim_interfaces::msg::Armor same_id_armor;
        int same_id_armors_count = 0;
        auto armor_position = state2position(target_state_);
        double yaw_diff = DBL_MAX;
        double min_position_error = DBL_MAX;

        target_yaw_ = orientation2yaw(tracking_armor_.pose.orientation);
        armor_type_ = "LARGE";
        for (const auto& armor : armors.armors) {
            // Only consider armors with the same id
            if (armor.number == tracking_number_) {
                same_id_armors_count++;
                same_id_armor = armor;
                auto p = armor.pose.position;
                Eigen::Vector3d position_vec(p.x, p.y, p.z);
                // Calculate the difference between the predicted position and the current armor position
                double position_diff = (armor_position - position_vec).norm();
                if (position_diff < min_position_error) {
                    // Find the closest armor
                    min_position_error = position_diff;
                    yaw_diff = abs(orientation2yaw(armor.pose.orientation) - prediction(3));
                    tracking_armor_ = armor;
                }
            }
        }
        // Check if the distance and yaw difference of closest armor are within the threshold
        if (min_position_error < params_.max_match_distance && yaw_diff < params_.max_match_yaw_diff) {
            // Matched armor found
            matched = true;
            auto position = tracking_armor_.pose.position;
            // Update EKF
            double measured_yaw = orientation2yaw(tracking_armor_.pose.orientation);
            Eigen::Vector4d measurement(position.x, position.y, position.z, measured_yaw);
            target_state_ = ekf_.Correct(measurement);
        } else if (same_id_armors_count == 1 && yaw_diff > params_.max_match_yaw_diff) {
            // Matched armor not found, but there is only one armor with the same id
            // and yaw has jumped, take this case as the target is spinning and armor jumped
            armor_jump(same_id_armor);
        } else {
            // No matched armor found
            RCLCPP_WARN(logger_, "No matched armor found!");
            RCLCPP_DEBUG(logger_, "Yaw Diff : %f", yaw_diff);
            RCLCPP_DEBUG(logger_, "Position Diff : %f", min_position_error);
            RCLCPP_DEBUG(logger_, "Same ID Number: %d", same_id_armors_count);
        }
    }
    // Prevent radius from spreading
    if (target_state_(8) < 0.2) {
        target_state_(8) = 0.2;
        ekf_.setState(target_state_);
    } else if (target_state_(8) > 0.4) {
        target_state_(8) = 0.4;
        ekf_.setState(target_state_);
    }
    // Update state machine
    if (find_state_ == DETECTING) {
        if (matched) {
            detect_cnt_++;
            if (detect_cnt_ > params_.max_detect) {
                detect_cnt_ = 0;
                find_state_ = TRACKING;
            }
        } else {
            detect_cnt_ = 0;
            find_state_ = LOST;
        }
    } else if (find_state_ == TRACKING) {
        if (!matched) {
            find_state_ = TEMP_LOST;
            lost_cnt_++;
        }
    } else if (find_state_ == TEMP_LOST) {
        if (!matched) {
            lost_cnt_++;
            // RCLCPP_WARN(logger_, "max lost %d, lost_cnt %d", params_.max_lost, lost_cnt_);
            if (lost_cnt_ > params_.max_lost) {
                RCLCPP_WARN(logger_, "Target lost!");
                find_state_ = LOST;
                lost_cnt_ = 0;
            }
        } else {
            find_state_ = TRACKING;
            lost_cnt_ = 0;
        }
    }
}


void BalanceObserver::reset_kalman() {
    RCLCPP_DEBUG(logger_, "Kalman Refreshed!");
    // reset kalman
    double armor_x = tracking_armor_.pose.position.x;
    double armor_y = tracking_armor_.pose.position.y;
    double armor_z = tracking_armor_.pose.position.z;
    last_yaw_ = 0;
    Eigen::VectorXd target(7);
    double yaw = orientation2yaw(tracking_armor_.pose.orientation);
    double r = 0.26;
    double car_center_x = armor_x + r * cos(yaw);
    double car_center_y = armor_y + r * sin(yaw);
    double car_center_z = armor_z;
    dz_ = 0;
    last_r_ = r;
    target << car_center_x, car_center_y, 0, car_center_z, r, yaw, 0;
    target_state_ = target;
    ekf_.setState(target_state_);
}

void StandardObserver::armor_jump(const autoaim_interfaces::msg::Armor same_id_armor) {
    double yaw = orientation2yaw(same_id_armor.pose.orientation);
    update_target_type(same_id_armor);
    target_state_(3) = yaw;
    auto position = same_id_armor.pose.position;
    Eigen::Vector3d current_position(position.x, position.y, position.z);
    Eigen::Vector3d infer_position = state2position(target_state_);
    // if the distance between current position and infer position is too large, then the state is wrong
    if ((current_position - infer_position).norm() > params_.max_match_distance) {
        double r = target_state_(8);
        target_state_(0) = position.x + r * cos(yaw); // xc
        target_state_(1) = position.y + r * sin(yaw); // yc
        target_state_(2) = position.z;                // zc
        target_state_(4) = 0;                         // vxc
        target_state_(5) = 0;                         // vyc
        target_state_(6) = 0;                         // vzc
        RCLCPP_WARN(logger_, "Reset State!");
    }
    RCLCPP_INFO(logger_, "Armor Updated!");
    ekf_.setState(target_state_);
}

Eigen::Vector3d BalanceObserver::state2position(const Eigen::VectorXd& state) {
    double car_center_x = state(0);
    double car_center_y = state(1);
    double car_center_z = state(2);
    double r = state(8), yaw = state(3);
    double armor_x = car_center_x - r * cos(yaw);
    double armor_y = car_center_y - r * sin(yaw);
    return Eigen::Vector3d(armor_x, armor_y, car_center_z);
}


} // namespace helios_cv