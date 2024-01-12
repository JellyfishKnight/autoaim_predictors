// created by liuhan on 2023/9/15
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#include "VehicleObserver.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <autoaim_interfaces/msg/detail/armor__struct.hpp>
#include <cfloat>
#include <rclcpp/logging.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>

namespace helios_cv {
VehicleObserver::VehicleObserver(const VOParams& params) {
    params_ =  params;
    find_state_ = LOST;
}

void VehicleObserver::init() {
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
        Eigen::MatrixXd f(9, 9);
        //  xc   yc   zc   yaw  vxc  vyc  vzc  vyaw r    
        f <<1,   0,   0,   0,   dt_, 0,   0,   0,   0, 
            0,   1,   0,   0,   0,   dt_, 0,   0,   0, 
            0,   0,   1,   0,   0,   0,   dt_, 0,   0, 
            0,   0,   0,   1,   0,   0,   0,   dt_, 0,   
            0,   0,   0,   0,   1,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   1,   0,   0,   0, 
            0,   0,   0,   0,   0,   0,   1,   0,   0, 
            0,   0,   0,   0,   0,   0,   0,   1,   0,   
            0,   0,   0,   0,   0,   0,   0,   0,   1;
        return f;
    };
    auto h = [](const Eigen::VectorXd & x) {
        Eigen::VectorXd z(4);
        double xc = x(0), yc = x(1), yaw = x(3), r = x(8);
        z(0) = xc - r * cos(yaw);  // xa
        z(1) = yc - r * sin(yaw);  // ya
        z(2) = x(2);               // za
        z(3) = x(3);               // yaw  
        return z;
    };
    auto j_h = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(4, 9);
        double yaw = x(3), r = x(8);
        //  xc   yc   zc   yaw         vxc  vyc  vzc  vyaw   r          
        h <<1,   0,   0,   r*sin(yaw), 0,   0,   0,   0,   -cos(yaw),
            0,   1,   0,   -r*cos(yaw),0,   0,   0,   0,   -sin(yaw),
            0,   0,   1,   0,          0,   0,   0,   0,   0,        
            0,   0,   0,   1,          0,   0,   0,   0,   0;
        return h;
    };
    // update_Q - process noise covariance matrix
    auto update_Q = [this]() -> Eigen::MatrixXd {
        double t = dt_, x = params_.ekf_params.sigma2_q_xyz, 
                y = params_.ekf_params.sigma2_q_yaw, 
                r = params_.ekf_params.sigma2_q_r;
        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * x, q_vy_vy = pow(t, 2) * y;
        double q_r = pow(t, 4) / 4 * r;
        Eigen::MatrixXd q(9, 9);
        //  xc      yc      zc      yaw     vxc     vyc     vzc     vyaw    r  
        q <<q_x_x,  0,      0,      0,      q_x_vx, 0,      0,      0,      0,
            0,      q_x_x,  0,      0,      0,      q_x_vx, 0,      0,      0,  
            0,      0,      q_x_x,  0,      0,      0,      q_x_vx, 0,      0,
            0,      0,      0,      q_y_y,  0,      0,      0,      q_y_vy, 0, 
            q_x_vx, 0,      0,      0,      q_vx_vx,0,      0,      0,      0,
            0,      q_x_vx, 0,      0,      0,      q_vx_vx,0,      0,      0,
            0,      0,      q_x_vx, 0,      0,      0,      q_vx_vx,0,      0,
            0,      0,      0,      q_y_vy, 0,      0,      0,      q_vy_vy,0,
            0,      0,      0,      0,      0,      0,      0,      0,      q_r;
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

autoaim_interfaces::msg::Target VehicleObserver::predict_target(autoaim_interfaces::msg::Armors armors, double dt) {
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
            if (armor.distance_to_image_center < min_distance) {
                min_distance = armor.distance_to_image_center;
                tracking_armor_ = armor;
            }
        }
        target_xyz_ = Eigen::Vector3d(tracking_armor_.pose.position.x, tracking_armor_.pose.position.y, tracking_armor_.pose.position.z);
        target_yaw_ = orientation2yaw(tracking_armor_.pose.orientation);
        armor_type_ = tracking_armor_.type == 0 ? "SMALL" : "LARGE";
        reset_kalman();
        tracking_number_ = tracking_armor_.number;
        find_state_ = DETECTING;
        update_target_type(tracking_armor_);
    } else {
        // get observation
        armor_predict(armors);
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
            target.radius_2 = last_r_;
            target.dz = dz_;
            target.id = tracking_number_;
            target.tracking = true;
            target.armor_type = armor_type_;
            target.armors_num = static_cast<int>(target_type_) + 2;
        } else {
            target.tracking = false;
        }
        // Update threshold of temp lost 
        params_.max_lost = static_cast<int>(params_.lost_time_thresh / dt_ * 4 / target.armors_num);
    }
    return target;
}

void VehicleObserver::armor_predict(autoaim_interfaces::msg::Armors armors) {
    bool matched = false;
    Eigen::VectorXd prediction = ekf_.Predict();
    // Use KF prediction as default target state if no matched armor is found
    target_state_ = prediction;
    if (!armors.armors.empty()) {
        auto armor_position = state2position(target_state_);
        double yaw_diff = DBL_MAX;
        double min_position_error = DBL_MAX;

        target_xyz_ = Eigen::Vector3d(tracking_armor_.pose.position.x, tracking_armor_.pose.position.y, tracking_armor_.pose.position.z);
        target_yaw_ = orientation2yaw(tracking_armor_.pose.orientation);
        armor_type_ = tracking_armor_.type == 0 ? "SMALL" : "LARGE";

        for (const auto& armor : armors.armors) {
            // Only consider armors with the same id
            if (armor.number == tracking_number_) {
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
        // Check if the tracking armor is one of last tracking target
        auto match_info = match_armor(tracking_armor_, prediction);
        if (match_info.first == false) {
            RCLCPP_WARN(logger_, "No matched armor found!");
        } else {
            if (match_info.second == 0) {
                // Matched armor found, update ekf
                matched = true;
                auto position = tracking_armor_.pose.position;
                double measured_yaw = orientation2yaw(tracking_armor_.pose.orientation);
                Eigen::Vector4d measurement(position.x, position.y, position.z, measured_yaw);
                target_state_ = ekf_.Correct(measurement);
            } else if (match_info.second == 1 || match_info.second == 3) {
                // Matched armor found, but is not facing armor
                // Take this situation as target spinning and armor jumped
                armor_jump(tracking_armor_);
            } else {
                RCLCPP_WARN(logger_, "Measurement is wrong, drop it");
                RCLCPP_DEBUG(logger_, "Yaw Diff : %f", yaw_diff);
                RCLCPP_DEBUG(logger_, "Position Diff : %f", min_position_error);
            }
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

std::pair<bool, int> VehicleObserver::match_armor(autoaim_interfaces::msg::Armor& armor, const Eigen::VectorXd& prediction) {
    bool matched = false;
    int matched_index = 0;
    auto position = armor.pose.position;
    Eigen::Vector3d position_vec(position.x, position.y, position.z);
    // Get armor sequence
    std::vector<Eigen::Vector3d> armor_position_sequence;
    std::vector<double> armor_yaw_sequence;
    static bool is_current_pair = false;
    double xc = target_state_(0);
    double yc = target_state_(1);
    double zc = target_state_(2);
    double yaw = target_state_(3);
    double v_yaw = target_state_(7);
    double radius_1 = target_state_(8);
    double radius_2 = last_r_;
    int a_n = static_cast<int>(target_type_) + 2;
    for (int i = 0; i < a_n; i++) {
        Eigen::Vector3d position;
        double r;
        double tmp_yaw = yaw + i * (2 * M_PI / a_n);
        // Only 4 armors has 2 radius and height
        if (a_n == 4) {
            r = is_current_pair ? radius_1 : radius_2;
            position[2] = zc + (is_current_pair ? 0 : dz_);
            is_current_pair = !is_current_pair;
        } else {
            r = radius_1;
            position[2] = zc;
        }
        position[0] = xc - r * cos(tmp_yaw);
        position[1] = yc - r * sin(tmp_yaw);
        armor_position_sequence.emplace_back(position);
        armor_yaw_sequence.emplace_back(tmp_yaw);
    }
    // Match armor, get min position diff armor first.
    // Cause position is more confident than orientation
    double min_position_diff = DBL_MAX;
    Eigen::Vector3d armor_position(armor.pose.position.x, 
                                    armor.pose.position.y, 
                                    armor.pose.position.z);
    for (int i = 0; i < a_n; i++) {
        double diff = (armor_position_sequence[i] - armor_position).norm();
        if (diff < min_position_diff) {
            min_position_diff = diff;
            matched_index = i;
        }
    }
    // Take large position diff with every armor as wrong detect
    if (min_position_diff > params_.max_match_distance) {
        return {false, -1};
    }
    // Check if tracking armor is the back armor, 
    // if it is, take this as the position error, fix it with prediction position
    if (matched_index == 2) {
        RCLCPP_WARN(logger_, "armor position is wrong, fix it to facing armor");
        armor.pose.position.x = armor_position_sequence[0][0];
        armor.pose.position.y = armor_position_sequence[0][1];
        armor.pose.position.z = armor_position_sequence[0][2];
        tf2::Quaternion tf_q;
        tf2::fromMsg(armor.pose.orientation, tf_q);
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
        tf_q.setRPY(roll, pitch, armor_yaw_sequence[0]);
        return {true, 0};
    }
    // Test yaw diff, if yaw diff is large than we expected, take this as a inaccurate result, 
    // then fix it with the predition yaw
    double yaw_diff = std::abs(armor_yaw_sequence[matched_index] - orientation2yaw(armor.pose.orientation));
    if (yaw_diff < params_.max_match_yaw_diff) {
        RCLCPP_WARN(logger_, "armor yaw is in wrong, fix it to %f", armor_yaw_sequence[matched_index]);
        tf2::Quaternion tf_q;
        tf2::fromMsg(armor.pose.orientation, tf_q);
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
        tf_q.setRPY(roll, pitch, armor_yaw_sequence[matched_index]);
        armor.pose.orientation = tf2::toMsg(tf_q);
    }
    return {matched, matched_index};
}

std::vector<double> VehicleObserver::get_state() const {
    std::vector<double> state;
    for (int i = 0; i < target_state_.size(); i++) {
        state.push_back(target_state_(i));
    }
    state.push_back(last_r_);
    state.push_back(dz_);
    return state;
}

void VehicleObserver::update_target_type(const autoaim_interfaces::msg::Armor& armor) {
    if (armor.type == static_cast<int>(ArmorType::LARGE) && (tracking_number_ == "3" || tracking_number_ == "4" || tracking_number_ == "5")) {
        target_type_ = TargetType::BALANCE;
    } else if (tracking_number_ == "outpost") {
        target_type_ = TargetType::OUTPOST;
    } else {
        target_type_ = TargetType::NORMAL;
    }
}

double VehicleObserver::orientation2yaw(const geometry_msgs::msg::Quaternion& orientation) {
    // Get armor yaw
    tf2::Quaternion tf_q;
    tf2::fromMsg(orientation, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    // RCLCPP_INFO(logger_, "roll %f pitch %f yaw %f", roll, pitch, yaw);
    // Make yaw change continuous
    yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
    last_yaw_ = yaw;
    return yaw;
}

void VehicleObserver::reset_kalman() {
    RCLCPP_DEBUG(logger_, "Kalman Refreshed!");
    // 初始化卡尔曼滤波器
    double armor_x = tracking_armor_.pose.position.x;
    double armor_y = tracking_armor_.pose.position.y;
    double armor_z = tracking_armor_.pose.position.z;
    last_yaw_ = 0;
    Eigen::VectorXd target(9);
    double yaw = orientation2yaw(tracking_armor_.pose.orientation);
    double r = 0.26;
    double car_center_x = armor_x + r * cos(yaw);
    double car_center_y = armor_y + r * sin(yaw);
    double car_center_z = armor_z;
    dz_ = 0;
    last_r_ = r;
    target << car_center_x, car_center_y, car_center_z, yaw, 0, 0, 0, 0, r;
    target_state_ = target;
    ekf_.setState(target_state_);
}

void VehicleObserver::armor_jump(const autoaim_interfaces::msg::Armor tracking_armor) {
    double yaw = orientation2yaw(tracking_armor.pose.orientation);
    update_target_type(tracking_armor);
    target_state_(3) = yaw;
    if (target_type_ == TargetType::NORMAL) {
        dz_ = target_state_(2) - tracking_armor.pose.position.z;
        target_state_(2) = tracking_armor.pose.position.z;
        std::swap(target_state_(8), last_r_);
    }
    RCLCPP_INFO(logger_, "Armor Updated!");
    auto position = tracking_armor.pose.position;
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
    ekf_.setState(target_state_);
}

Eigen::Vector3d VehicleObserver::state2position(const Eigen::VectorXd& state) {
    double car_center_x = state(0);
    double car_center_y = state(1);
    double car_center_z = state(2);
    double r = state(8), yaw = state(3);
    double armor_x = car_center_x - r * cos(yaw);
    double armor_y = car_center_y - r * sin(yaw);
    return Eigen::Vector3d(armor_x, armor_y, car_center_z);
}


} // namespace helios_cv