#include "ArmorPredictor.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <rclcpp/logging.hpp>

namespace helios_cv {

ArmorPredictor::ArmorPredictor(const APParams& params) {
    params_ = params;
    init_kalman();
}

void ArmorPredictor::init_kalman() {
    auto f = [this] (const Eigen::VectorXd& ) {
        Eigen::MatrixXd f(6, 6);
        //   x,   y,   z,   vx,  vy,  vz
        f << 1,   0,   0,   dt_, 0,   0, 
             0,   1,   0,   0,   dt_, 0, 
             0,   0,   1,   0,   0,   dt_, 
             0,   0,   0,   1,   0,   0, 
             0,   0,   0,   0,   1,   0, 
             0,   0,   0,   0,   0,   1;
        return f;
    };
    auto h = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(3, 6);
        //  x    y    z    vx   vy   vz  
        h <<1,   0,   0,   0,   0,   0,
            0,   1,   0,   0,   0,   0,
            0,   0,   1,   0,   0,   0;
        return h;
    };
    auto q = [this]() {
        double t = dt_, x = params_.kf_params.sigma2_q_xyz;
        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        Eigen::MatrixXd q(6, 6);
        //  xc      yc      zc       vxc     vyc     vzc     
        q <<q_x_x,  0,      0,      q_x_vx, 0,      0,      
            0,      q_x_x,  0,      0,      q_x_vx, 0,          
            0,      0,      q_x_x,  0,      0,      q_x_vx,  
            q_x_vx, 0,      0,      q_vx_vx,0,      0,       
            0,      q_x_vx, 0,      0,     q_vx_vx, 0, 
            0,      0,      q_x_vx, 0,      0,      q_vx_vx;
        return q;
    };
    auto r = [this](const Eigen::VectorXd & z) {
        Eigen::DiagonalMatrix<double, 3> r;
        double x = params_.kf_params.r_xyz_factor;
        r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]);
        return r;
    };
    Eigen::DiagonalMatrix<double, 6> p0;
    p0.setIdentity();
    kalman_filter_ = EigenKalmanFilter{f, h, q, r, p0};
}

Eigen::Vector3d ArmorPredictor::state2position(const Eigen::VectorXd& state) {
    Eigen::Vector3d position;
    position << state[0], state[1], state[2];
    return position;
}

void ArmorPredictor::update_target_type(const autoaim_interfaces::msg::Armor& armor) {
    if (armor.type == static_cast<int>(ArmorType::LARGE) && (tracking_number_ == "3" || tracking_number_ == "4" || tracking_number_ == "5")) {
        target_type_ = TargetType::BALANCE;
    } else if (tracking_number_ == "outpost") {
        target_type_ = TargetType::OUTPOST;
    } else {
        target_type_ = TargetType::NORMAL;
    }
}

void ArmorPredictor::armor_predict(autoaim_interfaces::msg::Armors armors) {
    Eigen::VectorXd prediction = kalman_filter_.predict();
    bool matched = false;
    // Use KF prediction as default target state if no matched armor is found
    target_state_ = prediction;
    if (!armors.armors.empty()) {
        autoaim_interfaces::msg::Armor same_id_armor;
        int same_id_armors_cnt = 0;
        auto armor_position = state2position(target_state_);
        double min_position_error = DBL_MAX;
        
        armor_type_ = tracking_armor_.type == 0 ? "SMALL" : "LARGE";
        for (const auto& armor : armors.armors) {
            // Only consider armors with the same id
            if (armor.number == tracking_number_) {
                same_id_armors_cnt++;
                same_id_armor = armor;
                auto p = armor.pose.position;
                Eigen::Vector3d position_vec(p.x, p.y, p.z);
                // Calculate the difference between the predicted position and the current armor position
                double position_diff = (position_vec - armor_position).norm();
                if (position_diff < min_position_error) {
                    // Find the closest armor
                    min_position_error = position_diff;
                    tracking_armor_ = armor;
                }
            }
        }
        // Check if the distance and yaw difference of closest armor are within the threshold
        if (min_position_error < params_.min_match_distance) {
            // Matched armor found
            matched = true;
            auto position = tracking_armor_.pose.position;
            // Update KF
            Eigen::Vector3d measurement(position.x, position.y, position.z);
            target_state_ = kalman_filter_.correct(measurement);
        } else if (judge_spinning(same_id_armor)) {
            // Matched armor not found, but there is only one armor with the same id
            // take this case as the target is spinning and armor jumped
            RCLCPP_INFO(logger_, "Armor updated!");
            // Shallow refresh kalman, keep velocity, refresh position
            Eigen::VectorXd target_state(6);
            target_state << same_id_armor.pose.position.x, same_id_armor.pose.position.y, same_id_armor.pose.position.z, 
                            target_state_(3), target_state_(4), target_state_(5);
            kalman_filter_.set_state(target_state);
        } else {
            // No matched armor found
            RCLCPP_WARN(logger_, "No matched armor found!");
        }
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

bool ArmorPredictor::judge_spinning(const autoaim_interfaces::msg::Armor& armor) {
    update_target_type(armor);
    auto position = armor.pose.position;
    Eigen::Vector3d current_position(position.x, position.y, position.z);
    Eigen::Vector3d infer_position = state2position(target_state_);
    // if the distance between current position and infer position is too large, then the state is wrong
    if ((current_position - infer_position).norm() > params_.max_match_distance) {
        target_state_(0) = position.x;                // x
        target_state_(1) = position.y;                // y
        target_state_(2) = position.z;                // z
        target_state_(3) = 0;                         // vxc
        target_state_(4) = 0;                         // vyc
        target_state_(5) = 0;                         // vzc
        RCLCPP_INFO(logger_, "Reset state!");
        kalman_filter_.set_state(target_state_);
        return false;
    } else {
        return true;
    }
}

autoaim_interfaces::msg::Target ArmorPredictor::predict_target(autoaim_interfaces::msg::Armors armors, double dt) {
    dt_ = dt;
    if (dt > 0.1) {
        RCLCPP_INFO(logger_, "Target Lost!");
        find_state_ = LOST;
    }
    // Msg to send back
    autoaim_interfaces::msg::Target target;
    target.header.frame_id = params_.target_frame;
    target.header.stamp = armors.header.stamp;
    if (find_state_ == LOST) {
        if (armors.armors.empty()) {
            target.tracking = false;
            return target;
        }
        // Find the nearest armor
        double min_distance = DBL_MAX;
        tracking_armor_ = armors.armors[0];
        for (auto & armor : armors.armors) {
            if (armor.distance_to_image_center < min_distance) {
                min_distance = armor.distance_to_image_center;
                tracking_armor_ = armor;
            }
        }
        target_xyz_ = Eigen::Vector3d(tracking_armor_.pose.position.x, tracking_armor_.pose.position.y, tracking_armor_.pose.position.z);
        armor_type_ = tracking_armor_.type == 0 ? "SMALL" : "LARGE";
        reset_kalman();
        tracking_number_ = tracking_armor_.number;
        find_state_ = DETECTING;
        update_target_type(tracking_armor_);
    } else {
        armor_predict(armors);
        params_.max_lost = static_cast<int>(params_.lost_time_thresh / dt_);
        if (find_state_ == TRACKING || find_state_ == TEMP_LOST) {
            // 数max_lost
            target.position.x = target_state_(0);
            target.position.y = target_state_(1);
            target.position.z = target_state_(2);
            target.velocity.x = target_state_(3);
            target.velocity.y = target_state_(4);
            target.velocity.z = target_state_(5);
            target.yaw = 0;
            target.v_yaw = 0;
            target.radius_1 = 0;
            target.radius_2 = 0;
            target.dz = 0;
            target.id = tracking_number_;
            target.tracking = true;
            target.armor_type = armor_type_;
            target.armors_num = static_cast<int>(target_type_) + 2;
        } else {
            target.tracking = false;
        }
    }
    return target;
}

void ArmorPredictor::reset_kalman() {
    RCLCPP_DEBUG(logger_, "Kalman Refreshed!");
    Eigen::VectorXd target_state(6);
    target_state << target_xyz_[0], target_xyz_[1], target_xyz_[2], 0, 0, 0;
    target_state_ = target_state;
    kalman_filter_.set_state(target_state_);
}


ArmorPredictor::~ArmorPredictor() {}

} // helios_cv