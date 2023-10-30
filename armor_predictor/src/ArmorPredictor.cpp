// created by liuhan on 2023/9/15
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#include "ArmorPredictor.hpp"
#include "Armor.hpp"
#include <angles/angles.h>
#include <complex>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace helios_cv {
ArmorPredictor::ArmorPredictor(std::shared_ptr<helios_autoaim::Params> params) {
    params_ =  params;
}

void ArmorPredictor::set_cam_info(sensor_msgs::msg::CameraInfo::SharedPtr cam_info) {}

void ArmorPredictor::init_predictor(std::shared_ptr<helios_autoaim::Params> params, tf2_ros::Buffer::SharedPtr tf_buffer) {
    // receive tf buffer
    tf2_buffer_ = tf_buffer;
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
        double t = dt_, x = params_->predictor.armor_predictor.ekf.sigma2_q_xyz, 
                y = params_->predictor.armor_predictor.ekf.sigma2_q_yaw, 
                r = params_->predictor.armor_predictor.ekf.sigma2_q_r;
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
    auto update_R = [this](const Eigen::VectorXd &z) -> Eigen::MatrixXd {
        Eigen::DiagonalMatrix<double, 4> r;
        double x = params_->predictor.armor_predictor.ekf.r_xyz_factor;
        r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), params_->predictor.armor_predictor.ekf.r_yaw;
        return r;
    };
    Eigen::DiagonalMatrix<double, 9> p0;
    p0.setIdentity();
    ekf_ = ExtendedKalmanFilter{f, h, j_f, j_h, update_Q,  update_R, p0};

}


helios_rs_interfaces::msg::Target ArmorPredictor::predict_target(helios_rs_interfaces::msg::Armors armors, const rclcpp::Time& now) {
    // 时间系统搭建 
    dt_ = now.seconds() - time_predictor_start_;
    time_predictor_start_ = now.seconds();
    // 回传数据
    helios_rs_interfaces::msg::Target target;
    target.header.stamp = now;
    target.header.frame_id = params_->predictor.armor_predictor.target_frame;
    // 分状态讨论：处于丢失状态，时若第一次找到装甲板，则重新初始化卡尔曼滤波器并且切换状态和发送位置。
    // 处于其他状态时，另作讨论
    if (find_state_ == LOST) {
        // 选取最优装甲板（距离）
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
        armor_type_ = tracking_armor_.type;
        reset_kalman();
        tracking_number_ = tracking_armor_.number;
        find_state_ = DETECTING;
        update_target_type(tracking_armor_);
    } else {
        // 装甲板预测
        armor_predict(armors);
        params_->predictor.armor_predictor.max_lost = static_cast<int>(params_->predictor.armor_predictor.lost_time_thres_ / dt_);
        if (find_state_ == TRACKING || find_state_ == TEMP_LOST) {
            // 数据发送
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
            target.armors_num = static_cast<int>(target_type_) + 2;
        } else {
            target.tracking = false;
        }
    }
    return target;
}

void ArmorPredictor::armor_predict(helios_rs_interfaces::msg::Armors armors) {
    // coordinate transform
    for (auto & armor : armors.armors) {
        geometry_msgs::msg::PointStamped point;
        point.header = armors.header;
        point.point = armor.pose.position;
        try {
            if (tf2_buffer_->canTransform(params_->predictor.armor_predictor.target_frame, armors.header.frame_id, armors.header.stamp)) {
                armor.pose = tf2_buffer_->transform(armor.pose, params_->predictor.armor_predictor.target_frame);
                armor.pose.position = point.point;
            }
        } catch (tf2::TransformException & ex) {
            RCLCPP_WARN(logger_, "Failed to transform armor pose: %s", ex.what());
            return ;
        }
    }
    Eigen::VectorXd prediction = ekf_.Predict();
    bool matched = false;
    // if there is no match, use directly prediction
    target_state_ = prediction;
    // if there is a match, use the position of match armor
    if (!armors.armors.empty()) {
        helios_rs_interfaces::msg::Armor same_id_armor;
        int same_id_armors_count = 0;
        auto armor_position = state2position(target_state_);
        double yaw_diff = DBL_MAX;
        double min_position_error = DBL_MAX;

        target_xyz_ = Eigen::Vector3d(tracking_armor_.pose.position.x, tracking_armor_.pose.position.y, tracking_armor_.pose.position.z);
        target_yaw_ = orientation2yaw(tracking_armor_.pose.orientation);
        armor_type_ = tracking_armor_.type;
        // debug point
        // geometry_msgs::msg::PointStamped ps;
        // ps.header.frame_id = target_frame_;
        // ps.header.stamp = this->now();
        // ps.point.x = prediction(0);
        // ps.point.y = prediction(1);
        // ps.point.z = prediction(2);
        // point_pub_->publish(ps);
        for (const auto& armor : armors.armors) {
            if (armor.number == tracking_number_) {
                same_id_armors_count++;
                same_id_armor = armor;

                auto p = armor.pose.position;
                Eigen::Vector3d position_vec(p.x, p.y, p.z);
                double position_diff = (armor_position - position_vec).norm();
                // RCLCPP_WARN(this->get_logger(), "Position Diff : %f", position_diff);
                if (position_diff < min_position_error) {
                    min_position_error = position_diff;
                    yaw_diff = abs(orientation2yaw(armor.pose.orientation) - prediction(3));
                    tracking_armor_ = armor;
                }
            }
        }
        // 如果最小距离误差满足阈值，就不使用预测值
        if (min_position_error < params_->predictor.armor_predictor.max_match_distance && yaw_diff < params_->predictor.armor_predictor.max_match_yaw_diff) {
            // 如果最小距离误差满足阈值，就使用预测值
            matched = true;
            auto position = tracking_armor_.pose.position;
            double measured_yaw = orientation2yaw(tracking_armor_.pose.orientation);
            Eigen::Vector4d measurement(position.x, position.y, position.z, measured_yaw);
            target_state_ = ekf_.Correct(measurement);
        } else if (same_id_armors_count == 1 && yaw_diff > params_->predictor.armor_predictor.max_match_yaw_diff) {
            armor_jump(same_id_armor);
            // RCLCPP_WARN(this->get_logger(), "Yaw Diff : %f", yaw_diff);
            // RCLCPP_WARN(this->get_logger(), "Position Diff : %f", min_position_error);
            // RCLCPP_WARN(this->get_logger(), "Same ID Number: %d", same_id_armors_count);
        } else {
            RCLCPP_WARN(logger_, "No matched armor found!");
            RCLCPP_WARN(logger_, "Yaw Diff : %f", yaw_diff);
            RCLCPP_WARN(logger_, "Position Diff : %f", min_position_error);
            RCLCPP_WARN(logger_, "Same ID Number: %d", same_id_armors_count);
        }
    }
    // 防止半径扩散
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
            if (detect_cnt_ > params_->predictor.armor_predictor.max_detect) {
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
            if (lost_cnt_ > params_->predictor.armor_predictor.max_lost) {
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

void ArmorPredictor::set_params(std::shared_ptr<helios_autoaim::Params> params) {
    params_ = params;
}

std::vector<double> ArmorPredictor::get_state() const {
    std::vector<double> state;
    for (int i = 0; i < target_state_.size(); i++) {
        state.push_back(target_state_(i));
    }
    state.push_back(last_r_);
    state.push_back(dz_);
    return state;
}

void ArmorPredictor::update_target_type(const helios_rs_interfaces::msg::Armor& armor) {
    if (armor.type == static_cast<int>(ArmorType::LARGE) && (tracking_number_ == "3" || tracking_number_ == "4" || tracking_number_ == "5")) {
        target_type_ = TargetType::BALANCE;
    } else if (tracking_number_ == "outpost") {
        target_type_ = TargetType::OUTPOST;
    } else {
        target_type_ = TargetType::NORMAL;
    }
}

double ArmorPredictor::orientation2yaw(const geometry_msgs::msg::Quaternion& orientation) {
    // Get armor yaw
    tf2::Quaternion tf_q;
    tf2::fromMsg(orientation, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    // Make yaw change continuous
    yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
    last_yaw_ = yaw;
    return yaw;
}

void ArmorPredictor::reset_kalman() {
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

void ArmorPredictor::armor_jump(const helios_rs_interfaces::msg::Armor tracking_armor) {
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
    if ((current_position - infer_position).norm() > params_->predictor.armor_predictor.max_match_distance) {
        double r = target_state_(8);
        target_state_(0) = position.x + r * cos(yaw);
        target_state_(1) = position.y + r * sin(yaw);
        target_state_(2) = position.z;
        target_state_(4) = 0;
        target_state_(5) = 0;
        target_state_(6) = 0;
        target_state_(7) = 0;
        RCLCPP_ERROR(logger_, "Reset State!");
    }
    ekf_.setState(target_state_);
}

Eigen::Vector3d ArmorPredictor::state2position(const Eigen::VectorXd& state) {
    double car_center_x = state(0);
    double car_center_y = state(1);
    double car_center_z = state(2);
    double r = state(8), yaw = state(3);
    double armor_x = car_center_x - r * cos(yaw);
    double armor_y = car_center_y - r * sin(yaw);
    return Eigen::Vector3d(armor_x, armor_y, car_center_z);
}


} // namespace helios_cv