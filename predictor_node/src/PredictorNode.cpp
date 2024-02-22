// created by liuhan on 2023/10/29
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
/*
 * ██   ██ ███████ ██      ██  ██████  ███████
 * ██   ██ ██      ██      ██ ██    ██ ██
 * ███████ █████   ██      ██ ██    ██ ███████
 * ██   ██ ██      ██      ██ ██    ██      ██
 * ██   ██ ███████ ███████ ██  ██████  ███████
 */

#include "PredictorNode.hpp"
#include <armor_predictor/StandardObserver.hpp>
#include <autoaim_utilities/Armor.hpp>
#include <memory>
#include <rclcpp/logging.hpp>

namespace helios_cv {

PredictorNode::PredictorNode(const rclcpp::NodeOptions& options) : 
    rclcpp::Node("PredictorNode", options) {
    // create params
    try {
        param_listener_ = std::make_shared<ParamListener>(this->get_node_parameters_interface());
        params_ = param_listener_->get_params();
    } catch (const std::exception &e) {
        RCLCPP_FATAL(logger_, "Failed to get parameters: %s, use empty params", e.what());
    }
    init_predictors();
    // create cam info subscriber
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::SharedPtr camera_info) {
        cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
        cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
        cam_info_sub_.reset();
    });
    // create publishers and subscribers
    target_pub_ = this->create_publisher<autoaim_interfaces::msg::Target>("/predictor/target", 10);
    // init tf2 utilities
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    // Create the timer interface before call to waitForTransform,
    // to avoid a tf2_ros::CreateTimerInterfaceException exception
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
    // subscriber and filter    
    armors_sub_.subscribe(this, "/detector/armors", rmw_qos_profile_sensor_data);
    // Register a callback with tf2_ros::MessageFilter to be called when transforms are available
    tf2_filter_ = std::make_shared<tf2_filter>(
        armors_sub_, *tf2_buffer_, params_.target_frame, 10, this->get_node_logging_interface(),
        this->get_node_clock_interface(), std::chrono::duration<int>(2));
    if (params_.autoaim_mode == 0) {
        tf2_filter_->registerCallback(&PredictorNode::armor_predictor_callback, this);        
    } else {
        tf2_filter_->registerCallback(&PredictorNode::energy_predictor_callback, this);
    }
    // Register reset predictor service
    reset_predictor_service_ = this->create_service<std_srvs::srv::Trigger>("/predictor/reset", 
        [this](const std_srvs::srv::Trigger::Request::SharedPtr,
               std_srvs::srv::Trigger::Response::SharedPtr response) {
        if (params_.autoaim_mode) {
            vehicle_observer_->find_state_ = LOST;
            response->success = true;
            RCLCPP_INFO(this->get_logger(), "Tracker reset!");
        } else {
            response->success = true;
            RCLCPP_INFO(this->get_logger(), "Reset failed! In Energy mode!");
        }
        return;
    });
    std::thread([this]()->void {
        while(rclcpp::ok()) {
            if (params_.autoaim_mode != last_autoaim_mode_) {
                if (params_.autoaim_mode == 0) {
                    RCLCPP_WARN(logger_, "Change state to armor mode");
                    // reset to release running callback function
                    tf2_filter_.reset();
                    tf2_filter_ = std::make_shared<tf2_filter>(
                        armors_sub_, *tf2_buffer_, params_.target_frame, 10, this->get_node_logging_interface(),
                        this->get_node_clock_interface(), std::chrono::duration<int>(2));
                    tf2_filter_->registerCallback(&PredictorNode::armor_predictor_callback, this);   
                    last_autoaim_mode_ = 0;     
                } else {
                    RCLCPP_WARN(logger_, "Change state to energy mode");
                    // reset to release running callback function
                    tf2_filter_.reset();
                    tf2_filter_ = std::make_shared<tf2_filter>(
                        armors_sub_, *tf2_buffer_, params_.target_frame, 10, this->get_node_logging_interface(),
                        this->get_node_clock_interface(), std::chrono::duration<int>(2));
                    tf2_filter_->registerCallback(&PredictorNode::energy_predictor_callback, this);        
                    last_autoaim_mode_ = params_.autoaim_mode;
                }
            }
        }
    }).detach();
}

void PredictorNode::init_predictors() {
    vehicle_observer_ = std::make_shared<StandardObserver>(
        StandardObserverParams{
            static_cast<int>(params_.armor_predictor.max_lost),
            static_cast<int>(params_.armor_predictor.max_detect),
            params_.armor_predictor.max_match_distance,
            params_.armor_predictor.max_match_yaw_diff,
            params_.armor_predictor.lost_time_thres_,
            params_.target_frame,
            StandardObserverParams::DDMParams{
                params_.armor_predictor.standard_observer.ekf.sigma2_q_xyz,
                params_.armor_predictor.standard_observer.ekf.sigma2_q_yaw,
                params_.armor_predictor.standard_observer.ekf.sigma2_q_r,
                params_.armor_predictor.standard_observer.ekf.r_xyz_factor,
                params_.armor_predictor.standard_observer.ekf.r_yaw
            }
        }
    );
    armor_observer_ = std::make_shared<ArmorObserver>(
        ArmorObserverParams{
            static_cast<int>(params_.armor_predictor.max_lost),
            static_cast<int>(params_.armor_predictor.max_detect),
            params_.armor_predictor.max_match_distance,
            params_.armor_predictor.max_match_yaw_diff,
            params_.armor_predictor.lost_time_thres_,
            params_.target_frame,
            params_.armor_predictor.armor_observer.kf.sigma2_q_xyz,
            params_.armor_predictor.armor_observer.kf.r_xyz_factor,
            params_.armor_predictor.armor_observer.min_match_distance
        }
    );
    vehicle_observer_->target_type_ = TargetType::NORMAL;
    last_target_type_ = TargetType::NORMAL;
    energy_observer_ = std::make_shared<EnergyObserver>(
        EnergyObserverParams{
            params_.autoaim_mode == 1 ? false : true,
            static_cast<int>(params_.energy_predictor.max_lost),
            static_cast<int>(params_.energy_predictor.max_detect),
            params_.target_frame,
            EnergyObserverParams::KalmanParams {
                params_.energy_predictor.kf_params.sigma_q_x,
                params_.energy_predictor.kf_params.sigma_q_v,
                params_.energy_predictor.kf_params.sigma_q_a,
                params_.energy_predictor.kf_params.sigma_r_x,
                params_.energy_predictor.kf_params.sigma_r_v
            }
        }
    );
}

void PredictorNode::armor_predictor_callback(autoaim_interfaces::msg::Armors::SharedPtr armors_msg) {
    if (param_listener_->is_old(params_)) {
        params_ = param_listener_->get_params();
        RCLCPP_WARN(logger_, "Parameters updated");
        update_predictor_params();
    }
    // build time series
    rclcpp::Time time = armors_msg->header.stamp;
    double dt = time.seconds() - time_predictor_start_;
    time_predictor_start_ = time.seconds();
    // transform armors to target frame
    for (auto &armor : armors_msg->armors) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = armors_msg->header;
        ps.pose = armor.pose;
        try {
            armor.pose = tf2_buffer_->transform(ps, params_.target_frame).pose;
        } catch (const tf2::ExtrapolationException & ex) {
            RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
            return;
        }
    }
    // choose predict mode
    autoaim_interfaces::msg::Target target;
    update_predictor_type(vehicle_observer_);
    last_target_type_ = vehicle_observer_->target_type_;
    // if distance is under prediction threshold, use vehicle observe,
    // otherwise use armor observer
    if (last_target_distance_ > params_.armor_predictor.prediction_thres) {
        vehicle_observer_->predict_target(*armors_msg, dt);
        target = armor_observer_->predict_target(*armors_msg, dt);
    } else {
        armor_observer_->predict_target(*armors_msg, dt);
        target = vehicle_observer_->predict_target(*armors_msg, dt);
    }
    Eigen::Vector3d target_position = Eigen::Vector3d{target.position.x, target.position.y, target.position.z};
    if (target.tracking) {
        last_target_distance_ = target_position.norm();
    }
    target.header.stamp = armors_msg->header.stamp;
    target.header.frame_id = params_.target_frame;
    target_pub_->publish(target);
}

void PredictorNode::energy_predictor_callback(autoaim_interfaces::msg::Armors::SharedPtr armors_msg) {
    if (param_listener_->is_old(params_)) {
        params_ = param_listener_->get_params();
        RCLCPP_WARN(logger_, "Parameters updated");
        update_predictor_params();
    }
    // build time series
    rclcpp::Time time = armors_msg->header.stamp;
    double dt = time.seconds() - time_predictor_start_;
    time_predictor_start_ = time.seconds();
    // transform armors to target frame
    for (auto &armor : armors_msg->armors) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = armors_msg->header;
        ps.pose = armor.pose;
        try {
            armor.pose = tf2_buffer_->transform(ps, params_.target_frame).pose;
        } catch (const tf2::ExtrapolationException & ex) {
            RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
            return;
        }
    }
    // Start Prediction
    auto target = energy_observer_->predict_target(*armors_msg, dt);
    // Publish
    target_pub_->publish(target);
}


void PredictorNode::update_predictor_params() {
    if (params_.autoaim_mode) {
        vehicle_observer_.reset();
        if (vehicle_observer_->target_type_ == TargetType::NORMAL) {
            vehicle_observer_ = std::make_shared<StandardObserver>(
                StandardObserverParams{
                    static_cast<int>(params_.armor_predictor.max_lost),
                    static_cast<int>(params_.armor_predictor.max_detect),
                    params_.armor_predictor.max_match_distance,
                    params_.armor_predictor.max_match_yaw_diff,
                    params_.armor_predictor.lost_time_thres_,
                    params_.target_frame,
                    StandardObserverParams::DDMParams{
                        params_.armor_predictor.standard_observer.ekf.sigma2_q_xyz,
                        params_.armor_predictor.standard_observer.ekf.sigma2_q_yaw,
                        params_.armor_predictor.standard_observer.ekf.sigma2_q_r,
                        params_.armor_predictor.standard_observer.ekf.r_xyz_factor,
                        params_.armor_predictor.standard_observer.ekf.r_yaw
                    }
                }
            );
        } else if (vehicle_observer_->target_type_ == TargetType::OUTPOST) {
            vehicle_observer_ = std::make_shared<OutpostObserver>(
                OutpostObserverParams{
                    static_cast<int>(params_.armor_predictor.max_lost),
                    static_cast<int>(params_.armor_predictor.max_detect),
                    params_.armor_predictor.max_match_distance,
                    params_.armor_predictor.max_match_yaw_diff,
                    params_.armor_predictor.lost_time_thres_,
                    params_.target_frame,
                    OutpostObserverParams::DDMParams{
                        params_.armor_predictor.outpost_observer.ekf.sigma2_q_yaw,
                        params_.armor_predictor.outpost_observer.ekf.sigma2_q_xyz,
                        params_.armor_predictor.outpost_observer.ekf.r_xyz_factor,
                        params_.armor_predictor.outpost_observer.ekf.r_yaw
                    }
                }
            );
        } else if (vehicle_observer_->target_type_ == TargetType::BALANCE) {
            vehicle_observer_ = std::make_shared<BalanceObserver>(
                BalanceObserverParams{
                    static_cast<int>(params_.armor_predictor.max_lost),
                    static_cast<int>(params_.armor_predictor.max_detect),
                    params_.armor_predictor.max_match_distance,
                    params_.armor_predictor.max_match_yaw_diff,
                    params_.armor_predictor.lost_time_thres_,
                    params_.target_frame,
                    BalanceObserverParams::DDMParams{
                        params_.armor_predictor.balance_observer.ekf.sigma2_q_xyz,
                        params_.armor_predictor.balance_observer.ekf.sigma2_q_yaw,
                        params_.armor_predictor.balance_observer.ekf.sigma2_q_r,
                        params_.armor_predictor.balance_observer.ekf.r_xyz_factor,
                        params_.armor_predictor.balance_observer.ekf.r_yaw
                    }
                }
            );
        }
    } else {
        bool is_large_energy = params_.autoaim_mode == 1 ? false : true;
        energy_observer_->set_params(EnergyObserverParams{
            is_large_energy,
            static_cast<int>(params_.energy_predictor.max_lost),
            static_cast<int>(params_.energy_predictor.max_detect),
            params_.target_frame,
            EnergyObserverParams::KalmanParams {
                params_.energy_predictor.kf_params.sigma_q_x,
                params_.energy_predictor.kf_params.sigma_q_v,
                params_.energy_predictor.kf_params.sigma_q_a,
                params_.energy_predictor.kf_params.sigma_r_x,
                params_.energy_predictor.kf_params.sigma_r_v
            }
        });
    }
}

void PredictorNode::update_predictor_type(std::shared_ptr<BaseObserver>& observer) {
    if (last_target_type_ == observer->target_type_) {
        // RCLCPP_INFO(logger_, "target state not changed %d", last_target_type_);        
        return ;
    }
    if (observer->target_type_ == TargetType::NORMAL && last_target_type_ != TargetType::NORMAL) {
        observer = std::make_shared<ArmorObserver>(
            ArmorObserverParams{
                static_cast<int>(params_.armor_predictor.max_lost),
                static_cast<int>(params_.armor_predictor.max_detect),
                params_.armor_predictor.max_match_distance,
                params_.armor_predictor.max_match_yaw_diff,
                params_.armor_predictor.lost_time_thres_,
                params_.target_frame,
                params_.armor_predictor.armor_observer.kf.sigma2_q_xyz,
                params_.armor_predictor.armor_observer.kf.r_xyz_factor,
                params_.armor_predictor.armor_observer.min_match_distance
            }
        );
        last_target_type_ = TargetType::NORMAL;
    } else if (observer->target_type_ == TargetType::OUTPOST && last_target_type_ != TargetType::OUTPOST) {
        observer = std::make_shared<OutpostObserver>(
            OutpostObserverParams{
                static_cast<int>(params_.armor_predictor.max_lost),
                static_cast<int>(params_.armor_predictor.max_detect),
                params_.armor_predictor.max_match_distance,
                params_.armor_predictor.max_match_yaw_diff,
                params_.armor_predictor.lost_time_thres_,
                params_.target_frame,
                OutpostObserverParams::DDMParams{
                    params_.armor_predictor.outpost_observer.ekf.sigma2_q_yaw,
                    params_.armor_predictor.outpost_observer.ekf.sigma2_q_xyz,
                    params_.armor_predictor.outpost_observer.ekf.r_xyz_factor,
                    params_.armor_predictor.outpost_observer.ekf.r_yaw
                }
            }
        );
        last_target_type_ = TargetType::OUTPOST;
    } else if (observer->target_type_ == TargetType::BALANCE && last_target_type_ != TargetType::BALANCE) {
        observer = std::make_shared<BalanceObserver>(
            BalanceObserverParams{
                static_cast<int>(params_.armor_predictor.max_lost),
                static_cast<int>(params_.armor_predictor.max_detect),
                params_.armor_predictor.max_match_distance,
                params_.armor_predictor.max_match_yaw_diff,
                params_.armor_predictor.lost_time_thres_,
                params_.target_frame,
                BalanceObserverParams::DDMParams{
                    params_.armor_predictor.balance_observer.ekf.sigma2_q_xyz,
                    params_.armor_predictor.balance_observer.ekf.sigma2_q_yaw,
                    params_.armor_predictor.balance_observer.ekf.sigma2_q_r,
                    params_.armor_predictor.balance_observer.ekf.r_xyz_factor,
                    params_.armor_predictor.balance_observer.ekf.r_yaw
                }
            }
        );
        last_target_type_ = TargetType::BALANCE;
    } else {
        RCLCPP_INFO(logger_, "last type %d", last_target_type_);
        RCLCPP_INFO(logger_, "this type %d", observer->target_type_);
    }
}

PredictorNode::~PredictorNode() {

}

} // namespace helios_cv

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(helios_cv::PredictorNode);
