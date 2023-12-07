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
#include <Eigen/src/Core/Matrix.h>
#include <armor_predictor/VehicleObserver.hpp>
#include <autoaim_interfaces/msg/detail/target__struct.hpp>
#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/rate.hpp>
#include <rclcpp/utilities.hpp>
#include <std_srvs/std_srvs/srv/detail/trigger__struct.hpp>

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
    if (params_.debug) {
        init_markers();
    } else {
        marker_pub_ = nullptr;
    }
    // create cam info subscriber
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::SharedPtr camera_info) {
        cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
        cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
        ///TODO: not sure if we need pnp solver in predictor node
        // pnp_solver_ = std::make_shared<PnPSolver>(cam_info_->k, camera_info->d, PnPParams{
        //     params_.pnp_solver.small_armor_width,
        //     params_.pnp_solver.small_armor_height,
        //     params_.pnp_solver.large_armor_width,
        //     params_.pnp_solver.large_armor_height,
        //     params_.pnp_solver.energy_armor_width,
        //     params_.pnp_solver.energy_armor_height
        // });
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
            armor_predictor_->find_state_ = LOST;
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
    vehicle_observer_ = std::make_shared<VehicleObserver>(VOParams{
        params_.target_frame,
        VOParams::EKFParams{
            params_.armor_predictor.ekf.sigma2_q_xyz,
            params_.armor_predictor.ekf.sigma2_q_yaw,
            params_.armor_predictor.ekf.sigma2_q_r,
            params_.armor_predictor.ekf.r_xyz_factor,
            params_.armor_predictor.ekf.r_yaw
        },
        static_cast<int>(params_.armor_predictor.max_lost),
        static_cast<int>(params_.armor_predictor.max_detect),
        params_.armor_predictor.max_match_distance,
        params_.armor_predictor.max_match_yaw_diff,
        params_.armor_predictor.lost_time_thres_
    });
    vehicle_observer_->init();
    armor_predictor_ = std::make_shared<ArmorPredictor>(APParams{
        APParams::KFParams{
            params_.armor_predictor.kf.sigma2_q_xyz,
            params_.armor_predictor.kf.r_xyz_factor
        },
        static_cast<int>(params_.armor_predictor.max_lost),
        static_cast<int>(params_.armor_predictor.max_detect),
        params_.armor_predictor.min_match_distance,
        params_.armor_predictor.max_match_yaw_diff,
        params_.armor_predictor.lost_time_thres_
    });
    ///TODO: init energy_predictor
    // energy_predictor = std::make_shared<EnergyPredictor>();
}


void PredictorNode::init_markers() {
    // Visualization Marker Publisher
    // See http://wiki.ros.org/rviz/DisplayTypes/Marker
    position_marker_.ns = "position";
    position_marker_.type = visualization_msgs::msg::Marker::SPHERE;
    position_marker_.scale.x = position_marker_.scale.y = position_marker_.scale.z = 0.1;
    position_marker_.color.a = 1.0;
    position_marker_.color.g = 1.0;
    linear_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
    linear_v_marker_.ns = "linear_v";
    linear_v_marker_.scale.x = 0.03;
    linear_v_marker_.scale.y = 0.05;
    linear_v_marker_.color.a = 1.0;
    linear_v_marker_.color.r = 1.0;
    linear_v_marker_.color.g = 1.0;
    angular_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
    angular_v_marker_.ns = "angular_v";
    angular_v_marker_.scale.x = 0.03;
    angular_v_marker_.scale.y = 0.05;
    angular_v_marker_.color.a = 1.0;
    angular_v_marker_.color.b = 1.0;
    angular_v_marker_.color.g = 1.0;
    armor_marker_.ns = "armors";
    armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
    armor_marker_.scale.x = 0.03;
    armor_marker_.scale.z = 0.125;
    armor_marker_.color.a = 1.0;
    armor_marker_.color.r = 1.0;
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/predictor/marker", 10);
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
    // keep both trackers update
    if (use_vehicle_observe_) {
        target = vehicle_observer_->predict_target(*armors_msg, dt);
        armor_predictor_->predict_target(*armors_msg, dt);
    } else {
        target = armor_predictor_->predict_target(*armors_msg, dt);
        vehicle_observer_->predict_target(*armors_msg, dt);
    }
    Eigen::Vector3d target_position = Eigen::Vector3d{target.position.x, target.position.y, target.position.z};
    last_target_distance_ = target_position.norm();
    // if distance is under prediction threshold, use vehicle observe,
    // otherwise use armor predictor
    if (last_target_distance_ > params_.armor_predictor.prediction_thres) {
        use_vehicle_observe_ = false;
    } else {
        use_vehicle_observe_ = true;
    }
    target.header.stamp = armors_msg->header.stamp;
    target.header.frame_id = params_.target_frame;
    // RCLCPP_WARN(logger_, "vx: %f, vy %f, vz: %f", target.velocity.x, target.velocity.y, target.velocity.z);
    target_pub_->publish(target);
    if (params_.debug) {
        // publish visualization marker
        publish_armor_markers(target);
    }
}

void PredictorNode::energy_predictor_callback(autoaim_interfaces::msg::Armors::SharedPtr armors_msg) {
    if (param_listener_->is_old(params_)) {
        params_ = param_listener_->get_params();
        RCLCPP_WARN(logger_, "Parameters updated");
        update_predictor_params();
    }
    RCLCPP_ERROR_ONCE(logger_, "Energy mode!");
    autoaim_interfaces::msg::Target target;
    target.header.stamp = armors_msg->header.stamp;
    target.header.frame_id = params_.target_frame;
}

void PredictorNode::publish_energy_markers(autoaim_interfaces::msg::Target target) {

}

void PredictorNode::publish_armor_markers(autoaim_interfaces::msg::Target target) {
    position_marker_.header = target.header;
    linear_v_marker_.header = target.header;
    angular_v_marker_.header = target.header;
    armor_marker_.header = target.header;

    visualization_msgs::msg::MarkerArray marker_array;
    if (target.tracking) {
        double yaw = target.yaw, r1 = target.radius_1, r2 = target.radius_2;
        double xc = target.position.x, yc = target.position.y, zc = target.position.z;
        double vxc = target.velocity.x, vyc = target.velocity.y, vzc = target.velocity.z, vyaw = target.v_yaw;
        double dz = target.dz;

        position_marker_.action = visualization_msgs::msg::Marker::ADD;
        position_marker_.pose.position.x = xc;
        position_marker_.pose.position.y = yc;
        position_marker_.pose.position.z = zc + dz / 2;

        linear_v_marker_.action = visualization_msgs::msg::Marker::ADD;
        linear_v_marker_.points.clear();
        linear_v_marker_.points.emplace_back(position_marker_.pose.position);
        geometry_msgs::msg::Point arrow_end = position_marker_.pose.position;
        arrow_end.x += vxc;
        arrow_end.y += vyc;
        arrow_end.z += vzc;
        linear_v_marker_.points.emplace_back(arrow_end);

        angular_v_marker_.action = visualization_msgs::msg::Marker::ADD;
        angular_v_marker_.points.clear();
        angular_v_marker_.points.emplace_back(position_marker_.pose.position);
        arrow_end = position_marker_.pose.position;
        arrow_end.z += vyaw / M_PI;
        angular_v_marker_.points.emplace_back(arrow_end);

        armor_marker_.action = visualization_msgs::msg::Marker::ADD;
        armor_marker_.scale.y = target.armor_type == "SMALL" ? 0.135 : 0.23;
        bool is_current_pair = true;
        size_t a_n = static_cast<int>(vehicle_observer_->target_type_) + 2;
        geometry_msgs::msg::Point p_a;
        double r = 0;
        for (size_t i = 0; i < a_n; i++) {
            double tmp_yaw = yaw + i * (2 * M_PI / a_n);
            // Only 4 armors has 2 radius and height
            if (a_n == 4) {
                r = is_current_pair ? r1 : r2;
                p_a.z = zc + (is_current_pair ? 0 : dz);
                is_current_pair = !is_current_pair;
            } else {
                r = r1;
                p_a.z = zc;
            }
            p_a.x = xc - r * cos(tmp_yaw);
            p_a.y = yc - r * sin(tmp_yaw);

            armor_marker_.id = i;
            armor_marker_.pose.position = p_a;
            tf2::Quaternion q;
            q.setRPY(0, target.armor_type == "outpost" ? -0.26 : 0.26, tmp_yaw);
            armor_marker_.pose.orientation = tf2::toMsg(q);
            marker_array.markers.emplace_back(armor_marker_);
        }
    } else {
        position_marker_.action = visualization_msgs::msg::Marker::DELETE;
        linear_v_marker_.action = visualization_msgs::msg::Marker::DELETE;
        angular_v_marker_.action = visualization_msgs::msg::Marker::DELETE;
        armor_marker_.action = visualization_msgs::msg::Marker::DELETE;
    }

    marker_array.markers.emplace_back(position_marker_);
    marker_array.markers.emplace_back(linear_v_marker_);
    marker_array.markers.emplace_back(angular_v_marker_);
    marker_array.markers.emplace_back(armor_marker_);
    marker_pub_->publish(marker_array);
}

void PredictorNode::update_predictor_params() {
    if (params_.autoaim_mode) {
        vehicle_observer_.reset();
        vehicle_observer_ = std::make_shared<VehicleObserver>(VOParams{
            params_.target_frame,
            VOParams::EKFParams{
                params_.armor_predictor.ekf.sigma2_q_xyz,
                params_.armor_predictor.ekf.sigma2_q_yaw,
                params_.armor_predictor.ekf.sigma2_q_r,
                params_.armor_predictor.ekf.r_xyz_factor,
                params_.armor_predictor.ekf.r_yaw
            },
            static_cast<int>(params_.armor_predictor.max_lost),
            static_cast<int>(params_.armor_predictor.max_detect),
            params_.armor_predictor.max_match_distance,
            params_.armor_predictor.max_match_yaw_diff,
            params_.armor_predictor.lost_time_thres_
        });
        vehicle_observer_->init();    
        armor_predictor_ = std::make_shared<ArmorPredictor>(APParams{
            APParams::KFParams{
                params_.armor_predictor.kf.sigma2_q_xyz,
                params_.armor_predictor.kf.r_xyz_factor
            },
            static_cast<int>(params_.armor_predictor.max_lost),
            static_cast<int>(params_.armor_predictor.max_detect),
            params_.armor_predictor.min_match_distance,
            params_.armor_predictor.max_match_yaw_diff,
            params_.armor_predictor.lost_time_thres_
        });
    } else {

    }
}

PredictorNode::~PredictorNode() {

}

} // namespace helios_cv

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(helios_cv::PredictorNode);
