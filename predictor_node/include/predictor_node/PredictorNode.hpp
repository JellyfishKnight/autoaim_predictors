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
 #pragma once

// ros
#include <cstdint>
#include <rclcpp/rclcpp.hpp>
#include <angles/angles.h>

// tf2
#include <rclcpp/service.hpp>
#include <std_srvs/std_srvs/srv/detail/trigger__struct.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <message_filters/subscriber.h>

// interfaces
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/target.hpp"
#include <std_srvs/srv/trigger.hpp>
// custom
#include "armor_predictor/VehicleObserver.hpp"
// #include "energy_predictor/EnergyPredictor.hpp"

// auto generated by ros2 generate_parameter_library
// https://github.com/PickNikRobotics/generate_parameter_library
#include "predictor_node_parameters.hpp"


namespace helios_cv {

using tf2_filter = tf2_ros::MessageFilter<autoaim_interfaces::msg::Armors>;
using ParamListener = predictor_node::ParamListener;
using Params = predictor_node::Params;

class PredictorNode : public rclcpp::Node {
public:
    PredictorNode(const rclcpp::NodeOptions& options);

    ~PredictorNode();

private:
    void armor_predictor_callback(autoaim_interfaces::msg::Armors::SharedPtr armors_msg);

    void energy_predictor_callback(autoaim_interfaces::msg::Armors::SharedPtr armors_msg);

    // time series
    double time_predictor_start_;


    std::shared_ptr<VehicleObserver> armor_predictor_;
    // std::shared_ptr<EnergyPredictor> energy_predictor_;
    void init_predictors();

    rclcpp::Publisher<autoaim_interfaces::msg::Target>::SharedPtr target_pub_;
    // tf2 
    // Subscriber with tf2 message_filter
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    message_filters::Subscriber<autoaim_interfaces::msg::Armors> armors_sub_;
    std::shared_ptr<tf2_filter> tf2_filter_;

    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr point_pub_;

    // Visualization marker publisher
    visualization_msgs::msg::Marker position_marker_;
    visualization_msgs::msg::Marker linear_v_marker_;
    visualization_msgs::msg::Marker angular_v_marker_;
    visualization_msgs::msg::Marker armor_marker_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    void init_markers();
    void publish_armor_markers(autoaim_interfaces::msg::Target target);
    void publish_energy_markers(autoaim_interfaces::msg::Target target);

    // Camera info part
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    cv::Point2f cam_center_;
    std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;

    //弧度制角度制转换常量
    double degree2rad = CV_PI / 180.0;
    // parameter utilities
    std::shared_ptr<ParamListener> param_listener_;
    Params params_;
    void update_predictor_params();

    // reset predictor service
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_predictor_service_;
    uint8_t last_autoaim_mode_ = 0;

    rclcpp::Logger logger_ = rclcpp::get_logger("PredictorNode");
};



} // helios_cv