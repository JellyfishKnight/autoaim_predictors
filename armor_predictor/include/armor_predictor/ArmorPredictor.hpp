// created by liuhan on 2023/9/15
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

#include <rclcpp/rclcpp.hpp>
#include <angles/angles.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "autoaim_utilities/ExtendedKalmanFilter.hpp"
#include "autoaim_utilities/Armor.hpp"

#include "sensor_msgs/msg/camera_info.hpp"
#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/target.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>

namespace helios_cv {

typedef enum {LOST, TEMP_LOST, TRACKING, DETECTING} TrakerState;
typedef enum {BALANCE, OUTPOST, NORMAL} TargetType;

typedef struct ArmorPredictorParams{
    std::string target_frame;
    typedef struct EKFParams{
        double sigma2_q_xyz;
        double sigma2_q_yaw;
        double sigma2_q_r;
        double r_xyz_factor;
        double r_yaw;
    }EKFParams;
    EKFParams ekf_params;
    int max_lost;
    int max_detect;
    double max_match_distance;
    double max_match_yaw_diff;
    double lost_time_thresh;
}APParams;

class ArmorPredictor {
public:
    ArmorPredictor(const APParams& params);

    void set_cam_info(sensor_msgs::msg::CameraInfo::SharedPtr cam_info);

    void init();

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, const rclcpp::Time& now);

    std::vector<double> get_state() const;
private:
    /**
     * @brief 
     * 
     * @param armors 
     */
    void armor_predict(autoaim_interfaces::msg::Armors armors);

    APParams params_;

    // State Machine
    int find_state_;
    int lost_cnt_ = 0;
    int detect_cnt_ = 0;

    double time_predictor_start_;
    //
    uint8_t car_name_;
    //
    std::string tracking_number_;

    // 识别到的目标点
    cv::Point2f target_point;
    // 上一次的装甲板状态
    double last_yaw_ = 0;
    double last_y_ = 0;
    double last_r_ = 0;
    double target_yaw_ = 0;
    std::string armor_type_;
    TargetType target_type_;
    autoaim_interfaces::msg::Armor last_armor_;

    autoaim_interfaces::msg::Armor tracking_armor_;
    // 目标车辆状态
    Eigen::VectorXd target_state_;
    // 车中心点
    Eigen::Vector3d car_center_ = {0, 0, 0};
    //目标xyz
    Eigen::Vector3d target_xyz_ = {0, 0, 0};
    //本次预测xyz
    Eigen::Vector3d predict_xyz_ = {0, 0, 0};
    //上一次的目标xyz
    Eigen::Vector3d last_xyz_ = {0, 0, 0};
    //上一次预测的xyz
    Eigen::Vector3d last_predict_xyz_ = {0, 0, 0};
    // kalman utilities
    double dz_;
    double dt_ = 0.008f;
    double s2qxyz_;
    double s2qyaw_;
    double s2qr_;
    double r_xyz_factor_;
    double r_yaw_;
    ExtendedKalmanFilter ekf_;

    void update_target_type(const autoaim_interfaces::msg::Armor& armor);

    double orientation2yaw(const geometry_msgs::msg::Quaternion& orientation);

    void reset_kalman();

    void armor_jump(const autoaim_interfaces::msg::Armor tracking_armor);

    Eigen::Vector3d state2position(const Eigen::VectorXd& state);

    // Logger
    rclcpp::Logger logger_ = rclcpp::get_logger("ArmorPredictor");
};

} // namespace helios_cv