// created by lijunqi, liuhan on 2023/9/15
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

// ros
#include <autoaim_utilities/Armor.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/rclcpp.hpp>
#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/target.hpp"
#include <sensor_msgs/msg/camera_info.hpp>
#include <string>
#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// custom
#include "autoaim_utilities/Omega.hpp"
#include "autoaim_utilities/Armor.hpp"
#include "autoaim_utilities/KalmanFilter.hpp"
// eigen 
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
// ceres-solver
#include <ceres/ceres.h>
#include <ceres/problem.h>
// opencv
#include <opencv2/core.hpp>
// STL
#include <thread>
#include <queue>
#include <tuple>

#define INIT 1
#define STANDBY 2
#define ESTIMATE 3
#define PREDICT 4
#define BIG_ENERGY 2
#define SMALL_ENRGY 1
#define AUTOAIM 0

namespace helios_cv {

typedef struct EnergyObserverParams {
    bool is_large_energy; // 0 small energy 1 large energy
    int max_lost;
    int max_detect;
    std::string target_frame;
    typedef struct KalmanParams{
        double sigma_q_x;
        double sigma_q_v;
        double sigma_q_a;
        double sigma_r_x;
        double sigma_r_v;
    }KalmanParams;
    KalmanParams kf_params;
}EnergyObserverParams;

typedef struct OmegaState {
    double omega;
    double t;
    int st;
    bool solve;
    double a;
    double w;
    double phi;
    bool refresh;
}OmegaState;

class LeastSquares {
public:
    LeastSquares();

    void estimate();

    std::queue<OmegaState> omega_state_out_fifo_;
    std::queue<OmegaState> omega_state_in_fifo_;
private:
    struct SinResidual{
        SinResidual(double t, double omega): omega_(omega), t_(t) {}

        template<class T>
        bool operator()(const T* const a, const T* const w,const T* phi ,  T* residual)const{//const T* phi
            residual[0] = omega_ - (a[0]*sin(w[0]*t_+phi[0]) + 2.09 - a[0]);
            return true;
        }
        private:
        const double omega_;
        const double t_;
    };
    std::shared_ptr<ceres::Problem> problem;

    void least_squares_refresh();

    std::vector<double> filter_omega_;
    std::vector<double> t_;
    int st_;
    double phi_;
    double w_;
    double a_;
    bool isSolve_;
    bool refresh_;
    // Logger
    rclcpp::Logger logger_ = rclcpp::get_logger("LeastSquares");
};

class EnergyObserver {
public:
    EnergyObserver(const EnergyObserverParams& params);

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt);

    void reset_observer();

    void set_params(const EnergyObserverParams &params);

    int find_state_;
private:
    double orientation2roll(const geometry_msgs::msg::Quaternion& orientation);

    void track_energy(const autoaim_interfaces::msg::Armor& armors);

    bool energy_state_switch();

    uint8_t circle_mode_;

    Omega omega_;
    bool isSolve_;
    bool refresh_;
    bool matched_;
    uint8_t ceres_cnt_;
    double pub_time_, pub_omega;

    double a_, w_, phi_;

    autoaim_interfaces::msg::Armor tracking_armor_;
    autoaim_interfaces::msg::Armor last_tracking_armor_;

    // params
    EnergyObserverParams params_;

    EigenKalmanFilter omega_kf_;
    double dt_;
    int lost_cnt_{};
    int detect_cnt_{};

    LeastSquares least_squares_;

    // estimate thread
    std::thread *estimate_thread_;

    // Logger
    rclcpp::Logger logger_ = rclcpp::get_logger("EnergyObserver");
};


} // namespace helios_cv