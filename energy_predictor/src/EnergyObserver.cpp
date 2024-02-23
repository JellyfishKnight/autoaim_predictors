// created by lijunqi, liuhan on 2023/9/15
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#include "EnergyObserver.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <autoaim_interfaces/msg/detail/armor__struct.hpp>
#include <autoaim_interfaces/msg/detail/target__struct.hpp>
#include <autoaim_utilities/Armor.hpp>
#include <ceres/problem.h>
#include <complex>
#include <cstdint>
#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/rate.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/utilities.hpp>
#include <tuple>

namespace helios_cv {

LeastSquares::LeastSquares() {
    filter_omega_.clear();
    t_.clear();
    phi_ = 3.141592;
    w_ = 1.942;
    a_ = 0.9125;
    isSolve_ = false;
    refresh_ = false;
    problem = std::make_shared<ceres::Problem>();
}

void LeastSquares::estimate() {
    if (omega_state_in_fifo_.empty()) {
        return ;
    }
    auto omega_state = omega_state_in_fifo_.front();
    omega_state_in_fifo_.pop();
    // Handle input
    refresh_ = omega_state.refresh;
    if(refresh_){
        least_squares_refresh();
    }
    filter_omega_.push_back(omega_state.omega);
    a_ = omega_state.a;
    w_ = omega_state.w;
    phi_ = omega_state.phi;
    t_.push_back(omega_state.t);
    isSolve_ = omega_state.solve;
    st_ = omega_state.st > 0 ? omega_state.st : 0;
    // Solve 
    if (isSolve_) {
        // set problem
        for(int i = st_; i < filter_omega_.size(); i++){
            ceres::CostFunction* const_func = new ceres::AutoDiffCostFunction<SinResidual, 1, 1, 1, 1>(
                    new SinResidual(t_[i], filter_omega_[i])
                );
            problem->AddResidualBlock(const_func, NULL, &a_, &w_, &phi_);
        }
        problem->SetParameterLowerBound(&a_, 0 ,0.78);//0 0.78
        problem->SetParameterUpperBound(&a_, 0 ,1.045);//0 1.045
        problem->SetParameterLowerBound(&w_, 0 ,1.884);//0 1.884
        problem->SetParameterUpperBound(&w_, 0 ,2.0);//0 2.0
        problem->SetParameterLowerBound(&phi_, 0 ,-M_PI);//CV_PI
        problem->SetParameterUpperBound(&phi_, 0 ,M_PI);
        ceres::Solver::Options options;
        options.max_num_iterations = 50;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, problem.get(), &summary);

        RCLCPP_DEBUG(logger_, "Final a: %f, w: %f, phi: %f", a_, w_, phi_);

        if (a_ < 0.780) {
            a_ = 0.780;
        } else if (a_ > 1.045) {
            a_ = 1.045;
        }
        if (w_ < 0) {
            w_=fabs(w_);
        }
        if (w_ < 1.884) {
            w_ = 1.884;
        } else if (w_ > 2.0) {
            w_ = 2.0;
        }
        // set output
        OmegaState omega_state_out;
        omega_state_out.a = a_;
        omega_state_out.w = w_;
        omega_state_out.phi = phi_;
        omega_state_out_fifo_.push(omega_state_out);
    }
}

void LeastSquares::least_squares_refresh() {
    problem.reset(new ceres::Problem);
    filter_omega_.clear();
    t_.clear();
    phi_ = 3.141592;
    w_ = 1.942;
    a_ = 0.9125;
    isSolve_ = false;
    refresh_ = false;
}

EnergyObserver::EnergyObserver(const EnergyObserverParams& params) {
    params_ = params;
    // Init 
    find_state_ = LOST;
    reset_observer();
    // Create kalman
    auto f = [this] (const Eigen::VectorXd& ) {
        Eigen::MatrixXd f(3, 3);
        double t2 = 0.5 * dt_ * dt_;
        f << 1,  dt_, t2,
             0,  1,   dt_,
             0,  0,   1;
        return f;
    };
    auto h = [this] (const Eigen::VectorXd& ) {
        Eigen::MatrixXd h(2, 3);
        h << 1, 0, 0,
             0, 1, 0;
        return h;
    };
    auto q = [this] () {
        Eigen::MatrixXd q(3, 3);
        q << params_.kf_params.sigma_q_x, 0, 0,
             0, params_.kf_params.sigma_q_x, 0,
             0, 0, params_.kf_params.sigma_q_x;
        return q;
    };
    auto r = [this] (const Eigen::VectorXd& z) {
        Eigen::DiagonalMatrix<double, 2> r;
        r.setIdentity();
        return r;
    };
    Eigen::DiagonalMatrix<double, 3> p;
    p.setIdentity();
    omega_kf_ = EigenKalmanFilter(f, h, q, r, p);
    // Create ceres thread
    std::thread([this]()->void {
        while (rclcpp::ok()) {
            least_squares_.estimate();
        }
    }).detach();
}

void EnergyObserver::set_params(const EnergyObserverParams &params) {
    params_ = params;
}

void EnergyObserver::reset_observer() {
    omega_.refresh();
    isSolve_ = false;
    ceres_cnt_ = 1;
    isSolve_ = false;
    refresh_ = true;
    circle_mode_ = INIT;
    omega_kf_.set_state(Eigen::Vector3d::Zero());
    RCLCPP_WARN(logger_, "Reset Energy Observer");
}

double EnergyObserver::orientation2roll(const geometry_msgs::msg::Quaternion& orientation) {
    double roll, pitch, yaw;
    tf2::Quaternion q(orientation.x, orientation.y, orientation.z, orientation.w);
    tf2::Matrix3x3 m(q);
    m.getRPY(roll, pitch, yaw);
    return roll;
}

autoaim_interfaces::msg::Target EnergyObserver::predict_target(autoaim_interfaces::msg::Armors armors, double dt) {
    dt_ = dt;
    if (dt > 1.0) {
        find_state_ = LOST;
    }
    autoaim_interfaces::msg::Target target;
    target.header.frame_id = params_.target_frame;
    target.header.stamp = armors.header.stamp;
    if (armors.armors.empty()) {
        target.tracking = false;
        matched_ = false;
        tracking_armor_.type = static_cast<int>(ArmorType::INVALID);
    } else {
        // Find target armor
        tracking_armor_ = armors.armors[0];
        for (auto armor : armors.armors) {
            if (armor.type == static_cast<int>(ArmorType::ENERGY_TARGET)) {
                tracking_armor_ = armor;
                break;
            }
        }
        omega_.set_time(rclcpp::Time(armors.header.stamp).seconds());
        // Start observation
        if (find_state_ == LOST) {
            reset_observer();
            find_state_ = DETECTING;
        } else {
            refresh_ = false;
            track_energy(tracking_armor_);
            if (find_state_ == TRACKING || find_state_ == TEMP_LOST) {
                // Pack data
                target.tracking = true;
                target.position = tracking_armor_.pose.position;
                target.velocity.x = a_;
                target.velocity.y = w_;
                target.velocity.z = phi_;
                target.armors_num = 5;
                target.armor_type = "energy";
                target.radius_1 = 0.7;
                target.yaw = orientation2roll(tracking_armor_.pose.orientation);
            }
        }
    }
    // Update state machine
    if (find_state_ == DETECTING) {
        if (matched_) {
            detect_cnt_++;
            if (detect_cnt_ > 10) {
                detect_cnt_ = 0;
                find_state_ = TRACKING;
            }
        }
        find_state_ = TRACKING;
    } else if (find_state_ == TRACKING) {
        if (!matched_) {
            find_state_ = TEMP_LOST;
            lost_cnt_++;
        }
    } else if (find_state_ == TEMP_LOST) {
        if (!matched_) {
            lost_cnt_++;
            if (lost_cnt_ > 20) {
                RCLCPP_WARN(logger_, "Target Lost!");
                lost_cnt_ = 0;
                find_state_ = LOST;
            }
        } else {
            find_state_ = TRACKING;
            lost_cnt_ = 0;
        }
    } 
    return target;
}

void EnergyObserver::track_energy(const autoaim_interfaces::msg::Armor& armor) {
    matched_ = false;
    // When we hit one armor, while the armor switch, we can only see the energy fan or nothing
    // we should take this as temp lost, don't reset params and keep predicting
    if (armor.type == static_cast<int>(ArmorType::ENERGY_TARGET)) {
        matched_ = true;
        omega_.set_theta(orientation2roll(armor.pose.orientation));
        if (!omega_.start_) {
            a_ = 0;
            w_ = 0;
            phi_ = 0;
        } else {
            pub_time_ = omega_.time_series_.back();
            if (params_.is_large_energy) {
                // Get omega from kalman filter
                Eigen::VectorXd measure(2);
                measure << omega_.total_theta_, omega_.current_theta_;
                omega_kf_.predict();
                auto state = omega_kf_.correct(measure);
                double omega = state(1);
                omega_.set_filter(omega);
                // Set ceres' problem
                least_squares_.omega_state_in_fifo_.push(
                    OmegaState{
                        omega_.filter_omega_.back(),
                        pub_time_,
                        omega_.st_,
                        isSolve_,
                        omega_.a_,
                        omega_.w_,
                        omega_.phi_,
                        refresh_
                    }
                );
                if (!least_squares_.omega_state_out_fifo_.empty()) {
                    auto omega_state = least_squares_.omega_state_out_fifo_.front();
                    omega_.set_a_w_phi(omega_state.a, omega_state.w, omega_state.phi);
                    least_squares_.omega_state_out_fifo_.pop();
                }
                if (energy_state_switch()) {
                    if (std::fabs(omega_.get_err()) > 0.5) {
                        omega_.fit_cnt_++;
                        a_ = omega_.a_;
                        w_ = omega_.w_;
                        phi_ = omega_.phi_;
                        if ((omega_.fit_cnt_ % 40 == 0 && ceres_cnt_ == 1) || 
                            (omega_.fit_cnt_ % 30 == 0 || ceres_cnt_ != 1)) {
                            circle_mode_ = STANDBY;
                            omega_.change_st();
                        }
                    } else {
                        a_ = omega_.a_;
                        w_ = omega_.w_;
                        phi_ = omega_.phi_;
                    }
                } 
            } else {
                a_ = 1.05;
                w_ = 0;
                phi_ = 0;
            }
        }
        last_tracking_armor_ = armor;
    }
}

bool EnergyObserver::energy_state_switch() {
    switch (circle_mode_) {
        case INIT:
            if (omega_.FindWavePeak()) {
                circle_mode_ = STANDBY;
                isSolve_ = false;
                omega_.refresh_after_wave();
            }
            return false;
        case STANDBY: {
            auto time_gap = omega_.get_time_gap();
            if (time_gap > 1.5) {
                circle_mode_ = ESTIMATE;
            }
            return false;
        }
        case ESTIMATE:
            ceres_cnt_++;
            isSolve_ = true;
            circle_mode_ = PREDICT;
            return true;
        case PREDICT:
            isSolve_ = false;
            return true;
        default:
            return true;
    }
}

} // namespace helios_cv