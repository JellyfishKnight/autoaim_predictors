#include <rclcpp/rclcpp.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/target.hpp"

#include "autoaim_utilities/Armor.hpp"
#include "autoaim_utilities/KalmanFilter.hpp"

#include <cstdint>

#include "BaseObserver.hpp"

namespace helios_cv {

typedef struct ArmorObserverParams : public BaseObserverParams {
    ArmorObserverParams(
        int max_lost,
        int max_detect,
        double max_match_distance,
        double max_match_yaw_diff,
        double lost_time_thresh,
        std::string target_frame,
        double sigma2_q_xyz,
        double r_xyz_factor,
        double min_match_distance
    ) : BaseObserverParams(max_lost, max_detect, max_match_distance, max_match_yaw_diff, 
        lost_time_thresh, std::move(target_frame)),
        kf_params({sigma2_q_xyz, r_xyz_factor}),
        min_match_distance(min_match_distance) {}
    typedef struct KFParams {
        double sigma2_q_xyz;
        double r_xyz_factor;
    } KFParams;
    KFParams kf_params;
    double min_match_distance;
}ArmorObserverParams;

class ArmorObserver : public BaseObserver {
public:
    ArmorObserver(const ArmorObserverParams& params);

    ~ArmorObserver() = default;

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt) final;

    void set_params(void *params) final;

    void reset_kalman() final;
private:
    void init() final;

    bool judge_spinning(const autoaim_interfaces::msg::Armor& armor);

    void armor_jump(const autoaim_interfaces::msg::Armor same_id_armor) final;

    Eigen::Vector3d state2position(const Eigen::VectorXd& state) final;

    void track_armor(autoaim_interfaces::msg::Armors armors) final;

    void update_target_type(const autoaim_interfaces::msg::Armor& armor) final;

    ArmorObserverParams params_;

    EigenKalmanFilter kalman_filter_;

    rclcpp::Logger logger_ = rclcpp::get_logger("ArmorObserver");
};

} // helios_cv