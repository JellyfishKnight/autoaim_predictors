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

typedef struct ArmorPredictorParams : public BaseObserverParams {
    typedef struct KFParams {
        double sigma2_q_xyz;
        double r_xyz_factor;
    } KFParams;
    KFParams kf_params;
    int max_lost;
    int max_detect;
    double min_match_distance;
    double max_match_distance;
    double lost_time_thresh;
    std::string target_frame;
}ArmorPredictorParams;

class ArmorPredictor : public BaseObserver {
public:
    ArmorPredictor(const ArmorPredictorParams& params);

    ~ArmorPredictor();

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt) override;

    void reset_kalman() override;
private:
    void init_kalman();

    Eigen::Vector3d kalman_predict();

    bool judge_spinning(const autoaim_interfaces::msg::Armor& armor);

    Eigen::Vector3d state2position(const Eigen::VectorXd& state) override;

    void track_armor(autoaim_interfaces::msg::Armors armors) override;

    void update_target_type(const autoaim_interfaces::msg::Armor& armor) override;

    ArmorPredictorParams params_;

    EigenKalmanFilter kalman_filter_;

    rclcpp::Logger logger_ = rclcpp::get_logger("ArmorPredictor");
};

} // helios_cv