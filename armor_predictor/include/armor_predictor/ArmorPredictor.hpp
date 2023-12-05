#include <rclcpp/rclcpp.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/target.hpp"

#include "autoaim_utilities/Armor.hpp"
#include "autoaim_utilities/KalmanFilter.hpp"

namespace helios_cv {

typedef struct ArmorPredictorParams {
    int max_lost_;
}APParams;

class ArmorPredictor {
public:
    ArmorPredictor(const APParams& params);

    ~ArmorPredictor();

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt);

private:
    void init_kalman();

    void reset_kalman();

    Eigen::Vector3d kalman_predict();

    APParams params_;

    bool find_state_ = false;

    Eigen::Vector3d target_xyz_;
    Eigen::Vector3d predicted_xyz_;

    std::shared_ptr<EigenKalmanFilter> kalman_filter_;

};

} // helios_cv