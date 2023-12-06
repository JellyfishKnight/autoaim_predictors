#include <autoaim_interfaces/msg/detail/armor__struct.hpp>
#include <rclcpp/rclcpp.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/target.hpp"

#include "autoaim_utilities/Armor.hpp"
#include "autoaim_utilities/KalmanFilter.hpp"

#include <cstdint>

namespace helios_cv {

typedef enum {LOST, TEMP_LOST, TRACKING, DETECTING} TrakerState;
typedef enum {BALANCE, OUTPOST, NORMAL} TargetType;

typedef struct ArmorPredictorParams {
    typedef struct KFParams {
        double sigma2_q_xyz;
        double r_xyz_factor;
    } KFParams;
    KFParams kf_params;
    int max_lost;
    int max_detect;
    double max_match_distance;
    double lost_time_thresh;
    std::string target_frame;
}APParams;

class ArmorPredictor {
public:
    ArmorPredictor(const APParams& params);

    ~ArmorPredictor();

    autoaim_interfaces::msg::Target predict_target(autoaim_interfaces::msg::Armors armors, double dt);

    void reset_kalman();
private:
    void init_kalman();

    Eigen::Vector3d kalman_predict();

    bool judge_spinning(const autoaim_interfaces::msg::Armor& armor);

    Eigen::Vector3d state2position(const Eigen::VectorXd& state);

    void armor_predict(autoaim_interfaces::msg::Armors armors);

    void update_target_type(const autoaim_interfaces::msg::Armor& armor);

    APParams params_;

    autoaim_interfaces::msg::Armor tracking_armor_;
    std::string armor_type_;
    std::string tracking_number_;
    TargetType target_type_;

    uint8_t find_state_ = LOST;
    uint8_t lost_cnt_ = 0;
    uint8_t detect_cnt_ = 0;
    double dt_ = 0.015;

    Eigen::Vector3d target_xyz_;
    Eigen::Vector3d target_v_xyz_;
    Eigen::VectorXd target_state_;

    EigenKalmanFilter kalman_filter_;

    rclcpp::Logger logger_ = rclcpp::get_logger("armor_predictor");
};

} // helios_cv