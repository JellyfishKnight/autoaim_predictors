#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

namespace helios_cv {

class ExtendedKalmanFilter
{
public:
    ExtendedKalmanFilter() = default;

    ExtendedKalmanFilter(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& F,
                         const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& H,
                         const std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>& Jf,
                         const std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>& Jh,
                         const std::function<Eigen::MatrixXd()>& UpdateQ, 
                         const std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>& UpdateR,
                         const Eigen::MatrixXd& P);

    Eigen::MatrixXd Predict();

    Eigen::MatrixXd Correct(const Eigen::VectorXd& z);

    void setState(const Eigen::VectorXd& state);

    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h;
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> jf;
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> jh;
    std::function<Eigen::MatrixXd()> update_Q;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> update_R;
    Eigen::MatrixXd F;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::VectorXd state_pre_;
    Eigen::VectorXd state_post_;
    Eigen::MatrixXd P_pre_;
    Eigen::MatrixXd P_post_;
    Eigen::MatrixXd Gain_;
};

} // namespace helios_cv