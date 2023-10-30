// created by liuhan on 2023/9/29
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink

#pragma once
#include"eigen3/Eigen/Core"
#include"eigen3/Eigen/Dense"


namespace helios_cv {

class EigenKalmanFilter{
public:
    EigenKalmanFilter(int state_param, int measure_param, int control_param = 0);
    void Init(int state_param, int measure_param, int control_param=0);
    void Update(Eigen::VectorXf &measurement);

    void initKalman();
    Eigen::MatrixXf predict(const Eigen::MatrixXf &control = Eigen::MatrixXf());
    Eigen::MatrixXf correct(const Eigen::VectorXf &measurement);

    Eigen::VectorXf state_pre_;// k时刻先验估计
    Eigen::VectorXf state_post_;//k时刻后验估计

    Eigen::MatrixXf trans_mat_;//状态转移矩阵
    Eigen::MatrixXf control_mat_;//控制矩阵
    Eigen::MatrixXf measure_mat_;//测量转移矩阵
    Eigen::MatrixXf process_noise_;//过程激励噪声协方差矩阵
    Eigen::MatrixXf measure_noise_;//测量噪声协方差矩阵
    Eigen::MatrixXf error_pre_;
    Eigen::MatrixXf gain_;//卡尔曼增益系数
    Eigen::MatrixXf error_post_;

private:
    Eigen::MatrixXf temp1;
    Eigen::MatrixXf temp2;
    Eigen::MatrixXf temp3;
    Eigen::MatrixXf temp4;

};

} // namespace helios_cv
