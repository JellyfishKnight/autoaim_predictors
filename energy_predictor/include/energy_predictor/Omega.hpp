#ifndef OMEGA_HPP
#define OMEGA_HPP
#include"iostream"
#include"cmath"

#include"eigen3/Eigen/Core"
#include"eigen3/Eigen/Dense"

#include"opencv4/opencv2/core.hpp"

class Omega
{
public:
    Omega();
    void refresh();
    void refresh_after_wave();
    void set_theta(float theta);
    bool FindWavePeak();
    float get_omega();
    float get_rad(float latency);
    float get_time_gap();
    void set_time(double time_t);
    void set_filter(float omega_filter);
    float get_err();
    void change_st();

private:
    float IdealOmega(float &t_);
    float IdealRad(float t1, float t2);
    int round2int(float f);
    float calOmegaNstep(int step, float &total_theta);
    Eigen::MatrixXd LeastSquare(std::vector<float>x, std::vector<float>y, int N);
    void JudgeFanRotation(float omega);
    void change_time_series();

public:

    std::vector<float>angle_;
    std::vector<float>filter_omega_;
    float total_theta_;
    float current_theta_;
    int st_;
    int fit_cnt_;
    float dt_;
    int energy_rotation_direction_;
    bool start_;
    std::vector<float> t_list_, time_series_;
    double phi_, w_, a_;

private:
    int change_cnt_;
    int differ_step_;
    double time_start_, time_fin_, time_gap_;
    int clockwise_cnt_;
    float max_omega_, min_omega_;




};


#endif