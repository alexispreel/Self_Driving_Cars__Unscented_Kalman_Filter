#include <iostream>
#include "ukf.h"
#include "Eigen/Dense"
//#include "gnuplot_i.hpp"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
//using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    is_initialized_ = false;
    previous_timestamp_ = 0;
    
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;
    
    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;
    
    // State dimension
    n_x_ = 5;
    
    // Augmented state dimension
    n_aug_ = 7;
    
    // Initializing state vector
    x_ = VectorXd(n_x_);
    
    
    // Process noise standard deviation longitudinal acceleration in m/s^2
    // Determined empirically using sample data set
    std_a_ = 2.6;
    
    // Process noise standard deviation yaw acceleration in rad/s^2
    // Determined empirically using sample data set
    std_yawdd_ = 0.6;
    
    
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;
    
    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;
    
    // Initializing measurement covariance matrix - laser
    R_laser_ = MatrixXd(2, 2);
    R_laser_ << std_laspx_ * std_laspx_, 0,
                0, std_laspy_ * std_laspy_;
    
    
    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;
    
    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;
    
    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    
    // Initializing measurement covariance matrix - radar
    R_radar_ = MatrixXd(3, 3);
    R_radar_ << std_radr_ * std_radr_, 0, 0,
                0, std_radphi_ * std_radphi_, 0,
                0, 0, std_radrd_ * std_radrd_;
    
    
    // Initializing covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    
    
    // Sigma points spreading parameter
    lambda_ = 3 - n_x_;
    
    // Initializing sigma points matrix
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    
    
    // Initializing vector for sigman points weights_
    weights_ = VectorXd(2 * n_aug_ + 1);
    double d = lambda_ + n_aug_;
    weights_(0) = lambda_ / d;
    for (int i = 1; i < 2 * n_aug_ + 1; i++){
        weights_(i) = 0.5 / d;
    }
    
    // For plotting
    
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    
    // Initialization
    if (!is_initialized_) {
        // First measurement
        cout << "UKF: " << endl;
        x_ = VectorXd(5);
        
        // Initializing covariance matrix P with zeros
        P_.fill(0.0);
        
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            // Getting initial radar measurement: range, angle, range rate
            float rho = meas_package.raw_measurements_[0];
            float phi = meas_package.raw_measurements_[1];
            float rhod = meas_package.raw_measurements_[2];
            
            // Converting to cartesian coordinates
            x_ << rho * cos(phi), rho * sin(phi), 0.0, 0.0, 0.0;
            
            // Initializing covariance matrix P: position
            // Covariance in x and y position: using radar noise parameters (std_radr_ and std_radphi_, converted to cartesian coordinates)
            P_(0,0) = std_radr_ * cos(std_radphi_) * std_radr_ * cos(std_radphi_);
            P_(1,1) = std_radr_ * sin(std_radphi_) * std_radr_ * sin(std_radphi_);
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            // Getting initial lidar measurement: position
            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0.0;
            
            // Initializing covariance matrix P: position
            // Covariance in x and y position: using lidar noise parameters (std_laspx_ and std_laspy_)
            P_(0,0) = std_laspx_ * std_laspx_;
            P_(1,1) = std_laspy_ * std_laspy_;
        }
        
        // Initializing covariance matrix P: velocity, yaw and yaw rate
        // Covariance in velocity: using process longitudinal acceleration noise (std_a_)
        P_(2,2) = std_a_ * std_a_;
        // Covariance in yaw: using pi/2 (arbitrary, max noise)
        P_(3,3) = (M_PI/2) * (M_PI/2);
        // Covariance in yaw rate: using pi/2 + process yaw acceleration noise (std_yawdd_)
        P_(4,4) = ((1 + std_yawdd_) * M_PI/2) * ((1 + std_yawdd_) * M_PI/2);
        
        // Done initializing, no need to predict or update
        previous_timestamp_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
        
        // For plotting
        //Gnuplot g1("NIS");
    }
    
    // Computing time elapsed between measurements in seconds
	double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
	previous_timestamp_ = meas_package.timestamp_;
    
    // Predicting - Running CTRV model
    Prediction(delta_t);
    
    // Updating state with sensor measurement
    UpdateSensor(meas_package);
    
    // Printing output
    cout << "x_ = " << x_ << endl;
    cout << "P_ = " << P_ << endl;
    cout << "NIS_radar_ = " << NIS_radar_ << endl;
    cout << "NIS_laser_ = " << NIS_laser_ << endl;
    
    // For plotting
    //timestamps_.push_back((double)meas_package.timestamp_);
    //line_.push_back((double)7.8);
    //NIS_lasers_.push_back((double)NIS_laser_);
    //NIS_radars_.push_back((double)NIS_radar_);
    
    
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    
    // Initializing augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);
    
    // Initializing augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    
    // Initializing augmented sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    
    // Creating augmented mean state
    x_aug.head(n_x_) = x_;
    x_aug(n_x_) = 0.0;
    x_aug(n_x_ + 1) = 0.0;
    
    // Creating augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_ * std_a_;
    P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
    
    // Calculating square root matrix
    MatrixXd A = P_aug.llt().matrixL();
    
    // Creating augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    double factor = sqrt(n_aug_ + lambda_);
    for (int i = 0; i < n_aug_; i++){
        Xsig_aug.col(i + 1) = x_aug + factor * A.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - factor * A.col(i);
    }
    
    
    // Predicting sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++){
        // Extracting values for easier usage
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);
        
        // Initializing state values
        double px_p, py_p;
        
        // Applying CTRV equations
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }
        
        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;
        
        // Adding noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;
        
        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;
        
        // Writing predicted sigma points
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }
    
    
    // Calculating predicted state mean
    x_ = Xsig_pred_ * weights_;
    
    // Recomputing covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        // Calculating state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        
        // Normalizing yaw
        while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;
        
        // Recomputing covariance matrix
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
}

/**
 * Updates the state and the state covariance matrix using a sensor measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateSensor(MeasurementPackage meas_package) {
    
    // Getting measurement and measurement dimension
    VectorXd z = meas_package.raw_measurements_;
    int n_z = z.size();
    
    // Initializing matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    
    // Initializing mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    
    // Initializing measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    
    Zsig.fill(0.0);
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        // Transforming sigma points into measurement space (polar coordinates)
        for (int i = 0; i < 2 * n_aug_ + 1; i++) {
            // Extracting values for easier usage
            double p_x = Xsig_pred_(0, i);
            double p_y = Xsig_pred_(1, i);
            double v = Xsig_pred_(2, i);
            double yaw = Xsig_pred_(3, i);
            double yawd = Xsig_pred_(4, i);
            
            // Computing radar range, angle and range rate from Cartesion coordinates
            double rho = sqrt(p_x * p_x + p_y * p_y);
            double phi = atan2(p_y, p_x);
            double rhod = (p_x * v * cos(yaw) + p_y * v * sin(yaw)) / rho;
            
            // Updating sigma points matrix
            Zsig(0, i) = rho;
            Zsig(1, i) = phi;
            Zsig(2, i) = rhod;
        }
    }
    else {
        // Updating sigma points matrix with laser measurements
        for (int i = 0; i < 2 * n_aug_ + 1; i++) {
            // Updating sigma points matrix
            Zsig(0, i) = Xsig_pred_(0, i);
            Zsig(1, i) = Xsig_pred_(1, i);
        }
    }
    
    // Calculating mean predicted measurement
    z_pred = Zsig * weights_;
    
    // Computing measurement covariance matrix S
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        // Getting difference from mean
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        // Computing measurement covariance matrix S
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    
    // Adding measurement noise
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        S += R_radar_;
    }
    else {
        S += R_laser_;
    }
    
    
    // Initializing matrix for cross-correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    
    // Calculating cross-correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        // Getting state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        
        // Normalizing yaw
        while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;
        
        // Getting difference from mean
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        // Normalizing yaw
        while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;
        
        // Computing Tc
        Tc += weights_(i) * x_diff * z_diff.transpose();
    }
    
    // Calculating Kalman gain K
    MatrixXd Si = S.inverse();
    MatrixXd K = Tc * Si;
    
    // Getting measurement difference
    VectorXd z_diff = z - z_pred;
    
    // Normalizing yaw
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;
    
    // Updating state mean
    x_ += K * z_diff;
    
    // Updating covariance matrix
    P_ -= K * S * K.transpose();
    
    // Computing Normalized Innovation Squared
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        NIS_radar_ = z_diff.transpose() * Si * z_diff;
    }
    else {
        NIS_laser_ = z_diff.transpose() * Si * z_diff;
    }
}