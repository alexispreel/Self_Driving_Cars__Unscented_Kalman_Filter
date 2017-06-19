#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    /**
    TODO:
    * Calculate the RMSE here.
    */
    // Initializing RMSE
    VectorXd rmse(4);
	rmse << 0,0,0,0;
    
    // Checking inputs
	if(estimations.size() != ground_truth.size() || estimations.size() == 0){
		cout << "Invalid inputs for RMSE" << endl;
		return rmse;
	}
    
	// Cumulating squared residuals
	for(unsigned int i=0; i < estimations.size(); ++i){
        
		VectorXd r = estimations[i] - ground_truth[i];
        
		//coefficient-wise multiplication
		r = r.array() * r.array();
		rmse += r;
	}
    
	// Calculating mean
	rmse = rmse / estimations.size();
    
	// Calculating squared root
	rmse = rmse.array().sqrt();
    
	// Return result
	return rmse;
}