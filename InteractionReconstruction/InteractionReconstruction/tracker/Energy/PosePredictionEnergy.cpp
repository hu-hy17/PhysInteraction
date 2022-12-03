#include "PosePredictionEnergy.h"
#include <set>

void energy::PosePrediction::track(LinearSystem& sys, const std::vector<Scalar> &current_theta, const std::vector<float>& theta_predict)
{
	Eigen::Matrix<Scalar, num_thetas, num_thetas> JTJ = Eigen::Matrix<Scalar, num_thetas, num_thetas>::Zero(num_thetas, num_thetas);
	Eigen::Matrix<Scalar, num_thetas, 1> JTe = Eigen::Matrix<Scalar, num_thetas, 1>::Zero(num_thetas, 1);

	for (size_t i = 0; i < num_thetas; ++i) {
		JTJ(i, i) = 0;
		if (i >= 7)
		{
			JTJ(i, i) = 1;
			JTe(i) = theta_predict[i] - current_theta[i];//theta_predict[i] 
//			JTe(i,1) = theta_predict[i] - current_theta[i];//JTe(i,1) is error index, the right is JTe(i,0) or JTe(i)
		}
		/*if(i==0)
			JTe(i) = 10 - current_theta[i];
		if(i==1)
			JTe(i) = -60 - current_theta[i];
		if(i==2)
			JTe(i) = 430 - current_theta[i];*/
	}

	//Eigen::Matrix<Scalar, num_thetas, num_thetas> JT = J.transpose();
	sys.lhs += posepred_weight * JTJ;
	sys.rhs += posepred_weight * JTe;

	///--- Check
	if (Energy::safety_check) Energy::has_nan(sys);
}

void energy::PhysKinDiff::track(LinearSystem& sys, const std::vector<float> &cur_theta, const std::vector<float>& theta_kin)
{
	Eigen::Matrix<Scalar, num_thetas, num_thetas> JTJ = Eigen::Matrix<Scalar, num_thetas, num_thetas>::Zero(num_thetas, num_thetas);
	Eigen::Matrix<Scalar, num_thetas, 1> JTe = Eigen::Matrix<Scalar, num_thetas, 1>::Zero(num_thetas, 1);

	std::set<int> ignore_joint_idx{
		0, 1, 2, 3, 4, 5, 6, 0, 10, 11, 12, 13, 14, 15, 16
	};

	for (size_t i = 0; i < 7; ++i) {
		JTJ(i, i) = 1e4;
		JTe(i) = theta_kin[i] - cur_theta[i];
	}

	//Eigen::Matrix<Scalar, num_thetas, num_thetas> JT = J.transpose();
	sys.lhs += diff_weight * JTJ;
	sys.rhs += diff_weight * JTe;

	///--- Check
	if (Energy::safety_check) Energy::has_nan(sys);
}