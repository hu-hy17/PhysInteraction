#pragma once
#include "Energy.h"
#include "tracker/Types.h"
#include <vector>

namespace energy {

	class PosePrediction : public Energy {
	public:
		float posepred_weight = 20000;

	public:
		void track(LinearSystem& sys, const std::vector<Scalar> &theta_0, const std::vector<float>& theta_predict/*, bool store_error, float &limit_error*/);
	};


	class PhysKinDiff : public Energy {
	public:
		float diff_weight = 1;
	public:
		void track(LinearSystem& sys, const std::vector<float> &theta, const std::vector<float>& theta_kin);
	};

}
