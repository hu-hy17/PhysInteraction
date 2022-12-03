#include <vector>

//class LowPassFilter
//{
//public:
//	
//
//private:
//	bool init_flag = false;
//	std::vector<float> prev_filtered_value;
//};

class OneEuroFilter
{
public:
	std::vector<float> LowPassFilter_run(std::vector<float> value, std::vector<float> prev_value, std::vector<float> alpha);
	std::vector<float> OneEuroFilter_run(std::vector<float> value);
	std::vector<float> compute_alpha(float cutoff);
	std::vector<float> compute_alpha(std::vector<float> cutoff);

	void init(int para_num=29);

private:
	std::vector<float> x_prev_raw_value;
	std::vector<float> x_prev_filtered_value;
	std::vector<float> x_cutoff;
	std::vector<float> x_alpha;

	std::vector<float> dx;
	std::vector<float> dx_prev_filtered_value;
	std::vector<float> edx;
	std::vector<float> dx_alpha;

	float mincutoff = 10.0;//10.0
	float dcutoff = 1.0;
	float beta = 2.0;//2.0
	bool init_flag = false;

	float freq = 30;
	float para_num = 29;
	const float pi = 3.14159265;
};