#include "Filter.h"

void OneEuroFilter::init(int para_num_set)
{
	para_num = para_num_set;
	dx.resize(para_num);
	x_cutoff.resize(para_num);
}

std::vector<float> OneEuroFilter::LowPassFilter_run( std::vector<float> value, std::vector<float> prev_value, std::vector<float> alpha)
{
	std::vector<float> filtered_value;
	
	if (prev_value.size() > 0)
	{
		filtered_value.resize(value.size());
		for (int i = 0; i < value.size(); i++)
		{
			filtered_value[i] = alpha[i] * value[i] + (1 - alpha[i])*prev_value[i];
		}
	}
	else
	{
		filtered_value = value;
	}

	return filtered_value;
}

std::vector<float> OneEuroFilter::compute_alpha(float cutoff)
{
	std::vector<float> alpha;
	alpha.resize(para_num);

	for (int i = 0; i < para_num; i++)
	{
		float te = 1.0 / freq;
		float tau = 1.0 / (2 * pi*cutoff);
		alpha[i] = 1.0 / (1.0 + tau / te);
	}

	return alpha;
}

std::vector<float> OneEuroFilter::compute_alpha(std::vector<float> cutoff)
{
	std::vector<float> alpha;
	alpha.resize(para_num);

	for (int i = 0; i < para_num; i++)
	{
		float te = 1.0 / freq;
		float tau = 1.0 / (2 * pi*cutoff[i]);
		alpha[i] = 1.0 / (1.0 + tau / te);
	}

	return alpha;
}


std::vector<float> OneEuroFilter::OneEuroFilter_run(std::vector<float> value)
{
	std::vector<float> x_filtered_value;

	//if (init_flag)
	{
		//compute dx
		if (x_prev_raw_value.size() > 0)
		{
			for (int i = 0; i < para_num; i++)
				dx[i] = value[i] - x_prev_raw_value[i];
		}
		else
		{
			for (int i = 0; i < para_num; i++)
				dx[i] = 0;
		}

		//compute edx
		dx_alpha = compute_alpha(dcutoff);
		edx = LowPassFilter_run(dx, dx_prev_filtered_value, dx_alpha);
		dx_prev_filtered_value = edx;

		//compute x_cutoff
		for (int i = 0; i < para_num; i++)
			x_cutoff[i] = mincutoff + beta*abs(edx[i]);

		//compute x_alpha
		x_alpha = compute_alpha(x_cutoff);

		//obtain filtered value
		x_filtered_value = LowPassFilter_run(value, x_prev_filtered_value, x_alpha);
	}
	//else
	//{
	//	//assign dx as 0
	//	for (int i = 0; i < para_num; i++)
	//		dx[i] = 0;

	//	//assign edx, dx_pre_filtered_value
	//	edx = dx;
	//	dx_prev_filtered_value = edx;
	//	x_filtered_value = value;

	//	init_flag = true;
	//}

	//assign last raw value
	x_prev_raw_value = value;
	x_prev_filtered_value = x_filtered_value;

	return x_filtered_value;
}