#pragma once
#include<iostream>
#include<string>
#include<vector>
#include<deque>
//#include<utility>
//#include<filesystem>
#include<opencv2/opencv.hpp>
//#include <ctime>
//#include <windows.h>

class TimeContinuityProcess {
private:
	std::deque<std::pair<cv::Mat, bool>> q;
	int buffer_size;
	std::vector<float> ious;

public:
	TimeContinuityProcess(int buffer_size = 5) {
		this->buffer_size = buffer_size;
	}

	cv::Mat* process(const cv::Mat* frame, bool de_unified = true, float iou_threshold = 0.8);

	void reset();

	bool contour_valid(const cv::Mat a, int num_mismatched_point_threshold = 50);

	cv::Mat denoise_filter(const cv::Mat a);
};