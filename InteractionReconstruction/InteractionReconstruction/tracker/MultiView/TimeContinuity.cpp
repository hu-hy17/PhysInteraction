#include "TimeContinuity.h"

// change the mask value to 0(background), 1(object), 2(hand)
void unify(cv::Mat &a) {
	//	cv::Mat b = a.clone();
	for (cv::Mat_<uchar>::iterator it = a.begin<uchar>(); it != a.end<uchar>(); it++) {
		if (*it < 20)
			*it = 0;
		else if (*it > 180)
			*it = 2;
		else
			*it = 1;
	}
	//	return b;
}

void de_unify(cv::Mat &a) {
	for (cv::Mat_<uchar>::iterator it = a.begin<uchar>(); it != a.end<uchar>(); it++) {
		if (*it == 1)
			*it = 127;
		else if (*it == 2)
			*it = 255;
	}
}

float iou(const cv::Mat &a, const cv::Mat &b, std::vector<int> labels = { 1, 2 }) {
	cv::Mat inter, uni;
	float iou_val = 0;
	for (int label : labels) {
		inter = (a == label & b == label) / 255;
		uni = (a == label | b == label) / 255;
		iou_val += cv::sum(inter)[0] / cv::sum(uni)[0];
	}
	return iou_val / labels.size();
}


cv::Mat* TimeContinuityProcess::process(const cv::Mat* frame, bool de_unified, float iou_threshold) {
	// Denoise
	cv::Mat* new_frame = new cv::Mat;
	*new_frame = *frame;
	//		*new_frame = denoise_filter(*frame);

	int last_valid_id = -1;
	ious.clear();
	for (int i = 0; i < q.size(); i++) {
		if (q[i].second) {
			ious.push_back(iou(*new_frame, q[i].first));
			last_valid_id = i;
		}
	}

	//		bool is_valid = contour_valid(*new_frame);
	bool is_valid = true;
	if (is_valid && ious.size() > 0) {
		float iou_sum = 0;
		for (auto iu : ious)
			iou_sum += iu;
		float mean_frame_iou = iou_sum / ious.size();
		if (mean_frame_iou < iou_threshold)
			is_valid = false;
	}

	cv::Mat* output = new cv::Mat;
	if (!is_valid && last_valid_id > -1)
		*output = q[last_valid_id].first.clone();
	else
		*output = (*new_frame).clone();
	if (de_unified)
		de_unify(*output);

	if (q.size() >= buffer_size)
		q.pop_front();
	if (is_valid)
		q.push_back({ *new_frame, is_valid });

	new_frame->release();
	return output;
}

void TimeContinuityProcess::reset() {
	q.clear();
	ious.clear();
}

bool TimeContinuityProcess::contour_valid(const cv::Mat a, int num_mismatched_point_threshold) {
	std::vector<cv::Mat> masks;
	cv::Mat compar;
	//		static std::vector<uchar> compar_data2;
	compar = (a == 2);
	//		compar_data2 = std::vector<uchar>(compar.datastart, compar.dataend);
	masks.push_back(compar);// / 255);
							//		cv::imshow("2", compar);

							//		static std::vector<uchar> compar_data1;
	compar = (a == 1);
	//		compar_data1 = std::vector<uchar>(compar.datastart, compar.dataend);
	masks.push_back(compar);// / 255);
							//		cv::imshow("1", compar);
							//		cv::waitKey(3);

	static std::vector<uchar> contour_uchar;
	for (auto mask : masks) {
		cv::Mat mask_temp = mask.clone();
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(mask_temp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		cv::Mat contours_mask(mask_temp.rows, mask_temp.cols, mask_temp.type(), cv::Scalar(0));
		cv::drawContours(contours_mask, contours, -1, cv::Scalar(255), CV_FILLED);
		float num_mismatched_point = cv::sum((mask != contours_mask) / 255)[0];// 255
		if (num_mismatched_point > num_mismatched_point_threshold)
			return false;
	}
	return true;
}

cv::Mat TimeContinuityProcess::denoise_filter(const cv::Mat a) {
	cv::Mat hand_mask = a == 2;
	cv::Mat object_mask = a == 1;

	cv::Mat big_k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat small_k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));

	cv::Mat new_hand_mask, new_object_mask;
	cv::morphologyEx(hand_mask, new_hand_mask, cv::MORPH_CLOSE, big_k);
	cv::morphologyEx(new_hand_mask, new_hand_mask, cv::MORPH_OPEN, small_k);
	cv::morphologyEx(object_mask, new_object_mask, cv::MORPH_CLOSE, small_k);
	cv::morphologyEx(new_object_mask, new_object_mask, cv::MORPH_OPEN, big_k);

	cv::Mat new_mat = 2 * (new_hand_mask > 0) / 255 + (new_object_mask > 0) / 255;
	return new_mat;
}