#pragma once
#include "tracker/ForwardDeclarations.h"
#include "tracker/Types.h"
#include "util/opencv_wrapper.h"
#include "tracker/Detection/TrivialDetector.h"
#include <cuda_runtime.h>

class HandFinder{
private:
    Camera*const camera=NULL;
    TrivialDetector*const trivial_detector=NULL;
public:
    HandFinder(Camera * camera);
	~HandFinder() {
		delete[] sensor_indicator;
	}

/// @{ Settings
public:
    struct Settings{
        bool show_hand = false;
        bool show_wband = false;
        float depth_range = 150;
        float wband_size = 30;
        cv::Scalar hsv_min = cv::Scalar( 94, 111,  37); ///< potentially read from file
        cv::Scalar hsv_max = cv::Scalar(120, 255, 255); ///< potentially read from file
    } _settings;
    Settings*const settings=&_settings;
/// @}

public:
    bool _has_useful_data = false;
    bool _wristband_found;
    Vector3 _wband_center;
    Vector3 _wband_dir;
public:
    cv::Mat sensor_silhouette; ///< created by binary_classifier
	cv::Mat sensor_silhouette_HO;//ZH added
	cv::Mat mask_wristband; ///< created by binary_classifier, not used anywhere else
	int * sensor_indicator;
	int num_sensor_points;
	std::vector<int> point_index;//ZH added
	std::vector<float3> point_cloud;
	std::vector<int> point_index2;//ZH added
	std::vector<float3> point_cloud2;
	float3 camera_dir1;
	float3 camera_dir2;

public:
    bool has_useful_data(){ return _has_useful_data; }
    bool wristband_found(){ return _wristband_found; }
    Vector3 wristband_center(){ return _wband_center; }
    Vector3 wristband_direction(){ return _wband_dir; }
    void wristband_direction_flip(){ _wband_dir=-_wband_dir; }

public:

	void binary_classification(cv::Mat& depth, cv::Mat& color, cv::Mat& real_color);

	void obtain_point_cloud(const int camera_id, const cv::Mat &silhouette, const cv::Mat &depth_map, const Eigen::Matrix4f &camera_pose, const float fx, const float fy, const float cx, const float cy);

	void obtain_camera_direction(const Eigen::Vector3f left_camera_dir, const Eigen::Vector3f right_camera_dir);

	/************************************************************************/
	/* No use																*/
	/************************************************************************/
	void binary_classification_wrist(const cv::Mat & depth);
	void extract_skin_by_HSV(cv::Mat &color_hsv, cv::Mat &skin_mask);
	void extract_skin_by_HSV_glove(cv::Mat &color_hsv, cv::Mat &skin_mask);
	void extract_hand_object_by_marker(cv::Mat& depth, cv::Mat& color, cv::Mat& hand_object_silhouette, cv::Mat &wband_cut, bool is_left);
	void extract_hand_object_by_marker2(cv::Mat& depth, cv::Mat& color, cv::Mat& hand_object_silhouette, cv::Mat &wband_cut, bool is_left);
	void extract_hand_object_by_marker3(cv::Mat& depth, cv::Mat& color, cv::Mat& hand_object_silhouette, cv::Mat &wband_cut, bool& is_wristband_found, bool& has_useful_data, Vector3& wband_center, Vector3& wband_dir, cv::Mat& mask_wristband_temp);
	void extract_hand_object_by_marker4(cv::Mat& depth, cv::Mat& color, cv::Mat& hand_object_silhouette, cv::Mat &wband_cut, bool& is_wristband_found, bool& has_useful_data, Vector3& wband_center, Vector3& wband_dir, cv::Mat& mask_wristband_temp);
	void obtain_hand_silhouette(int sequence_num, cv::Mat& depth, cv::Mat& color, cv::Mat& color_320, cv::Mat& hand_object_silhouette, cv::Mat& hand_silhouette, cv::Mat& object_silhouette);
	void obtain_hand_object_silhouette(int sequence_num, cv::Mat& depth_320, cv::Mat& color_320, cv::Mat& hand_object_silhouette);
	void obtain_point_cloud1(const cv::Mat &silhouette1, const cv::Mat &depth_map1, const Eigen::Matrix4f &camera_pose, const float fx, const float fy, const float cx, const float cy);
	void obtain_point_cloud2(const cv::Mat &silhouette2, const cv::Mat &depth_map2, const Eigen::Matrix4f &camera_pose, const float fx, const float fy, const float cx, const float cy);

};
