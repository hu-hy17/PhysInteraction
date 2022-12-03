#pragma once
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/contrib/contrib.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <iostream>  
#include <vector>  
#include <time.h>
#include <windows.h>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <Eigen/Eigen>
#include <cuda_runtime.h>
#include "reconstruction/include/gpu/vector_operations.hpp"
#include "reconstruction/include/data_process.h"
#include "tracker/CommonVariances.h"
#include "curl/curl.h"
#include "tracker/HandFinder/connectedComponents.h"
//#include "TimeContinuity.h"
#include <deque>
#include <thread>

#include <json/json.h>

#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>


using namespace std;
using namespace cv;

struct image_bias
{
	int u_bias;
	int v_bias;
};

struct Point3D
{
	float x;
	float y;
	float z;
};

class TimeContinuityProcess {
private:
	std::deque<std::pair<cv::Mat, bool>> q;
	int buffer_size;
	std::vector<float> ious;

public:
	TimeContinuityProcess(int buffer_size = 5) {
		this->buffer_size = buffer_size;
	}

	cv::Mat process(const cv::Mat frame, bool& is_valid, bool de_unified = true, float iou_threshold = 0.8);

	cv::Mat process2(const cv::Mat frame, bool& is_valid, bool de_unified = true, float iou_threshold = 0.8);

	void reset();

	bool contour_valid(const cv::Mat a, int num_mismatched_point_threshold = 50);

	cv::Mat denoise_filter(const cv::Mat a);
};

class DataParser
{
public:

	void initial_parsar(int sequence_num, image_bias image_bias_value, camera_intr color_intrinsic, camera_intr depth_intrinsic, camera_intr camera_para, Eigen::Matrix4f depth2color_extrinsic, Eigen::Matrix4f camera_pose/*, HandFinder* handfinder*/);

	void set_online_org_data(cv::Mat& depth_img, cv::Mat& color_img);

	void load_org_data(std::string data_path, int current_frame);

	void obtain_resized_data2();

	void obtain_aligned_data2();

	void show_original_data();

	void show_result_data();

	bool Base64Decode(const string & input, string * output);

	cv::Mat get_color_bgr();

	cv::Mat get_color_bgr_320();

	cv::Mat get_depth_mm_320();

	cv::Mat get_org_color_320();

	cv::Mat get_aligned_color_320();

	int get_sequence_number() { return _sequence_id; };

	cv::Mat draw_3Dkeypoint2image(std::vector<float3> &local_key_points, cv::Vec3b color_value);
	cv::Mat draw_2Dkeypoint2image(std::vector<float2> &key_points, cv::Vec3b color_value);

	cv::Mat TrackedHand_3DKeypointsProjection2Dimage(const std::vector<float3> global_key_points, std::vector<float2> &image_2D_keypoints);

	cv::Mat project_keypoint2image640(std::vector<float3> &key_points, std::vector<float2> &key_points_pixel);

	mat34 get_camera_pose_mat34();

	void cal_depth_map_texture();

	void TransformDistance(unsigned char *label_image, int mask_th, int width, int height, std::vector<float> &realDT, std::vector<float> &DTTps, std::vector<int> &ADTTps, std::vector<int> &realADT,
		std::vector<float> &v, std::vector<float> &z);

	cv::Mat extract_hand_object_by_marker3(cv::Mat& depth, cv::Mat& color/*, cv::Mat& hand_object_silhouette*//*, cv::Mat &wband_cut,*//* bool& is_wristband_found,*/ /*bool& has_useful_data,*/ /*Eigen::Vector3f& wband_center,*/ /*Eigen::Vector3f& wband_dir,*//* cv::Mat& mask_wristband_temp*/);
	
	void obtain_hand_object_silhouette();

	void cal_ADT();

	cv::Mat hand_object_segment_320_id_pure();

	cv::Mat hand_object_segment_320_id_init_pure();

	void HandObjectSil_CalADT();

	void HandObjectSegmentation_init();
	void HandObjectSegmentation();

	void CalculateObjADT();

	void ObtainJoints();

	cv::Mat get_segmentation_greyC3();
	cv::Mat get_segmentation_color();
	cv::Mat get_segmentation_org();

	cv::Mat get_keypoint_pred_color();

	~DataParser() { free(chunk1.memory); }

public:

	//for stage1: preprocessing
	cv::Mat _color_org_pre;
	cv::Mat _depth_org_pre;
	cv::Mat _color_org_320_pre;
	cv::Mat _depth_org_320_pre;
	cv::Mat _depth_mm_320_pre;
	cv::Mat _aligned_color_320_pre;
	cv::Mat _hand_object_silhouette_pre;

	//for segmentation, ADT, joint_pred
	cv::Mat _color_org_320_seg_in;
	cv::Mat _depth_org_320_seg_in;
	cv::Mat _depth_mm_320_seg_in;
	cv::Mat _aligned_color_320_seg_in;
	cv::Mat _hand_object_silhouette_seg_in;

	cv::Mat _color_org_320_seg_out;
	cv::Mat _depth_org_320_seg_out;
	cv::Mat _depth_mm_320_seg_out;
	cv::Mat _aligned_color_320_seg_out;
	cv::Mat _hand_object_silhouette_seg_out;
	cv::Mat _segmentation_org_seg;
	cv::Mat _hand_silhouette_seg;
	cv::Mat _object_silhouette_seg;
	std::vector<int> realADT_seg;
	std::vector<int> realADT_obj_seg;
	std::vector<float3> _joints_pred_xyz_seg;
	std::vector<float2> _joints_pred_uv_seg;
	

	//for main pipeline
	cv::Mat _color_org_320;
	cv::Mat _depth_org_320;
	cv::Mat _segmentation_org;
	cv::Mat _depth_mm_320;
	cv::Mat _aligned_color_320;
	cv::Mat _hand_object_silhouette;
	cv::Mat _hand_silhouette;
	cv::Mat _object_silhouette;
	std::vector<int> realADT;
	std::vector<int> realADT_obj;
	std::vector<float3> _joints_pred_xyz;
	std::vector<float2> _joints_pred_uv;
	
	
	//intermediate variables 
	cv::Mat _aligned_color_bgr_320;
	cv::Mat _segmentation;
	std::vector<float> _joints_uvz;

	Json::Reader reader;
	Json::Value json_value;
	

	Eigen::Matrix4f _camera_RT;
	Eigen::Vector3f _camera_dir;
	camera_intr _depth_para;
	DataProcess _depth_processor;
	TimeContinuityProcess process_pipeline;

	LARGE_INTEGER time_stmp;
	double count_freq, count_interv, time_inter;

	curl_mime *form = nullptr;
	curl_mimepart *field = nullptr;

public:
	struct MemoryStruct {
		char *memory;
		size_t size = 0;
	};

	static size_t
		WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp)
	{
		size_t realsize = size * nmemb;
		struct MemoryStruct *mem = (struct MemoryStruct *)userp;

		char *ptr = (char *)realloc(mem->memory, mem->size + realsize + 1);
		if (!ptr) {
			/* out of memory! */
			printf("not enough memory (realloc returned NULL)\n");
			return 0;
		}

		mem->memory = ptr;
		memcpy(&(mem->memory[mem->size]), contents, realsize);
		mem->size += realsize;
		mem->memory[mem->size] = 0;

		return realsize;
	}

	static size_t
		WriteMemoryCallback2(void *contents, size_t size, size_t nmemb, void *userp)
	{
		size_t realsize = size * nmemb;
		struct MemoryStruct *mem = (struct MemoryStruct *)userp;

		//char *ptr = (char *)realloc(mem->memory, mem->size + realsize + 1);
		//if (!ptr) {
		//	/* out of memory! */
		//	printf("not enough memory (realloc returned NULL)\n");
		//	return 0;
		//}

		/*mem->memory = ptr;*/

		memcpy(&(mem->memory[mem->size]), contents, realsize);
		mem->size += realsize;
		mem->memory[mem->size] = 0;

		return realsize;
	}

	struct MemoryStruct chunk1;

private:
	cv::Mat _depth_gray;

	cv::Mat _aligned_color;
	cv::Mat _aligned_color_bgr;
	cv::Mat _depth_mm;

	//for data preprocessing pipeline
	int invalid_num;

	camera_intr _color_intr, _depth_intr;
	Eigen::Matrix4f _depth2color_extr;
	int _u_bias, _v_bias;
	int _sequence_id;
	std::string camera_id;
	float _depth_scale = 8000.0f;
	int _depth_to_mm_scale = 8;

//	Eigen::Vector3f wband_old_dir = Eigen::Vector3f(0, 0, 0);
	std::deque<Eigen::Vector3f> wband_dir_array; 
	
	int width = 320;
	int height = 240;

	int frame_idx = 0;
	
	std::vector<float> realDT;
	std::vector<float> DTTps;
	std::vector<int> ADTTps;
	std::vector<float> v;
	std::vector<float> z;

	std::vector<float> realDT_obj;
	std::vector<float> DTTps_obj;
	std::vector<int> ADTTps_obj;
	std::vector<float> v_obj;
	std::vector<float> z_obj;

//	cv::Mat segimage;
};