#pragma once
#include <QTimer>
#include <QObject>
#include "util/mylogger.h"
#include "util/tictoc.h"
#include "tracker/ForwardDeclarations.h"
// #include "tracker/Sensor/Sensor.h"
#include "tracker/Data/DataStream.h"
#include "tracker/Worker.h"
#include "tracker/Data/SolutionStream.h"
#include "tracker/Detection/QianDetection.h"
#include "tracker/Data/TextureColor8UC3.h"
#include "tracker/Data/TextureDepth16UC1.h"
#include "tracker/TwSettings.h"
#include "tracker/HModel/Model.h"
#include "HandFinder/connectedComponents.h" 
#include "VarianceAssistance.h"
#include "reconstruction/include/ObjectReconstruction.h"
#include "reconstruction/include/VariantSmooth.h"
#include "reconstruction/include/recon_externs.h"
#include "reconstruction/include/solver/rigid_icp.h"
#include "physhand/PhysHandSolver.h"
#include "physhand/Defs.h"
#include "physhand/Utils.h"
#include "cuda_profiler_api.h"
#include <json/json.h>
#include "Interaction.h"
#include "Sensor/RealSenseSR300.h"
#include "PosePredict.h"

#include <ctime>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <thread> 

class Tracker : public QTimer {
public:
	enum Mode { LIVE, BENCHMARK } mode = LIVE;

	/************************************************************************/
	/* Datastream                                                           */
	/************************************************************************/
	Sensor* sensor;
	DataStream* datastream;
	SolutionStream* solutions;
	Worker* worker = NULL;//Worker*const worker = NULL;
	RealSenseSR300& sensor_sr300;
	std::string data_path;
	std::string file_left_camera = "D:/Project/Datas/20181119/sequence/1/617204007612";
	std::string file_right_camera = "D:/Project/Datas/20181119/sequence/1/619204001397";
	std::string input_store_path = "D:/Project/HandReconstruction/hmodel-master_vs2015_MultiCamera_s/result/input/";
	std::string org_data_store_path = "D:/Project/HandReconstruction/hmodel-master_vs2015_MultiCamera_s_TechnicalContribution_NetSeg_live3/data/";
	std::string hand_pose_store_path = "D:/Project/HandReconstruction/hmodel-master_vs2015_MultiCamera_s_TechnicalContribution_NetSeg_live3/result/";
	std::string render_store_path = "./";
	std::string Keypoint_2D_GT_path = "./";
	std::vector<std::string> joint_names_output_brief{
		"index_top", "index_middle", "index_bottom", "index_base",
		"middle_top", "middle_middle", "middle_bottom", "middle_base",
		"pinky_top", "pinky_middle", "pinky_bottom", "pinky_base",
		"ring_top", "ring_middle", "ring_bottom", "ring_base",
		"thumb_top", "thumb_middle", "thumb_bottom", "thumb_base",
	};
	std::vector<std::string> joint_names_output_full{
		"index_top", "index_middle", "index_bottom", "index_base",
		"middle_top", "middle_middle", "middle_bottom", "middle_base",
		"pinky_top", "pinky_middle", "pinky_bottom", "pinky_base",
		"ring_top", "ring_middle", "ring_bottom", "ring_base",
		"thumb_top", "thumb_middle", "thumb_bottom", "thumb_base",
		"index_membrane", "middle_membrane", "pinky_membrane", "thumb_additional", "thumb_membrane_left","ring_membrane", "thumb_membrane_middle",
		"palm_back", "palm_index", "palm_left", "palm_middle", "palm_pinky", "palm_right", "palm_ring", "palm_thumb",
		"wrist_bottom_left", "wrist_bottom_right", "wrist_top_left", "wrist_top_right"
	};

	cv::Mat real_color_map;
	std::vector<cv::Mat> depth_image_array;
	std::vector<cv::Mat> color_bgr_image_array;
	std::vector<cv::Mat> color_image_array;
	HandSegmentation hand_segmentation;
	DataParser left_data_parser, right_data_parser;
	MappedResource camera_depth_map_res;
	ObjectReconstruction ObjectRecon;
	VariantSmooth HandObjDis;
	RigidRegistration rigid_solver;
	Interaction interaction_datas;
	PosePredictNet pose_prediction_net;
	PhysHandSolver phys_hand_solver;

	/************************************************************************/
	/* Camera parameters                                                    */
	/************************************************************************/
	Eigen::Matrix4f left_camera_RT, right_camera_RT;			// camera extrinsics (left depth camera = I)
	Eigen::Matrix4f pose_camera1 = Eigen::Matrix4f::Identity();	// transformation matrix from left depth camera coord to right depth camera coord
	Eigen::Matrix4f depth2color_cl, depth2color_cr;				// transformation matrix from depth camera coord to color camera coord
	camera_intr depth_camera_cl, color_camera_cl;				// left camera intrinsics
	camera_intr depth_camera_cr, color_camera_cr;				// right camera intrinsics

	Eigen::Matrix4f object_pose;

	std::vector<cv::Mat> left_depth_org;
	std::vector<cv::Mat> left_color_org;
	std::vector<cv::Mat> right_depth_org;
	std::vector<cv::Mat> right_color_org;

	/************************************************************************/
	/* Hand data                                                            */
	/************************************************************************/
	std::vector<float> joint_radius;
	std::vector<float3> joint_position;

	std::vector<float> current_hand_pose;
	std::vector<float> predicted_hand_pose;

	std::vector<std::vector<float>> tip_2D_key_points;
	std::vector<Eigen::Vector3f> tip_3D_keypoint_pos;	// Target tip positions from physical refinement

	/************************************************************************/
	/* Performance Evaluation                                               */
	/************************************************************************/
	LARGE_INTEGER time_stmp;
	double count_freq, count_interv;
	double time_inter, time_inter_tracking, time_inter_tracking_hand, time_inter_tracking_obj, time_inter_fusion, time_inter_mainpipe, time_inter_total;
	double time_process, time_seg;
	std::vector<float> time_sequence;
	std::vector<float> mean_time_sequence;
	std::vector<int> frame_id_sequence;

	double mean_time_total = 0;
	double mean_time_mainpipeline=0;
	double mean_main_thread = 0;
	double mean_time_tracking = 0;
	double mean_time_fusion = 0;
	int sum_number=0;

	double mean_time_sum_seg = 0;
	int sum_number_seg = 0;

	float current_fps = 0;
	int speedup = 1;
	int first_frame_lag = 0;

	/************************************************************************/
	/* Settings                                                             */
	/************************************************************************/
	bool real_color;
	bool tracking_failed = true;
	bool initialization_enabled = true;
	bool tracking_enabled = true;
	bool verbose = false;	// output detailed time performance data
	bool run_surface2surface = true;
	
	bool track_hand_with_phys = false;
	bool track_with_two_cameras = false;
	bool record_right_camera_data = false;

	bool store_org_data = false;
	bool nonrigid_tracking = false;
	bool is_nonrigid = false;
	int frame_offset = 0;
	int start_frame = 10;
	bool set_recon_frame = false;
	bool is_set_recon_frame = false;
	bool is_set_nonrigid_frame = false;
	int reconstruction_frame = 66;//150
	int current_frame = 0;
	bool fuse_firstfra = 0;
	bool store_time = 0;
	bool show_ADT = false;
	bool show_input = false;
	bool store_input = false;
	bool store_solution_hand = false;
	int stop_recon_frame = 1000;
	bool show_mediate_result = false;
	bool show_seg = false;
	bool store_seg = false;
	int camera_use = 0;//0-both camera 1-left camera 2-right camera
	bool is_using_2D_keypoint = false;
	int nonrigid_start_frame = 10000;
	bool output_contact_info = false;

	/************************************************************************/
	/* 2D keypoint position for evaluation                                  */
	/************************************************************************/
	bool use_tip_2D_key_point = false;
	std::string tip_2D_key_point_path = "";
	const int tip_2D_key_point_num = 10;
	float total_tip_keypoint_dist = 0;
	float total_tip_points = 0;
	std::map<std::string, int> tips_name_to_idx;
	Eigen::Matrix4f dep_to_col;
	camera_intr col_proj;
	
	/************************************************************************/
	/* Smooth                                                               */
	/************************************************************************/
	float obj_smooth_ratio = 0;
	float hand_smooth_ratio = 0;
	float hand_smooth_ratio2 = 0;
	float force_smooth_ratio = 0.7;
	std::vector<float> old_hand_theta;
	Eigen::Matrix4f old_obj_motion;

public:
//	Tracker() {};
	Tracker(Worker*worker, RealSenseSR300& sr300, double FPS, std::string data_path, Json::Value& m_jsonRoot, bool real_color, bool is_benchmark);
	~Tracker();

	void toggle_tracking(bool on);
	void toggle_benchmark(bool on);
private:
	void timerEvent(QTimerEvent*);

public:
	void get_online_data_from_sensors();
	void get_predicted_pose();
	void data_preprocessing();
	void data_segmentation_init();
	void data_segmentation();
	void result_assignment_pre2seg();
	void result_assignment_seg2main();
	void result_assignment_pre2seg_buffer();
	void result_assignment_seg2main_buffer();
	void hand_tracking(bool with_tips_pos);
	void object_tracking_with_friction();
	void object_tracking_without_friction();
	void find_interaction_correspondence();
	void object_reconstruction();
	void printf_hello();
	void show_store_seg(int store_frame_id);
	void showADT();
	void show_store_input(int store_frame_id);
	void check_keypoints_energy(int store_frame_id);
	void show_visible_keypoints(int store_frame_id);
	void load_multi_view_frames2(size_t current_frame);
	void display_color_and_depth_input();
	void perform_smooth();
	void perform_force_smooth();
	void track_hand_with_physics();
	void outputContactInfo(std::string file_prefix);
	void outputHandObjMotion(std::string filename);
	void eval_tips_err(int frame_id);
	void process_track();

};
