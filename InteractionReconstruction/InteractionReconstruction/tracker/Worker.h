#pragma once
#include "util/gl_wrapper.h"
#include "tracker/ForwardDeclarations.h"
#include "tracker/Types.h"
#include "tracker/HandFinder/HandFinder.h"
#include "OpenGL/OffscreenRenderer.h"
#include "Data/DataFrame.h"
#include "Energy/JointLimits.h"
#include "Energy/Damping.h"
#include "Energy/Collision.h"
#include "Energy/PoseSpace.h"
#include "Energy/Fitting.h"
#include "Energy/Fitting/TrackingMonitor.h"
#include "Energy/Temporal.h"
#include "Energy/InteractionHandTracking.h"
#include "Energy/PosePredictionEnergy.h"
#include "Energy/KeypointEnergy.h"

#include "GLWidget.h"

#include "opencv2/core/core.hpp"       ///< cv::Mat
#include "opencv2/highgui/highgui.hpp" ///< cv::imShow

#include "reconstruction/include/safe_call.hpp"
#include <pcl/gpu/containers/device_array.h>
#include "reconstruction/include/nvprofiler.h"
#include "Filter.h"

/// @note do not construct more than one instance of this class
class Worker {

public:
	struct Settings {
		int termination_max_iters = 6;
		int termination_max_rigid_iters = 1;
		int second_stage_iter_times = 5;
	} _settings;
	Settings*const settings = &_settings;

public:
	QGLWidget* glarea = NULL;
	GLWidget* gl_resrc_ptr = NULL;
public:
	void bind_glwidget(QGLWidget* glarea) { this->glarea = glarea; }
	void bind_gl_resrc(GLWidget* gl_resrc_ptr) {this->gl_resrc_ptr = gl_resrc_ptr;	}
	void initialRendererCamMatrix(int _width, int _height, CamerasParameters camera_par);
	void initialWorkerCamMatrix(int _width, int _height, CamerasParameters camera_par);
	void updateGL();
	void render_predicted_hand();
	void render_models();
	void render_texture_rigid();
	void render_texture_nonrigid();
	void set_camera_object_motion(Eigen::Matrix4f object_motion);

	std::tuple<cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t> map_rigid_icp_texobj();
	void unmap_rigid_icp_texobj();
	std::tuple<cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t> map_nonrigid_icp_texobj();
	void unmap_nonrigid_icp_texobj();

	std::tuple<	pcl::gpu::DeviceArray<float4>,
		pcl::gpu::DeviceArray<float4>,
		pcl::gpu::DeviceArray<float4>,
		pcl::gpu::DeviceArray<float4>> map_vertex_attributes_to_CUDA();

	void map_vertex_attributes();
	void unmap_vertex_attributes();

public:
	bool is_watch=false;
	bool is_reconstruction = false;
	bool is_only_track = false;
	bool test;
	bool benchmark;
	bool save_rastorized_model;
	int user_name;
	int camera_use = 0;//0-all camera  1-left camera  2-right camera
	std::string data_path;
	int frame_id;
	int reconstruction_frame = 10000;
	int start_frame_with_only_2D_keypoint = 10000;
	bool run_surface2surface = true;

	camera_intr para_camera0, para_camera1;
	Eigen::Matrix4f pose_camera0, pose_camera1;
	Eigen::Matrix4f view_camera0, view_camera1;
	Eigen::Matrix4f rigid_motion;

	Camera* camera = NULL;
	Model * model;
	DataFrame current_frame = DataFrame(-1);
	TrackingError tracking_error;
	//std::vector<TrackingError> tracking_error_optimization;

	DepthTexture16UC1* sensor_depth_texture = NULL;
	ColorTexture8UC3* sensor_color_texture = NULL;
	DepthTexture16UC1Recon* camera_depth_texture = NULL;

	energy::Fitting E_fitting;
	energy::InteractionHandTracking E_interaction;
	energy::PosePrediction E_poseprediction;
	energy::Temporal E_temporal;
	energy::Damping E_damping;
	energy::JointLimits E_limits;
	energy::JointLimits E_limits_second;
	energy::Collision E_collision;
	energy::Collision E_collision_second;
	energy::PoseSpace E_pose;
	energy::PoseSpace E_pose_second;
	energy::Keypoint2D E_KeyPoint2D;
	energy::Keypoint3D E_KeyPoint3D;
	energy::KeypointTips E_KeyPointTips;
	energy::PhysKinDiff E_PhysKinDiff;

	OneEuroFilter OEFilter;

	HandFinder* handfinder = NULL;
	TrivialDetector* trivial_detector = NULL;
	OffscreenRenderer offscreen_renderer;
	OffscreenRenderer rastorizer;
	TrackingMonitor monitor;

	int vertex_number = 0;

	cv::Mat segmentation;
	cv::Mat keypt_pred_img;

	std::vector<float4> object_can_points;
	std::vector<float4> object_can_normals;
	std::vector<float4> object_live_points;
	std::vector<float4> object_live_normals;

	std::vector<float3> object_nodes;
	std::vector<std::vector<int>> node_tip_idx;
	std::vector<int> node_close2tip_idx;
	std::vector<float> variant_smooth;

	std::vector<float3> interaction_corrs_vertex;
	std::vector<float3> interaction_corrs_normal;
	std::vector<unsigned char> interaction_corrs_finger_idx;
	std::vector<float3> contact_spheremesh;
	std::vector<float3> fingertips;

	std::vector<float> predicted_pose;

	bool is_using_2D_keypoint = false;
	std::string Keypoint_2D_GT_path;
	std::vector<std::vector<float2>> Keypoint_2D_GT_vec_left;
	std::vector<std::vector<float3>> Keypoint_3D_GT_vec_left;
	std::vector<std::vector<float3>> Keypoint_3D_combine_vec_left;
	std::vector<std::vector<float>> Keypoint_3D_visible_vec_left;
	std::vector<float3> Keypoint_3D_pred;

	std::vector<int> Keypoint_block;
	std::vector<int> Keypoint2SphereCenter;
	std::vector<int> using_keypoint_2D;
	std::vector<int> using_keypoint_tips;		// Add by hhy

	//cuda variances to get gl resources 
	cudaResourceDesc m_res_desc;
	cudaTextureDesc m_tex_desc;

	// mapped textures
	cudaArray_t m_live_vmap_array_dv_c0;
	cudaArray_t m_live_nmap_array_dv_c0;
	cudaArray_t m_live_vmap_array_dv_c1;
	cudaArray_t m_live_nmap_array_dv_c1;

	cudaTextureObject_t m_live_vmap_texobj_dv_c0;
	cudaTextureObject_t m_live_nmap_texobj_dv_c0;
	cudaTextureObject_t m_live_vmap_texobj_dv_c1;
	cudaTextureObject_t m_live_nmap_texobj_dv_c1;

	cudaArray_t m_can_vmap_array_dv_c0;
	cudaArray_t m_can_nmap_array_dv_c0;
	cudaArray_t m_can_vmap_array_dv_c1;
	cudaArray_t m_can_nmap_array_dv_c1;

	cudaTextureObject_t m_can_vmap_texobj_dv_c0;
	cudaTextureObject_t m_can_nmap_texobj_dv_c0;
	cudaTextureObject_t m_can_vmap_texobj_dv_c1;
	cudaTextureObject_t m_can_nmap_texobj_dv_c1;

	cudaGraphicsResource_t m_tex_res[4];
	cudaGraphicsResource_t m_tex_res_nonrigid[4];

	//input buffer resources for cuda-opengl inter-operations 
	cudaGraphicsResource_t m_buffer_res[4];
	
	//float3* object_model_v_ptr;
	//float3* object_model_n_ptr;
	//int point_size;

	// Add by hhy
	std::vector<float3> tar_tip_point;		// target tips pos given by physics part
	std::vector<double> tips_rel_conf;		// relative confidence of tips
	std::vector<int> tips_org_conf;			// original confidence of tips
	std::vector<Eigen::Vector3f> contact_points;	// classified contact points given by phys solver
	std::vector<Eigen::Vector3f> contact_forces;	// contact forces solved by phys solver
	std::vector<int> contact_corr;
	std::vector<Eigen::Vector3f> force_history;
	Eigen::Vector3f target_force;
	Eigen::Vector3f target_moment;
	Eigen::Vector3f object_center;
	Eigen::Vector3f object_vel;
	Eigen::Vector3f object_ang_vel;
	Eigen::Matrix3f object_rot;

public:
	Worker(Camera *camera, bool test, bool benchmark, bool save_rasotrized_model, int user_name, string data_path);
	~Worker();
	void init_graphic_resources(); ///< not in constructor as needs valid OpenGL context
	void cleanup_graphic_resources();
	void init_graphic_map_resources();
	void load_2D_keypoint_GT(std::string file_path);
	void load_3D_keypoint_GT(std::string file_path);
	void load_3D_keypoint_combine(std::string file_path);
	void load_3D_keypoint_visible(std::string file_path);

	std::vector<float3> get_hand_keypoints();
	std::vector<float3> get_hand_skeleton();
	std::vector<float3> get_pred_skeleton();
	std::vector<float3> get_hand_keypoints_color();

public:
	// Single Camera Version
	void track(int iter/*, int frame_id*/, 
			   cv::Mat& silhouette_c0, /*cv::Mat& silhouette_c1,*/ 
			   std::vector<int>& real_ADT_c0, /*std::vector<int>& real_ADT_c1,*/ 
			   Interaction& interaction_data, 
			   std::vector<float3> joints_pred, 
			   std::vector<float2> joints_pred_2D,
			   bool time_test);

	// Double Cameras Version
	void track(int iter,
		cv::Mat& silhouette_c0, cv::Mat& silhouette_c1,
		std::vector<int>& real_ADT_c0, std::vector<int>& real_ADT_c1,
		Interaction& interaction_data,
		std::vector<float3> joints_pred,
		std::vector<float2> joints_pred_2D,
		bool time_test);

	// Single Camera Version
	bool track_till_convergence(/*int frame_id, std::ofstream &file,*/
								cv::Mat& silhouette_c0,/* cv::Mat& silhouette_c1, */
								std::vector<int>& real_ADT_c0, /*std::vector<int>& real_ADT_c1,*/ 
								Interaction& interaction_data, 
								Eigen::Matrix4f rigid_mot, 
								std::vector<float3> joints_pred, 
								std::vector<float2> joints_pred_2D,
								bool time_test);

	// Double Cameras Version
	bool track_till_convergence(cv::Mat& silhouette_c0, cv::Mat& silhouette_c1, 
								std::vector<int>& real_ADT_c0, std::vector<int>& real_ADT_c1,
								Interaction& interaction_data,
								Eigen::Matrix4f rigid_mot,
								std::vector<float3> joints_pred,
								std::vector<float2> joints_pred_2D,
								bool time_test);

	void update_confidence();

	void track_second_stage(const std::vector<Eigen::Vector3f>& tips_joint_pos, Interaction& interaction_data);

	void track_second_stage_debug(const std::vector<Eigen::Vector3f>& tips_joint_pos);
};
