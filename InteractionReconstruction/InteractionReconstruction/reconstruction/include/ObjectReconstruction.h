#pragma once
#include "pcl/gpu/containers/device_array.h"
#include <vector_types.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "../tracker/HandFinder/connectedComponents.h"
//#include <math.h>
//#include <iomanip>
#include "gpu/vector_operations.hpp"
#include <numeric>

#include "Sparse/ITMScene.h"
#include "Sparse/ITMSceneReconstructionEngine_CUDA.h"
#include "Sparse/ITMVoxelBlockHash.h"
#include "Sparse/ITMRenderState_VH.h"
#include "Sparse/ITMMeshingEngine_CUDA.h"
#include "warp_field.h"
#include "solver/shading_based_registration.h"
#include "../tracker/Interaction.h"

class ReconstructDataInput
{
public:
	int input_width;
	int input_height;

//	std::vector<unsigned char> Hand_Silhouette_host;
	std::vector<unsigned char> Hand_Object_Zone_host;

//	pcl::device::DeviceArray<unsigned char> Hand_Silhouette;
	pcl::device::DeviceArray<unsigned char> Hand_Object_Zone;

	pcl::device::DeviceArray<unsigned char> depth_hand;

	void update_hand_object_zone(std::vector<unsigned char> &hand_object_zone);
	void update_hand_object_zone_mat(cv::Mat &hand_object_zone_mat);

	ReconstructDataInput(int width, int height);
	~ReconstructDataInput() {}
};
 
class HandSegmentation
{
public: 

	int input_width;
	int input_height;

	pcl::device::DeviceArray<unsigned short> Depth_mm;
	std::vector<unsigned char> Hand_Object_Zone_host;
	pcl::device::DeviceArray<unsigned char> Hand_Object_Zone;
	pcl::device::DeviceArray<unsigned char> depth_hand;

	void update_hand_object_zone_mat(const cv::Mat &hand_object_zone_mat);

	void update_hand_object_mat(const cv::Mat &hand_object_zone_mat, const cv::Mat &hand_object_depth_mm_mat);

	float fx;
	float fy;
	float cx;
	float cy;

	mat34 camera_RT;

	std::vector<float3> Hand_KeyPoints_host;
	std::vector<float> Hand_KeyPoints_Radius_host;
	std::vector<int3> Hand_Block_host;

	pcl::device::DeviceArray<float3> Hand_KeyPoints;
	pcl::device::DeviceArray<float> Hand_KeyPoints_Radius;
	pcl::device::DeviceArray<int3> Hand_Block;

	void init_HandBlock(std::vector<int3> hand_block);
	void init_HandKeyPointsRadius(std::vector<float> hand_key_points_radius);

	void set_CameraParameter(float camera_fx, float camera_fy, int camera_cx, int camera_cy, mat34 camera_pose);
	void set_KeyPoints(std::vector<float3>& hand_keypoints, std::vector<float>& hand_radiis);
	void update_HandKeyPoints(std::vector<float3> hand_key_points);
	void update_HandKeyPointsRadius(std::vector<float> hand_key_points_radius);

	void hand_extraction(pcl::device::DeviceArray<unsigned char> &Hand_Object_Zone, pcl::device::DeviceArray<unsigned char> &depth_hand, int width, int height);

	void hand_extraction2(const pcl::device::DeviceArray<unsigned short> &Depth_mm, const pcl::device::DeviceArray<unsigned char> &Hand_Object_Zone, pcl::device::DeviceArray<unsigned char> &depth_hand, int width, int height);

	void object_extraction(const cv::Mat& hand_object_mask, const cv::Mat& hand_mask, cv::Mat& object_mask);

	void data_segmentation(const cv::Mat& hand_object_zone_mat, cv::Mat& hand_zone_mat, cv::Mat& object_zone_mask);

	void data_segmentation2(const cv::Mat& hand_object_zone_mat, const cv::Mat& depth_mm, cv::Mat& hand_zone_mat, cv::Mat& object_zone_mask);

	void data_segmentation3(const cv::Mat& hand_object_zone_mat, const cv::Mat& depth_mm, cv::Mat& hand_zone_mat, cv::Mat& object_zone_mask);

	HandSegmentation(int width, int height) 
	{
		input_width = width;
		input_height = height;

		depth_hand.create(input_width*input_height);
	};
	~HandSegmentation() {}
};

class ObjectReconstruction
{
public:
	int width;
	int height;

	enum {
		MAX_NUM_OF_VERTICES = 6000000,
		MAX_NUM_OF_VOXEL_CANDIDATES = 2000000,
		MAX_DATA_PAIRS = 200000,
		MAX_VALID_ALBEDO_PIXELS = 200000,
		MAX_NUM_OF_NODES = 4096
	};

public:

	int camera_use = 0;//0-all camera  1-left camera  2-right camera

	pcl::device::Intr depth_intr_camera0, depth_intr_camera1;
	Eigen::Matrix4f extrin_camera0, extrin_camera1;
	cv::Mat depth_image_camera0, depth_image_camera1;

	pcl::gpu::DeviceArray2D<unsigned short> first_depth_frame;

	pcl::gpu::DeviceArray<float4> m_can_vertices_buffer;
	pcl::gpu::DeviceArray<float4> m_can_normals_buffer;
	pcl::gpu::DeviceArray<float4> m_warp_vertices_buffer;
	pcl::gpu::DeviceArray<float4> m_warp_normals_buffer;
	pcl::gpu::DeviceArray<float4> m_live_vertices_buffer;
	pcl::gpu::DeviceArray<float4> m_live_normals_buffer;

	pcl::gpu::DeviceArray<float4> m_valid_can_vertices;
	pcl::gpu::DeviceArray<float4> m_valid_can_normals;
	pcl::gpu::DeviceArray<float4> m_valid_warp_vertices;
	pcl::gpu::DeviceArray<float4> m_valid_warp_normals;
	pcl::gpu::DeviceArray<float4> m_valid_live_vertices;
	pcl::gpu::DeviceArray<float4> m_valid_live_normals;

	/***************************************************/
	/*              nonrigid solver                    */
	// cublas handle for pcg solver
	cublasHandle_t m_cublas_handle;
	std::string m_results_dir;

	// preallocate buffers used in solver//
	// used in associate_depth_data_pairs_dev
	pcl::gpu::DeviceArray<int> m_depth_pairs_occupied_array;
	pcl::gpu::DeviceArray<int4> m_depth_pairs_pair_array;
	pcl::gpu::DeviceArray<int> m_depth_pairs_scan_array;
	pcl::gpu::DeviceArray<int> m_depth_pairs_scan_storage;
	pcl::gpu::DeviceArray<int4> m_depth_pairs_compact_array;

	pcl::gpu::DeviceArray<float4> m_depth_pairs_compact_array_cv_f4;//ZH use these four arrays to store correspondences
	pcl::gpu::DeviceArray<float4> m_depth_pairs_compact_array_cn_f4;
	pcl::gpu::DeviceArray<float4> m_depth_pairs_compact_array_dv_f4;
	pcl::gpu::DeviceArray<float4> m_depth_pairs_compact_array_dn_f4;

	// used in query_depth_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> m_depth_pairs_knn_array;
	pcl::gpu::DeviceArray<float4> m_depth_pairs_weight_array;

	// used in interaction_depth_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> m_interaction_knn_array;
	pcl::gpu::DeviceArray<float4> m_interaction_weight_array;

	// used in query_albedo_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> m_albedo_pixel_knn_array;
	pcl::gpu::DeviceArray<float4> m_albedo_pixel_weight_array;

	// used in evaluate_energy_device
	pcl::gpu::DeviceArray<float> m_residual_array;
	pcl::gpu::DeviceArray<float> m_evaluate_energy_reduce_storage;
	pcl::gpu::DeviceArray<float> m_total_energy;

	// used in calculate_Ii_and_Iij
	pcl::gpu::DeviceArray<int> m_Ii_key;
	pcl::gpu::DeviceArray<int> m_Ii_value;

	pcl::gpu::DeviceArray<int> m_Ii_sorted_key;
	pcl::gpu::DeviceArray<int> m_Ii_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> m_Ii_radixsort_storage;
	pcl::gpu::DeviceArray<int> m_Ii_offset;

	pcl::gpu::DeviceArray<int> m_Iij_key;
	pcl::gpu::DeviceArray<int> m_Iij_value;

	pcl::gpu::DeviceArray<int> m_Iij_sorted_key;
	pcl::gpu::DeviceArray<int> m_Iij_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> m_Iij_radixsort_storage;

	pcl::gpu::DeviceArray<int> m_Iij_segment_label;
	pcl::gpu::DeviceArray<int> m_Iij_scan;
	pcl::gpu::DeviceArray<unsigned char> m_Iij_scan_storage;

	pcl::gpu::DeviceArray<int> m_compact_Iij_key;
	pcl::gpu::DeviceArray<int> m_compact_Iij_offset;

	pcl::gpu::DeviceArray<int> m_row_offset;
	pcl::gpu::DeviceArray<int> m_row_length;
	pcl::gpu::DeviceArray<int> m_bin_length;

	// used in construct_ata_atb
	pcl::gpu::DeviceArray<float> m_depth_term_values;
	pcl::gpu::DeviceArray<float> m_smooth_term_values;
	pcl::gpu::DeviceArray<float> m_interaction_term_values;
	pcl::gpu::DeviceArray<float> m_shading_term_values;
	pcl::gpu::DeviceArray<float> m_b_values;
	pcl::gpu::DeviceArray<float> m_Bii_values;
	pcl::gpu::DeviceArray<float> m_Bij_values;
	pcl::gpu::DeviceArray<int> m_nonzero_rowscan;
	pcl::gpu::DeviceArray<float> m_ATA_data;
	pcl::gpu::DeviceArray<int> m_ATA_colidx;
	pcl::gpu::DeviceArray<int> m_ATA_rowptr;
	pcl::gpu::DeviceArray<float> m_ATb_data;

	// used in pcl_solver
	pcl::gpu::DeviceArray<float> m_x_pcg;
	pcl::gpu::DeviceArray<float> m_M_inv;
	pcl::gpu::DeviceArray<float> m_p;
	pcl::gpu::DeviceArray<float> m_q;
	pcl::gpu::DeviceArray<float> m_r;
	pcl::gpu::DeviceArray<float> m_s;
	pcl::gpu::DeviceArray<float> m_t;

	// streams for associate_depth_data_pairs and find_valid_albedo_pixels
	cudaStream_t m_depth_data_pairs_stream;
//	cudaStream_t m_valid_albedo_pixels_stream;

	// page-locked memory for the numbers of depth_data_pairs and valid_albedo_pairs
	int *m_depth_data_pairs_num;
//	int *m_valid_albedo_pixels_num;

	// pre-allocated for Huber weight on smooth terms
	pcl::gpu::DeviceArray<float> m_huber_buffer;

	/********************************************************************/
	//related with the sparse voxel reconstruction
	ITMLib::Objects::ITMSceneParams *sceneParams;
	ITMLib::Objects::ITMScene *scene;//ITMLib::Objects::ITMScene<ITMVoxel, ITMVoxelIndex> *scene;
	ITMSceneReconstructionEngine_CUDA *SceneReconstrution;
	ITMRenderState_VH *renderState_live;
	ITMLib::Engine::ITMMeshingEngine_CUDA *meshEngine;

	WarpField m_warp_field;

	void initialization(int image_width, int image_height, const pcl::device::Intr intr_camera0, const pcl::device::Intr intr_camera1, const Eigen::Matrix4f pose_camera0, const Eigen::Matrix4f pose_camera1);

	void assign_SourceImage(cv::Mat& depth_cam0, cv::Mat& depth_cam1);

	void fuse_FirstFrame(cv::Mat& depth_cam0, cv::Mat& depth_cam1);

	void rigid_fusion(cudaTextureObject_t Depth_tex, Eigen::Matrix4f object_motion);

	void rigid_fusion_C2(cudaTextureObject_t Depth_tex0, cudaTextureObject_t Depth_tex1, Eigen::Matrix4f object_motion);

	void nonrigid_fusion(cudaTextureObject_t Depth_tex, Eigen::Matrix4f object_motion);

	void nonrigid_fusion_C2(cudaTextureObject_t Depth_tex0, /*cudaTextureObject_t Depth_tex1,*/ Eigen::Matrix4f object_motion);

	void extract_SceneModel(mat34 object_pose, int weight_thr);

	void extract_NonRigid_SceneModel(int weight_thr);
	void warp_NonRigid_SceneModel();

	void find_interaction_finger_joint_correspondence(const mat34 object_motion,Interaction& interaction_data);

	void find_interaction_hand_correspondence(const mat34 object_motion, Interaction& interaction_data);

	void find_interaction_hand_correspondence_surface_contact(const mat34 object_motion, Interaction& interaction_data);

	void construct_WarpField();

	void nonrigid_MotionEstimation(int frame_idx, cudaTextureObject_t depth_vmap, cudaTextureObject_t depth_nmap, cudaTextureObject_t can_vmap, cudaTextureObject_t can_nmap, 
		Eigen::Matrix4f& object_rigid_motion);

	void nonrigid_MotionEstimation2(int frame_idx, cudaTextureObject_t depth_vmap_c0, cudaTextureObject_t depth_nmap_c0, cudaTextureObject_t can_vmap_c0, cudaTextureObject_t can_nmap_c0,
		cudaTextureObject_t depth_vmap_c1, cudaTextureObject_t depth_nmap_c1, cudaTextureObject_t can_vmap_c1, cudaTextureObject_t can_nmap_c1,	Eigen::Matrix4f& object_rigid_motion);

	void nonrigid_MotionEstimation3(int frame_idx, cudaTextureObject_t depth_vmap_c0, cudaTextureObject_t depth_nmap_c0, cudaTextureObject_t can_vmap_c0, cudaTextureObject_t can_nmap_c0,
		cudaTextureObject_t depth_vmap_c1, cudaTextureObject_t depth_nmap_c1, cudaTextureObject_t can_vmap_c1, cudaTextureObject_t can_nmap_c1, Eigen::Matrix4f& object_rigid_motion, 
		pcl::gpu::DeviceArray<float>& node_smooth_coef);

	void nonrigid_MotionEstimation4(int frame_idx, cudaTextureObject_t depth_vmap_c0, cudaTextureObject_t depth_nmap_c0, cudaTextureObject_t can_vmap_c0, cudaTextureObject_t can_nmap_c0,
		cudaTextureObject_t depth_vmap_c1, cudaTextureObject_t depth_nmap_c1, cudaTextureObject_t can_vmap_c1, cudaTextureObject_t can_nmap_c1, Eigen::Matrix4f& object_rigid_motion,
		pcl::gpu::DeviceArray<float>& node_smooth_coef, Interaction& interaction_data);

	void nonrigid_MotionEstimation5(int frame_idx, cudaTextureObject_t depth_vmap_c0, cudaTextureObject_t depth_nmap_c0, cudaTextureObject_t can_vmap_c0, cudaTextureObject_t can_nmap_c0,
		Eigen::Matrix4f& object_rigid_motion, pcl::gpu::DeviceArray<float>& node_smooth_coef, Interaction& interaction_data);

	void nonrigid_MotionEstimation_handblock(int frame_idx, cudaTextureObject_t depth_vmap_c0, cudaTextureObject_t depth_nmap_c0, cudaTextureObject_t can_vmap_c0, cudaTextureObject_t can_nmap_c0,
									Eigen::Matrix4f& object_rigid_motion, pcl::gpu::DeviceArray<float>& node_smooth_coef, Interaction& interaction_data);

	std::vector<float4> get_node_coordinate_host();

	std::vector<DualQuaternion> get_node_motion_host();

	ObjectReconstruction() {};
	~ObjectReconstruction() {};

public:

	ORUMatrix4f eigen2ORU2(Eigen::Matrix4f camera_pose_eigen)
	{

		Eigen::Matrix4f camera_pose_eigen_tran = camera_pose_eigen.transpose();

		ORUMatrix4f camera_pose_ORU;

		camera_pose_ORU.m00 = camera_pose_eigen_tran(0, 0);
		camera_pose_ORU.m01 = camera_pose_eigen_tran(0, 1);
		camera_pose_ORU.m02 = camera_pose_eigen_tran(0, 2);
		camera_pose_ORU.m03 = camera_pose_eigen_tran(0, 3);

		camera_pose_ORU.m10 = camera_pose_eigen_tran(1, 0);
		camera_pose_ORU.m11 = camera_pose_eigen_tran(1, 1);
		camera_pose_ORU.m12 = camera_pose_eigen_tran(1, 2);
		camera_pose_ORU.m13 = camera_pose_eigen_tran(1, 3);

		camera_pose_ORU.m20 = camera_pose_eigen_tran(2, 0);
		camera_pose_ORU.m21 = camera_pose_eigen_tran(2, 1);
		camera_pose_ORU.m22 = camera_pose_eigen_tran(2, 2);
		camera_pose_ORU.m23 = camera_pose_eigen_tran(2, 3);

		camera_pose_ORU.m30 = camera_pose_eigen_tran(3, 0);
		camera_pose_ORU.m31 = camera_pose_eigen_tran(3, 1);
		camera_pose_ORU.m32 = camera_pose_eigen_tran(3, 2);
		camera_pose_ORU.m33 = camera_pose_eigen_tran(3, 3);

		return camera_pose_ORU;
	}

	mat34 Eigen2mat2(Eigen::Matrix4f mat_eigen)
	{
		mat34 mat;
		mat.rot.m00() = mat_eigen(0, 0); mat.rot.m01() = mat_eigen(0, 1); mat.rot.m02() = mat_eigen(0, 2);
		mat.rot.m10() = mat_eigen(1, 0); mat.rot.m11() = mat_eigen(1, 1); mat.rot.m12() = mat_eigen(1, 2);
		mat.rot.m20() = mat_eigen(2, 0); mat.rot.m21() = mat_eigen(2, 1); mat.rot.m22() = mat_eigen(2, 2);
		mat.trans.x = mat_eigen(0, 3); mat.trans.y = mat_eigen(1, 3); mat.trans.z = mat_eigen(2, 3);

		return mat;
	}
};