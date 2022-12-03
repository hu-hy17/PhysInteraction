#ifndef _RIGID_ICP_H_
#define _RIGID_ICP_H_

#include <cuda_runtime.h>
#include <pcl/gpu/containers/device_array.h>
#include "reconstruction/include/gpu/vector_operations.hpp"
#include "tracker/CommonVariances.h"
//#include <opencv2/core.hpp>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" ///< cv::imShow
#include "tracker/Interaction.h"
#include <iostream>

class RigidRegistration
{
public:
	RigidRegistration();
	~RigidRegistration() {};

	void initialize(int image_width, int image_height, camera_intr came_intr0, camera_intr came_intr1, Eigen::Matrix4f _camera_pose0, Eigen::Matrix4f _camera_pose1, int camera_using);
	void initialize_for_friction(const std::vector<float>& joints_radius, const int max_interaction_points_number);

	//set the input variances
	void set_parameters(int image_width, int image_height, float fx, float fy, float cx, float cy,
		cudaTextureObject_t _depth_vmap_0, cudaTextureObject_t _depth_nmap_0, cudaTextureObject_t _live_vmap_0, cudaTextureObject_t _live_nmap_0, Eigen::Matrix4f object_rigid_motion);

	void set_parameters2(cudaTextureObject_t _depth_vmap_c0, cudaTextureObject_t _depth_nmap_c0, cudaTextureObject_t _live_vmap_c0, cudaTextureObject_t _live_nmap_c0, 
		/*cudaTextureObject_t _depth_vmap_c1, cudaTextureObject_t _depth_nmap_c1, cudaTextureObject_t _live_vmap_c1, cudaTextureObject_t _live_nmap_c1,*/ Eigen::Matrix4f object_rigid_motion, 
		std::vector<int>& real_left_ADT/*, std::vector<int>& real_right_ADT*/);

	// run rigid ICP procedure
	Eigen::Matrix4f run();
	Eigen::Matrix4f run2(int frame_idx, cv::Mat left_color_HO,/* cv::Mat right_color_HO,*/ Interaction& interaction_data, bool run_surface2surface/*, cudaStream_t m_object_stream*/);
	Eigen::Matrix4f run2_with_friction(int frame_idx, cv::Mat left_color_HO, Interaction& interaction_data, bool run_surface2surface);
	Eigen::Matrix4f run2_with_friction_huber(int frame_idx, cv::Mat left_color_HO, Interaction& interaction_data, bool run_surface2surface,
											 int iter_num = 5, bool use_force = true, bool use_friction = true);

	// validate rigid ICP on the host
	Eigen::Matrix4f validate(Eigen::Matrix4f camera_view);

	Eigen::Matrix4f validate2(int frame_idx);

	void rigid_icp_parallel_reduction_friction_cpu(const std::vector<unsigned char> _interaction_finger_idx,
		const std::vector<float4> _interaction_warp_vertice,
		const std::vector<float4> _interaction_warp_normal,
		const std::vector<float3> _finger_joints_pos,
		const std::vector<float> _finger_joints_radius,
		const std::vector<float3> _finger_move_vec,
		const mat34 &rigid_motion, const mat33 &_R, const float3 &_t, Eigen::Matrix<float, 6, 6> &AF, Eigen::Matrix<float, 6, 1> &bf, int iter);

	void rigid_icp_parallel_reduction_friction_cpu2(
		const std::vector<std::vector<float3>> _interaction_warp_vertice_each_joint,
		const std::vector<std::vector<float3>> _interaction_warp_normal_each_joint,
		const std::vector<bool> _joint_touch_status,
		const std::vector<float3> _finger_joints_pos,
		const std::vector<float> _finger_joints_radius,
		const mat34 &rigid_motion, const mat33 &_R, const float3 &_t, Eigen::Matrix<float, 6, 6> &AF, Eigen::Matrix<float, 6, 1> &bf, bool save_use);

	void rigid_icp_parallel_reduction_force_cpu(
		const std::vector<unsigned char> _interaction_finger_idx,
		const std::vector<float4> _interaction_warp_vertice,
		const std::vector<float4> _interaction_warp_normal,
		const std::vector<float4> _finger_joints_pos,
		const std::vector<float> _finger_joints_radius,
		const mat34 &rigid_motion, const mat33 &_R, const float3 &_t, Eigen::Matrix<float, 6, 6> &AF, Eigen::Matrix<float, 6, 1> &bf, int iter);

	void rigid_icp_parallel_reduction_force_block2surface_cpu(
		const std::vector<float4> _interaction_warp_vertice,
		const std::vector<float4> _interaction_warp_normal,
		const std::vector<float4> _hand_joints_pos,
		const std::vector<float> _hand_joints_radius,
		const std::vector<int3> _block_sphere_idx,
		const std::vector<float3> _sphere_coordinate,
		const std::vector<unsigned char> _block_idx,
		const mat34 &rigid_motion, const mat33 &_R, const float3 &_t, Eigen::Matrix<float, 6, 6> &AF, Eigen::Matrix<float, 6, 1> &bf, int iter);

	void rigid_icp_parallel_reduction_friction_block2surface_cpu(
		const std::vector<float4> _interaction_warp_vertice,
		const std::vector<float4> _interaction_warp_normal,
		const std::vector<float4> _hand_joints_pos,
		const std::vector<float> _hand_joints_radius,
		const std::vector<int3> _block_sphere_idx,
		const std::vector<float3> _sphere_coordinate,
		const std::vector<unsigned char> _block_idx,
		const mat34 &rigid_motion, const mat33 &_R, const float3 &_t, Eigen::Matrix<float, 6, 6> &AF, Eigen::Matrix<float, 6, 1> &bf, int iter);

	void rigid_icp_parallel_reduction_friction_block2surface_delta_cpu(
		const std::vector<float4> _interaction_warp_vertice,
		const std::vector<float4> _interaction_warp_normal,
		const std::vector<float4> _hand_joints_pos,
		const std::vector<float4> _hand_joints_pos_before,
		const std::vector<float> _hand_joints_radius,
		const std::vector<int3> _block_sphere_idx,
		const std::vector<float3> _sphere_coordinate,
		const std::vector<unsigned char> _block_idx,
		const mat34 &rigid_motion, const mat33 &_R, const float3 &_t, Eigen::Matrix<float, 6, 6> &AF, Eigen::Matrix<float, 6, 1> &bf, int iter);

	void rigid_icp_parallel_reduction_friction_surface2surface_cpu(
		const std::vector<float4> _interaction_warp_vertice,
		const std::vector<float4> _interaction_warp_normal,
		const std::vector<float4> _hand_joints_pos_before,
		const std::vector<float> _hand_joints_radius,
		const std::vector<int3> _block_sphere_idx,
		const std::vector<float3> _sphere_coordinate,
		const std::vector<unsigned char> _block_idx,
		const std::vector<unsigned char> _block2phalange,
		const std::vector<mat34> _phalange_local2global_before,
		const std::vector<mat34> _phalange_local2global,
		/*const std::vector<unsigned char> _phalange_centerId,*/
		const std::vector<float3> _local_vector_contact_spheremesh,
		const mat34 &rigid_motion, const mat33 &_R, const float3 &_t, Eigen::Matrix<float, 6, 6> &AF, Eigen::Matrix<float, 6, 1> &bf, int iter);
	
	void update_rigid_poses(const Eigen::Matrix4f &_depth_se3_update);

	mat34 Eigen2mat(Eigen::Matrix4f mat_eigen)
	{
		mat34 mat;
		mat.rot.m00() = mat_eigen(0, 0); mat.rot.m01() = mat_eigen(0, 1); mat.rot.m02() = mat_eigen(0, 2);
		mat.rot.m10() = mat_eigen(1, 0); mat.rot.m11() = mat_eigen(1, 1); mat.rot.m12() = mat_eigen(1, 2);
		mat.rot.m20() = mat_eigen(2, 0); mat.rot.m21() = mat_eigen(2, 1); mat.rot.m22() = mat_eigen(2, 2);
		mat.trans.x = mat_eigen(0, 3); mat.trans.y = mat_eigen(1, 3); mat.trans.z = mat_eigen(2, 3);

		return mat;
	}

	//variables for friction
	std::vector<float3> joints_use_pos_before, joints_use_pos_after, delta_joints_pos, joints_use_position;
	std::vector<int> joints_use_for_friction;
	std::vector<float> joints_use_radius;

	//use to supply with opengl
	std::vector<float3> used_interaction_vertex;
	std::vector<float3> used_interaction_normal;
	std::vector<unsigned char> used_interaction_corrs_joints_idx;
	std::vector<float3> used_contact_point_spheremesh;

public:
	float sil_coe = 0.2f;
	float force_coef = 0.01f;
	float friction_coe = 0.1f;
private:
	// update se3 after each ICP iteration
	void update_se3(mat33 &_rot, float3 &_trans);

	// depth vmaps and nmaps of 3 levels
	cudaTextureObject_t m_depth_vmap_c0;
	cudaTextureObject_t m_depth_vmap_c1;
	cudaTextureObject_t m_depth_vmap_c2;
	cudaTextureObject_t m_depth_nmap_c0;
	cudaTextureObject_t m_depth_nmap_c1;
	cudaTextureObject_t m_depth_nmap_c2;
	
	// rendered live vmaps and nmaps for projective ICP
	cudaTextureObject_t m_live_vmap_c0;
	cudaTextureObject_t m_live_vmap_c1;
	cudaTextureObject_t m_live_vmap_c2;
	cudaTextureObject_t m_live_nmap_c0;
	cudaTextureObject_t m_live_nmap_c1;
	cudaTextureObject_t m_live_nmap_c2;

	float fx_c0;
	float fy_c0;
	float cx_c0;
	float cy_c0;

	float fx_c1;
	float fy_c1;
	float cx_c1;
	float cy_c1;

	int width;
	int height;

	int camera_use = 0;//0-all camera  1-left camera 2-right camera

	Eigen::Matrix4f camera_pose0;
	Eigen::Matrix4f camera_pose1;

	Eigen::Matrix4f camera_view0;
	Eigen::Matrix4f camera_view1;

	Eigen::Matrix4f rigid_pose;

	//pre-allocated GPU memory for ADT
	pcl::gpu::DeviceArray<int> m_ADT_c0;
	pcl::gpu::DeviceArray<int> m_ADT_c1;

	// pre-allocated GPU memory for reduction
	pcl::gpu::DeviceArray<float> m_block_reduction_gmem;//global memory
	pcl::gpu::DeviceArray<float> m_reduced_values; // 27 values of both symmetric A and b

	//pre-allocated GPU memory for silhouette indicate
	pcl::gpu::DeviceArray<int2> using_ADT;

	//pre-allocated GPU memory for friction
	pcl::gpu::DeviceArray<float> m_joints_radius;
	pcl::gpu::DeviceArray<float3> m_joints_position;
	pcl::gpu::DeviceArray<float3> m_delta_joints_vector;
	pcl::gpu::DeviceArray<float> m_interaction_reduction_gmem;
};

#endif