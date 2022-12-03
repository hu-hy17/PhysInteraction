#pragma once
#include <vector_types.h>
#include <vector_functions.hpp>
#include <pcl/gpu/containers/device_array.h>
#include "reconstruction/include/gpu/vector_operations.hpp"
#include "glm/glm.hpp"
#include <iostream>

#define MAX_INTERACTION_CORRESPONDENCE 1000//1000

class Interaction
{
public:

	std::vector<unsigned char> phalange_center_id_host;
	pcl::device::DeviceArray<unsigned char> phalange_center_id_device;

	std::vector<unsigned char> block2phalange_host;
	pcl::device::DeviceArray<unsigned char> block2phalange_device;

	std::vector<mat34> global2local_phalange_host;
	pcl::device::DeviceArray<mat34> global2local_phalange_device;
	
	std::vector<mat34> local2global_phalange_before_host;
	pcl::device::DeviceArray<mat34> local2global_phalange_before_device;

	std::vector<mat34> local2global_phalange_host;
	pcl::device::DeviceArray<mat34> local2global_phalange_device;

	std::vector<float> joint_radius_host;
	pcl::device::DeviceArray<float> joint_radius_device;

	std::vector<float> hand_joints_radius_host;
	pcl::device::DeviceArray<float> hand_joints_radius_device;

	std::vector<int3> finger_blocks_host;
	pcl::device::DeviceArray<int3> finger_blocks_device;

	std::vector<float4> joint_position_host;
	pcl::device::DeviceArray<float4> joint_position_device;

	std::vector<float4> hand_joints_position_before_host;
	pcl::device::DeviceArray<float4> hand_joints_position_before_device;

	std::vector<float4> hand_joints_position_host;
	pcl::device::DeviceArray<float4> hand_joints_position_device;

	std::vector<unsigned char> block_id_host;
	pcl::device::DeviceArray<unsigned char> block_id_device;

	pcl::device::DeviceArray<float4> interaction_warped_vertex_buffer;
	pcl::device::DeviceArray<float4> interaction_warped_normal_buffer;
	pcl::device::DeviceArray<float4> interaction_cano_vertex_buffer;
	pcl::device::DeviceArray<float4> interaction_cano_normal_buffer;
	pcl::device::DeviceArray<unsigned char> interaction_finger_idx_buffer;
	pcl::device::DeviceArray<int3> interaction_sphere_block_buffer;
	pcl::device::DeviceArray<float3> interaction_sphere_coordinate_buffer;
	pcl::device::DeviceArray<unsigned char> interaction_block_idx_buffer;
	pcl::device::DeviceArray<float3> local_contact_vector_spheremesh_buffer;

	pcl::device::DeviceArray<float4> valid_interaction_warped_vertex;
	pcl::device::DeviceArray<float4> valid_interaction_warped_normal;
	pcl::device::DeviceArray<float4> valid_interaction_cano_vertex;
	pcl::device::DeviceArray<float4> valid_interaction_cano_normal;
	pcl::device::DeviceArray<unsigned char> valid_interaction_finger_idx;
	pcl::device::DeviceArray<int3> valid_interaction_sphere_block;
	pcl::device::DeviceArray<float3> valid_interaction_sphere_coordinate;
	pcl::device::DeviceArray<unsigned char> valid_interaction_block_idx;
	pcl::device::DeviceArray<float3> valid_local_contact_vector_spheremesh;

	std::vector<float4> valid_interaction_warped_vertex_host;
	std::vector<float4> valid_interaction_warped_normal_host;
	std::vector<float4> valid_interaction_cano_vertex_host;
	std::vector<float4> valid_interaction_cano_normal_host;
	std::vector<unsigned char> valid_interaction_finger_idx_host;
	std::vector<unsigned char> valid_interaction_block_idx_host;
	std::vector<int3> valid_interaction_sphere_block_host;
	std::vector<float3> valid_interaction_sphere_coordinate_host;
	std::vector<float3> valid_local_contact_vector_spheremesh_host;

	//for friction
	std::vector<std::vector<float3>> interaction_vertex_each_joint;
	std::vector<std::vector<float3>> interaction_normal_each_joint;
	std::vector<bool> finger_touch_status;
	int touch_on = 20;
	int touch_off = 10;

	//for opengl render
	std::vector<float3> use_interaction_vertex;
	std::vector<float3> use_interaction_normal;
	std::vector<unsigned char> use_interaction_joint_idx;

	int max_corresp_num = 5 * MAX_INTERACTION_CORRESPONDENCE;
	int joint_num = 0;
	
	int valid_interaction_corres_num = 0;

	int phalange_num = 17;

public:

	void initial_finger_radius(std::vector<float>& joint_radius);

	void initial_SphereHand_radius(std::vector<float>& joint_radius);

	void initial_finger_blocks(std::vector<glm::ivec3>& finger_blocks);

	void initial_phalange(std::vector<unsigned char>& phalange_centerId);

	void set_finger_joint_position(std::vector<float3>& joint_positions);

	void update_hand_joint_position(std::vector<glm::vec3>& joint_positions);

	void store_hand_joint_position_before(std::vector<glm::vec3>& joint_positions);

	void set_phalange_transformation_global2local(std::vector<Eigen::Matrix4f> phalange_global2local);

	void set_phalange_transformation_local2global_before(std::vector<Eigen::Matrix4f> phalange_global2local);

	void set_phalange_transformation_local2global(std::vector<Eigen::Matrix4f> phalange_global2local);//std::vector<Eigen::Matrix<float, 4, 4, Eigen::ColMajor>

	void obtain_touch_status(mat34 &object_rigid_motion);

	mat34 Eigen2mat(Eigen::Matrix4f matrix);

	std::vector<float4> get_valid_interaction_corrs_warped_vertex();
	std::vector<float4> get_valid_interaction_corrs_warped_normal();
	std::vector<float4> get_valid_interaction_corrs_cano_vertex();
	std::vector<float4> get_valid_interaction_corrs_cano_normal();
	std::vector<unsigned char> get_valid_interaction_finger_idx();
	std::vector<unsigned char> get_valid_interaction_block_idx();
	std::vector<int3> get_valid_interaction_sphere_block();
	std::vector<float3> get_valid_interaction_sphere_coordinate();

	Interaction()
	{
		interaction_warped_vertex_buffer.create(5 * MAX_INTERACTION_CORRESPONDENCE);
		interaction_warped_normal_buffer.create(5 * MAX_INTERACTION_CORRESPONDENCE);

		interaction_cano_vertex_buffer.create(5 * MAX_INTERACTION_CORRESPONDENCE);
		interaction_cano_normal_buffer.create(5 * MAX_INTERACTION_CORRESPONDENCE);

		interaction_finger_idx_buffer.create(5 * MAX_INTERACTION_CORRESPONDENCE);

		interaction_sphere_block_buffer.create(5 * MAX_INTERACTION_CORRESPONDENCE);
		interaction_sphere_coordinate_buffer.create(5 * MAX_INTERACTION_CORRESPONDENCE);
		interaction_block_idx_buffer.create(5 * MAX_INTERACTION_CORRESPONDENCE);
		local_contact_vector_spheremesh_buffer.create(5 * MAX_INTERACTION_CORRESPONDENCE);
	}

	~Interaction() {}
};