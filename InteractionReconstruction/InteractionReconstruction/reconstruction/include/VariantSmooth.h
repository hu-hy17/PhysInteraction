#pragma once
#include <vector>
#include <Eigen/Eigen>
#include <pcl/gpu/containers/device_array.h>
#include "gpu/dual_quaternion.hpp"

class VariantSmooth
{
public:

	std::vector<int> node_close2tips_idx_host;
	std::vector<Eigen::Vector3f> hand_fingertip_use;
	std::vector<float> radius_fingertip_use;
	std::vector<float> Node2Tips_dis_host;
	std::vector<float> Node2Tips_min_dis_host;
//	pcl::gpu::DeviceArray<float> Node2Tips_dis_device;
	std::vector<float> NodeSmoothCoef_host;
	pcl::gpu::DeviceArray<float> NodeSmoothCoef_device;

public:

	void fixed_smooth_coefficient(std::vector<float4>& h_node_coords, std::vector<DualQuaternion>& h_warp_field, std::vector<float3>& hand_key_points, std::vector<std::vector<int>>& node_tip_idx, std::vector<float>& variant_smooth, Eigen::Matrix4f rigid_pose);

	void cal_Euler_distance(std::vector<float4>& h_node_coords, std::vector<DualQuaternion>& h_warp_field, std::vector<float3>& hand_key_points, std::vector<std::vector<int>>& node_tip_idx, std::vector<float>& variant_smooth, Eigen::Matrix4f rigid_pose, int current_frame);

	void cal_Euler_distance2(std::vector<float4>& h_node_coords, std::vector<DualQuaternion>& h_warp_field, std::vector<float3>& hand_key_points, std::vector<float>& joint_radius, std::vector<std::vector<int>>& node_tip_idx, std::vector<float>& variant_smooth, Eigen::Matrix4f rigid_pose);

	void cal_node2fingertips(std::vector<float4>& h_node_coords, std::vector<DualQuaternion>& h_warp_field, std::vector<float3>& hand_key_points, std::vector<float>& joint_radius, std::vector<std::vector<int>>& node_tip_idx, std::vector<int>& node_close2tip_idx, std::vector<float>& variant_smooth, Eigen::Matrix4f rigid_pose);
};
