#pragma once
#include "Energy.h"
#include "tracker/Types.h"
#include <vector>
#include <vector_types.h>
#include <vector_functions.hpp>
#include "tracker/HModel/Model.h"
#include "../CommonVariances.h"
#include <iomanip>
#include <fstream>

namespace energy {

	class Keypoint2D : public Energy {
	private:
		float Keypoint2D_weight = 10; //10 is good
		std::string keypoint_2D_debug_file_path = "../../../../result/keypoint_2D_debug_file.txt";
	public:
		Eigen::Matrix<float, 2, 3 > calculate_projectionJacobian(glm::vec3 sphere_pos, camera_intr para_cam);
		void track(LinearSystem& sys, const Model *model, const std::vector<int> &keypoint2block, const std::vector<int> &keypoint2SphereCenter,
			const std::vector<float2> &keypoint_2D_GT_vec, const std::vector<int> &using_keypoint, camera_intr para_cam, int frame_id, int itr_id);
	};

	class Keypoint3D : public Energy {
	public:
		float Keypoint3D_weight = 10;//10 is good
		std::string keypoint_3D_debug_file_path = "../../../../result/keypoint_3D_debug_file.txt";
		ofstream keypoint_3D_debug_file;

	public:
//		Eigen::Matrix<float, 2, 3 > calculate_projectionJacobian(glm::vec3 sphere_pos, camera_intr para_cam);
		void track(LinearSystem& sys, const Model *model, 
				   const std::vector<int> &keypoint2block, 
				   const std::vector<int> &keypoint2SphereCenter,
				   const std::vector<float3> &keypoint_3D_GT_vec, 
				   const std::vector<int> &using_keypoint,
				   Eigen::Matrix4f camera_pose, 
				   int frame_id, int itr_id, float weight_factor);

		void track_with_weight(LinearSystem& sys, const Model *model, 
							   const std::vector<int> &keypoint2block, 
							   const std::vector<int> &keypoint2SphereCenter,
							   const std::vector<float3> &keypoint_3D_GT_vec, 
							   const std::vector<float> &keypoint_3D_visible, 
							   const std::vector<int> &using_keypoint, 
							   Eigen::Matrix4f camera_pose, 
							   int frame_id, int itr_id);
	};

	// Add By HuHaoyu
	class KeypointTips : public Energy {
	public:
		float KeypointTips_weight = 1e10;

	public:
		void track(LinearSystem& sys, const Model *model, 
				   const std::vector<int> &keypoint2block, 
				   const std::vector<int> &keypoint2SphereCenter,
				   const std::vector<Eigen::Vector3f> &keypoint_3D_GT_vec, 
				   const std::vector<int> &using_keypoint);
	};
}
