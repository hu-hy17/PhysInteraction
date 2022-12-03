#include "InteractionHandTracking.h"

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip> 

namespace energy
{
	/*InteractionHandTracking::InteractionHandTracking(Interaction& _interaction_data) :
		interaction_data(_interaction_data)
	{}*/

	void InteractionHandTracking::track_joints(Interaction& _interaction_data, LinearSystem &system, Eigen::Matrix4f rigid_motion, bool set_parameter, bool store_result, int frame_idx)
	{
		//clear the sys
		system.lhs = Matrix_MxN::Zero(num_thetas, num_thetas);
		system.rhs = VectorN::Zero(num_thetas);

		Eigen::Matrix3f rigid_mot_r = rigid_motion.block(0, 0, 3, 3);
		Eigen::Vector3f rigid_mot_t = rigid_motion.block(0, 3, 3, 1);

		int interaction_terms = _interaction_data.valid_interaction_warped_vertex.size();

		if (interaction_terms > 50)
		{
			if (set_parameter)
			{
				kernel_assign_interaction_ptr_joints(_interaction_data.valid_interaction_warped_vertex.ptr(), _interaction_data.valid_interaction_warped_normal.ptr(), _interaction_data.valid_interaction_finger_idx.ptr());
				kernel_interaction_assign_rigid_motion(rigid_mot_r.data(), rigid_mot_t.data());
				/*if (cudaGetLastError() != cudaSuccess)
				{
					std::cout << "something wrong" << std::endl;
				}*/
			}
		}

		kernel_interaction_joints(system.lhs.data(), system.rhs.data(), interaction_terms, store_result, frame_idx);

		//if (store_result)
		//{
		//	if (interaction_terms < 10)
		//		return;

		//	static std::vector<float4> valid_interaction_warped_vertex_host;
		//	valid_interaction_warped_vertex_host.clear();
		//	_interaction_data.valid_interaction_warped_vertex.download(valid_interaction_warped_vertex_host);

		//	static std::vector<unsigned char> valid_interaction_finger_idx_host;
		//	valid_interaction_finger_idx_host.clear();
		//	_interaction_data.valid_interaction_finger_idx.download(valid_interaction_finger_idx_host);

		//	unsigned char finger2jointpoint[10] = { 16,17,12,13,8,9,4,5,0,1 };
		//	unsigned char finger2block[10] = { 12,13,9,10,6,7,3,4,0,1 };

		//	std::ofstream output_file;
		//	std::string data_path = "";

		//	output_file.open(data_path + "Host_interaction" + std::to_string(frame_idx) + ".txt");
		//	output_file << "total number:" << interaction_terms << std::endl;
		//	for (size_t i = 0; i < interaction_terms; i++) {
		//		/*float4 warped_vertex = valid_interaction_warped_vertex_host[i];
		//		Eigen::Vector3f warped_vertex3f = Eigen::Vector3f(warped_vertex.x, warped_vertex.y, warped_vertex.z);
		//		Eigen::Vector3f live_vertex = rigid_mot_r*warped_vertex3f + rigid_mot_t;
		//		output_file << sqrt(live_vertex[0] * live_vertex[0] + live_vertex[1] * live_vertex[1] + live_vertex[2] * live_vertex[2]) << " ";*/

		//		int finger_idx = valid_interaction_finger_idx_host[i];
		//		output_file <<(float)finger2block[finger_idx] << " ";
		//	}
		//	output_file.close();
		//}
	}

	void InteractionHandTracking::track_blocks(Interaction& _interaction_data, LinearSystem &system, Eigen::Matrix4f rigid_motion, bool set_parameter, bool store_result, int frame_idx)
	{
		//clear the sys
		system.lhs = Matrix_MxN::Zero(num_thetas, num_thetas);
		system.rhs = VectorN::Zero(num_thetas);

		Eigen::Matrix3f rigid_mot_r = rigid_motion.block(0, 0, 3, 3);
		Eigen::Vector3f rigid_mot_t = rigid_motion.block(0, 3, 3, 1);

		int interaction_terms = _interaction_data.valid_interaction_warped_vertex.size();
		//		std::cout << interaction_terms << std::endl;
		if (interaction_terms > 10)
		{
			if (set_parameter)
			{
				kernel_assign_interaction_ptr_blocks(_interaction_data.valid_interaction_warped_vertex.ptr(), _interaction_data.valid_interaction_warped_normal.ptr(),
													 _interaction_data.valid_interaction_sphere_block.ptr(), _interaction_data.valid_interaction_sphere_coordinate.ptr(), _interaction_data.valid_interaction_block_idx.ptr());

				kernel_interaction_assign_rigid_motion(rigid_mot_r.data(), rigid_mot_t.data());
				/*if (cudaGetLastError() != cudaSuccess)
				{
				std::cout << "something wrong" << std::endl;
				}*/
			}

			kernel_interaction_blocks(system.lhs.data(), system.rhs.data(), interaction_terms, store_result, frame_idx);
		}
	}
}