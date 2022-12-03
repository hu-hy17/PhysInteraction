#pragma once
#include "ComputeJacobianRow.h"

//== NAMESPACE ================================================================
namespace cudax {
	//=============================================================================

	//following structure is impletemented by ZH
	struct ComputeJacobianInteractionJoints : public ComputeJacobianRow {

		const glm::mat3x3& rigid_motion_r;
		const glm::vec3& rigid_motion_t;

		float * centers;
		float * radii;
		int * blocks;
		float * tangent_points;
		float * outline;
		int * blockid_to_jointid_map;

		float * _JtJ;
		float * _Jte;

		float4 * warped_vertice;
		float4 * warped_normal;
		unsigned char * vertice_joint_idx;
		unsigned char * finger2jointpoint_id;
		unsigned char * finger2block_id;

	public:
		ComputeJacobianInteractionJoints(J_row* J_raw, float* e_raw) :
			ComputeJacobianRow(J_raw, e_raw),
			rigid_motion_r(*cudax::rigid_motion_r),
			rigid_motion_t(*cudax::rigid_motion_t){

			this->centers = thrust::raw_pointer_cast(device_pointer_centers->data());
			this->radii = thrust::raw_pointer_cast(device_pointer_radii->data());
			this->blocks = thrust::raw_pointer_cast(device_pointer_blocks->data());
			this->tangent_points = thrust::raw_pointer_cast(device_pointer_tangent_points->data());
			this->outline = thrust::raw_pointer_cast(device_pointer_outline->data());
			this->blockid_to_jointid_map = thrust::raw_pointer_cast(device_pointer_blockid_to_jointid_map->data());

			warped_vertice = _warped_vertex;
			warped_normal = _warped_normal;
			vertice_joint_idx = _vertex_joint_idx;

			finger2jointpoint_id = thrust::raw_pointer_cast(&(*finger2joint_point_idx)[0]);
			finger2block_id = thrust::raw_pointer_cast(&(*finger2block_idx)[0]);
		}

		__device__
			void skeleton_jacobian(const int joint_id, const glm::vec3& pos, J_row* sub_J, glm::vec3 nrm = glm::vec3(0), bool project = false) {

			//float j_buffer[CHAIN_MAX_LENGTH];
			for (int i_column = 0; i_column < CHAIN_MAX_LENGTH; i_column++) {
				int jointinfo_id = chains[joint_id].data[i_column];
				if (jointinfo_id == -1) break;
				const CustomJointInfo& jinfo = jointinfos[jointinfo_id];
				glm::vec3& axis = jointinfos[jointinfo_id].axis;

				glm::vec3 col;
				switch (jinfo.type) {
				case 1: {
					col = glm::vec3(jointinfos[jointinfo_id].mat * glm::vec4(axis, 1));
					break;
				}
				case 0: {
					glm::vec3 t(jointinfos[jointinfo_id].mat[3][0], jointinfos[jointinfo_id].mat[3][1], jointinfos[jointinfo_id].mat[3][2]);
					glm::vec3 a = glm::normalize(glm::vec3(jointinfos[jointinfo_id].mat * glm::vec4(axis, 1)) - t);
					col = glm::cross(a, pos - t);
					break;
				}
				}

				{
					sub_J->data[jinfo.index] = glm::dot(col, nrm);
				}
			}
		}

		__device__ void assemble_linear_system(int constraint_index, glm::vec3 live_vertice, glm::vec3 live_normal, glm::vec3 joint_center, float joint_radius, int block_id) {
			
			///--- Access to a 3xN block of Jacobian
			/*J_row* J_sub = J_raw + constraint_index;
			float* e_sub = e_raw + constraint_index;

			if (isnan(live_normal[0]) || isnan(live_normal[1]) || isnan(live_normal[2]))
			{
				*e_sub = 0;
				for (int i = 0; i < NUM_THETAS; i++)
				{
					J_sub->data[i] = 0;
				}
				return;
			}

			float r = joint_radius + glm::dot(live_normal, live_vertice - joint_center);

			if (r <= 0)
			{
				*e_sub = 0;
				for (int i = 0; i < NUM_THETAS; i++)
				{
					J_sub->data[i] = 0;
				}
				return;
			}

			int joint_id = blockid_to_jointid_map[block_id];

			*e_sub = r;
			skeleton_jacobian(joint_id, joint_center, J_sub, live_normal);*/

			if (isnan(live_normal[0]) || isnan(live_normal[1]) || isnan(live_normal[2])) return;
			float r = joint_radius + glm::dot(live_normal, live_vertice - joint_center);
			if (r <= 0)return;

			int joint_id = blockid_to_jointid_map[block_id];

			///--- Access to a 3xN block of Jacobian
			J_row* J_sub = J_raw + constraint_index;
			float* e_sub = e_raw + constraint_index;

			*e_sub = r;
			skeleton_jacobian(joint_id, joint_center, J_sub, live_normal);
		}

		__device__
			void operator()(int index) {

			//calculate the live vertex and live normal in hand tracking coordination
			float4 warp_vertice_temp = warped_vertice[index];
			glm::vec3 warp_vertice_v3 = glm::vec3(warp_vertice_temp.x, warp_vertice_temp.y, warp_vertice_temp.z);
			glm::vec3 live_vertice = rigid_motion_r*warp_vertice_v3 + rigid_motion_t;
			live_vertice[1] = -live_vertice[1];

			float4 warp_normal_temp = warped_normal[index];
			glm::vec3 warp_normal_v3 = glm::vec3(warp_normal_temp.x, warp_normal_temp.y, warp_normal_temp.z);
			glm::vec3 live_normal = rigid_motion_r*warp_normal_v3;
			live_normal[1] = -live_normal[1];
			
			int finger_idx = (int)vertice_joint_idx[index];
			int joint_position_idx = finger2jointpoint_id[finger_idx];
			int block_idx = finger2block_id[finger_idx];

			glm::vec3 center_temp = glm::vec3(centers[D * joint_position_idx], centers[D * joint_position_idx + 1], centers[D * joint_position_idx + 2]);
			float radii_temp = radii[joint_position_idx];

			assemble_linear_system(index, live_vertice, live_normal, center_temp, radii_temp, block_idx);
		}

	};


	//following structure is impletemented by ZH
	struct ComputeJacobianInteractionBlocks : public ComputeJacobianRow {

		const glm::mat3x3& rigid_motion_r;
		const glm::vec3& rigid_motion_t;

		float * centers;
		float * radii;
		int * blocks;
		float * tangent_points;
		float * outline;
		int * blockid_to_jointid_map;

		float * _JtJ;
		float * _Jte;

		float4 * warped_vertice;
		float4 * warped_normal;
		int3 * sphere_block;
		float3 * sphere_coordinate;
		unsigned char * vertice_joint_idx;
		unsigned char * vertice_block_idx;
		unsigned char * finger2jointpoint_id;
		unsigned char * finger2block_id;

	public:
		ComputeJacobianInteractionBlocks(J_row* J_raw, float* e_raw) :
			ComputeJacobianRow(J_raw, e_raw),
			rigid_motion_r(*cudax::rigid_motion_r),
			rigid_motion_t(*cudax::rigid_motion_t) {

			this->centers = thrust::raw_pointer_cast(device_pointer_centers->data());
			this->radii = thrust::raw_pointer_cast(device_pointer_radii->data());
			this->blocks = thrust::raw_pointer_cast(device_pointer_blocks->data());
			this->tangent_points = thrust::raw_pointer_cast(device_pointer_tangent_points->data());
			this->outline = thrust::raw_pointer_cast(device_pointer_outline->data());
			this->blockid_to_jointid_map = thrust::raw_pointer_cast(device_pointer_blockid_to_jointid_map->data());

			warped_vertice = _warped_vertex;
			warped_normal = _warped_normal;
			sphere_block = _sphere_block;
			sphere_coordinate = _sphere_coordinate;
			vertice_block_idx = _vertex_block_idx;
		}

		__device__ void Point2SpherePhalange(const glm::vec3 Point, const glm::vec3 C1, const glm::vec3 C2, const float r1, const float r2, glm::vec3& sphere_pos, float& sphere_rad)
		{
			float tsdf;

			const float epsilon = 1e-4;

			//calculate the perpendicular root from Point to skeleton constructed by C1 and C2
			glm::vec3 V_C1P = Point - C1;
			glm::vec3 V_C1C2 = C2 - C1;
			float length_VC1C2 = length(V_C1C2);
			glm::vec3 p = C1 + V_C1C2*(dot(V_C1C2, V_C1P) / (length_VC1C2*length_VC1C2));
			glm::vec3 V_pP = Point - p;
			float length_VpP = length(V_pP);

			//calculate Cp: the intersection between the perpendicular line across Point with the phalange surface and the skeleton line of C1 and C2 
			glm::vec3 Cp;
			if (abs(r1 - r2) < epsilon)
			{
				Cp = p;// this is the sphere center
			}
			else
			{
				glm::vec3 C = C1 + (C2 - C1)*(r1 / (r1 - r2));
				glm::vec3 V_C1C = C - C1;
				float length_VC1C = length(V_C1C);
				float sin_theta = r1 / length_VC1C;
				float cos_theta = sqrt(1 - (sin_theta*sin_theta));
				float tan_theta = sin_theta / cos_theta;

				glm::vec3 f_normalized = V_C1C*(1 / length_VC1C);
				Cp = p - f_normalized*(tan_theta*length_VpP);// this is the sphere center
			}

			glm::vec3 V_C1Cp = Cp - C1;
			float length_VC1Cp = length(V_C1Cp);

			if (dot(V_C1Cp, V_C1C2) < 0)
			{
				//				tsdf = length(Point - C1) - r1;
				sphere_pos = C1;
				sphere_rad = r1;
			}
			else
			{
				if (length_VC1C2 >= length_VC1Cp)
				{
					float alpha = 1 - length_VC1Cp / length_VC1C2;
					float r_Cp = alpha*r1 + (1 - alpha)*r2;

					//					tsdf = length(Point - Cp) - r_Cp;
					sphere_pos = Cp;
					sphere_rad = r_Cp;
				}
				else
				{
					//					tsdf = length(Point - C2) - r2;
					sphere_pos = C2;
					sphere_rad = r2;
				}
			}
		}

		__device__ void find_correspondence_sphere(const glm::vec3  vertice, const int block_id, glm::vec3 &sphere_pos, float &sphere_rad)
		{
			int index1 = blocks[D * block_id];
			int index2 = blocks[D * block_id + 1];
			glm::vec3 c1 = glm::vec3(centers[D * index1], centers[D * index1 + 1], centers[D * index1 + 2]);
			glm::vec3 c2 = glm::vec3(centers[D * index2], centers[D * index2 + 1], centers[D * index2 + 2]);
			float r1 = radii[index1];
			float r2 = radii[index2];

			if (r1 < r2)
			{
				glm::vec3 c_temp;
				c_temp = c1;
				c1 = c2;
				c2 = c_temp;

				float r_temp;
				r_temp = r1;
				r1 = r2;
				r2 = r_temp;
			}

			Point2SpherePhalange(vertice, c1, c2, r1, r2, sphere_pos, sphere_rad);
		}

		__device__
			void skeleton_jacobian(const int joint_id, const glm::vec3& pos, J_row* sub_J, glm::vec3 nrm = glm::vec3(0), bool project = false) {

			//float j_buffer[CHAIN_MAX_LENGTH];
			for (int i_column = 0; i_column < CHAIN_MAX_LENGTH; i_column++) {
				int jointinfo_id = chains[joint_id].data[i_column];
				if (jointinfo_id == -1) break;
				const CustomJointInfo& jinfo = jointinfos[jointinfo_id];
				glm::vec3& axis = jointinfos[jointinfo_id].axis;

				glm::vec3 col;
				switch (jinfo.type) {
					case 1: {
						col = glm::vec3(jointinfos[jointinfo_id].mat * glm::vec4(axis, 1));
						break;
					}
					case 0: {
						glm::vec3 t(jointinfos[jointinfo_id].mat[3][0], jointinfos[jointinfo_id].mat[3][1], jointinfos[jointinfo_id].mat[3][2]);
						glm::vec3 a = glm::normalize(glm::vec3(jointinfos[jointinfo_id].mat * glm::vec4(axis, 1)) - t);
						col = glm::cross(a, pos - t);
						break;
					}
				}

				{
					sub_J->data[jinfo.index] = glm::dot(col, nrm);
				}
			}
		}

		__device__ void assemble_linear_system(int constraint_index, glm::vec3 live_vertice, glm::vec3 live_normal, glm::vec3 joint_center, float joint_radius, int block_id) {

			if (isnan(live_normal[0]) || isnan(live_normal[1]) || isnan(live_normal[2])) return;
			float r = joint_radius + glm::dot(live_normal, live_vertice - joint_center);
			if (r <= 0)return;

			int joint_id = blockid_to_jointid_map[block_id];

			///--- Access to a 3xN block of Jacobian
			J_row* J_sub = J_raw + constraint_index;
			float* e_sub = e_raw + constraint_index;

			*e_sub = r;
			skeleton_jacobian(joint_id, joint_center, J_sub, live_normal);

		}

		__device__
			void operator()(int index) {

			//calculate the live vertex and live normal in hand tracking coordination
			float4 warp_vertice_temp = warped_vertice[index];
			glm::vec3 warp_vertice_v3 = glm::vec3(warp_vertice_temp.x, warp_vertice_temp.y, warp_vertice_temp.z);
			glm::vec3 live_vertice = rigid_motion_r*warp_vertice_v3 + rigid_motion_t;
			live_vertice[1] = -live_vertice[1];

			float4 warp_normal_temp = warped_normal[index];
			glm::vec3 warp_normal_v3 = glm::vec3(warp_normal_temp.x, warp_normal_temp.y, warp_normal_temp.z);
			glm::vec3 live_normal = rigid_motion_r*warp_normal_v3;
			live_normal[1] = -live_normal[1];

			//obtain the block idx
			int block_idx = vertice_block_idx[index];

			//find the closest sphere
			glm::vec3 center_temp;
			float radii_temp;

			//find correspondent sphere by calculating with the closest block
#if 0
			find_correspondence_sphere(live_vertice, block_idx, center_temp, radii_temp);
#else
			//find the closest sphere by block coordinate
			{
				int3 block_temp = sphere_block[index];
				float3 block_coordinate_temp = sphere_coordinate[index];
				float epsilon = 0.0001;

				if (block_temp.z > NUM_CENTERS)
				{
					float coordinate_sum = block_coordinate_temp.x + block_coordinate_temp.y;

					if (abs(1 - coordinate_sum) > epsilon)
						return;
					else
					{
						int index1 = block_temp.x;
						int index2 = block_temp.y;
						glm::vec3 c1 = glm::vec3(centers[D * index1], centers[D * index1 + 1], centers[D * index1 + 2]);
						glm::vec3 c2 = glm::vec3(centers[D * index2], centers[D * index2 + 1], centers[D * index2 + 2]);
						float r1 = radii[index1];
						float r2 = radii[index2];

						center_temp = c1*block_coordinate_temp.x + c2*block_coordinate_temp.y;
						radii_temp = r1*block_coordinate_temp.x + r2*block_coordinate_temp.y;
					}
				}
				else
				{
					float coordinate_sum = block_coordinate_temp.x + block_coordinate_temp.y + block_coordinate_temp.z;

					if (abs(1 - coordinate_sum) > epsilon)
						return;
					else
					{
						int index1 = block_temp.x;
						int index2 = block_temp.y;
						int index3 = block_temp.z;
						glm::vec3 c1 = glm::vec3(centers[D * index1], centers[D * index1 + 1], centers[D * index1 + 2]);
						glm::vec3 c2 = glm::vec3(centers[D * index2], centers[D * index2 + 1], centers[D * index2 + 2]);
						glm::vec3 c3 = glm::vec3(centers[D * index3], centers[D * index3 + 1], centers[D * index3 + 2]);
						float r1 = radii[index1];
						float r2 = radii[index2];
						float r3 = radii[index3];

						center_temp = c1*block_coordinate_temp.x + c2*block_coordinate_temp.y + c3*block_coordinate_temp.z;
						radii_temp = r1*block_coordinate_temp.x + r2*block_coordinate_temp.y + r3*block_coordinate_temp.z;
					}
				}
			}
#endif
			//assemble jacobian data
			assemble_linear_system(index, live_vertice, live_normal, center_temp, radii_temp, block_idx);
		}

	};

}
