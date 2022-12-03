#include "Interaction.h"

void Interaction::initial_finger_radius(std::vector<float>& joint_radius)
{
	joint_radius_host.clear();

	//thumb
	joint_radius_host.push_back(joint_radius[16]);
	joint_radius_host.push_back(joint_radius[17]);
	//index finger
	joint_radius_host.push_back(joint_radius[12]);
	joint_radius_host.push_back(joint_radius[13]);
	//middle finger
	joint_radius_host.push_back(joint_radius[8]);
	joint_radius_host.push_back(joint_radius[9]);
	//ring finger
	joint_radius_host.push_back(joint_radius[4]);
	joint_radius_host.push_back(joint_radius[5]);
	//pinky finger
	joint_radius_host.push_back(joint_radius[0]);
	joint_radius_host.push_back(joint_radius[1]);

	joint_num = joint_radius_host.size();

	joint_radius_device.create(joint_radius_host.size());
	joint_radius_device.upload(joint_radius_host.data(), joint_radius_host.size());

	interaction_vertex_each_joint.resize(joint_num);
	interaction_normal_each_joint.resize(joint_num);
	finger_touch_status.resize(joint_num);
	for (int i = 0; i < joint_num; i++)
	{
		finger_touch_status[i] = false;
	}
}

void Interaction::initial_SphereHand_radius(std::vector<float>& joint_radius)
{
	hand_joints_radius_host.clear();

	hand_joints_radius_host = joint_radius;

	hand_joints_radius_device.create(hand_joints_radius_host.size());
	hand_joints_radius_device.upload(hand_joints_radius_host.data(), hand_joints_radius_host.size());
}

void Interaction::initial_finger_blocks(std::vector<glm::ivec3>& finger_blocks)
{
	finger_blocks_host.clear();
	block_id_host.clear();

	//this 15 could be changed based on necessary 
	for (int i = 0; i < 15; i++)
	{
		/*if(i%3==2)
		continue;*/
		finger_blocks_host.push_back(make_int3(finger_blocks[i][0], finger_blocks[i][1], finger_blocks[i][2]));
		block_id_host.push_back(i);
	}

	//add thumb tip 20200522
	finger_blocks_host.push_back(make_int3(finger_blocks[27][0], finger_blocks[27][1], finger_blocks[27][2]));
	block_id_host.push_back(27);

	finger_blocks_device.create(finger_blocks_host.size());
	finger_blocks_device.upload(finger_blocks_host.data(), finger_blocks_host.size());

	block_id_device.create(block_id_host.size());
	block_id_device.upload(block_id_host.data(), block_id_host.size());
}

void Interaction::initial_phalange(std::vector<unsigned char>& phalange_centerId)
{
	//initial phalange center id 
	phalange_center_id_host = phalange_centerId;
	
	phalange_center_id_device.create(phalange_center_id_host.size());
	phalange_center_id_device.upload(phalange_center_id_host.data(), phalange_center_id_host.size());

	//initial block2phalange
	int block_num = 30;
	block2phalange_host.resize(block_num);
	block2phalange_host[0] = 6;
	block2phalange_host[1] = 5;
	block2phalange_host[2] = 4;

	block2phalange_host[3] = 9;
	block2phalange_host[4] = 8;
	block2phalange_host[5] = 7;

	block2phalange_host[6] = 12;
	block2phalange_host[7] = 11;
	block2phalange_host[8] = 10;

	block2phalange_host[9] = 15;
	block2phalange_host[10] = 14;
	block2phalange_host[11] = 13;

	block2phalange_host[12] = 3;
	block2phalange_host[13] = 2;
	block2phalange_host[14] = 1;

	//from 20-26, the block is not controlled by current phalange,
	//the transformation should be calculated further, it can be achieved
	for (int i = 15; i < 27; i++) 
		block2phalange_host[i] = 0;

	block2phalange_host[27] = 3;

	block2phalange_host[28] = 16;
	block2phalange_host[29] = 16;

	block2phalange_device.create(block2phalange_host.size());
	block2phalange_device.upload(block2phalange_host.data(), block2phalange_host.size());

	//create the global2local and local2global transformation
	global2local_phalange_host.resize(phalange_num);
	global2local_phalange_device.create(phalange_num);

	local2global_phalange_before_host.resize(phalange_num);
	local2global_phalange_before_device.create(phalange_num);

	local2global_phalange_host.resize(phalange_num);
	local2global_phalange_device.create(phalange_num);

}

void Interaction::set_finger_joint_position(std::vector<float3>& joint_positions)
{
	joint_position_host.resize(10);

	float3 joint_temp;

	//thumb
	joint_temp = joint_positions[16];
	joint_position_host[0] = make_float4(joint_temp.x, -joint_temp.y, joint_temp.z, 1);

	joint_temp = joint_positions[17];
	joint_position_host[1] = make_float4(joint_temp.x, -joint_temp.y, joint_temp.z, 1);

	//index finger
	joint_temp = joint_positions[12];
	joint_position_host[2] = make_float4(joint_temp.x, -joint_temp.y, joint_temp.z, 1);

	joint_temp = joint_positions[13];
	joint_position_host[3] = make_float4(joint_temp.x, -joint_temp.y, joint_temp.z, 1);

	//middle finger
	joint_temp = joint_positions[8];
	joint_position_host[4] = make_float4(joint_temp.x, -joint_temp.y, joint_temp.z, 1);

	joint_temp = joint_positions[9];
	joint_position_host[5] = make_float4(joint_temp.x, -joint_temp.y, joint_temp.z, 1);

	//pinky finger
	joint_temp = joint_positions[4];
	joint_position_host[6] = make_float4(joint_temp.x, -joint_temp.y, joint_temp.z, 1);

	joint_temp = joint_positions[5];
	joint_position_host[7] = make_float4(joint_temp.x, -joint_temp.y, joint_temp.z, 1);

	//little finger
	joint_temp = joint_positions[0];
	joint_position_host[8] = make_float4(joint_temp.x, -joint_temp.y, joint_temp.z, 1);

	joint_temp = joint_positions[1];
	joint_position_host[9] = make_float4(joint_temp.x, -joint_temp.y, joint_temp.z, 1);

	//upload to GPU
	joint_position_device.create(joint_position_host.size());
	joint_position_device.upload(joint_position_host.data(), joint_position_host.size());
}

void Interaction::update_hand_joint_position(std::vector<glm::vec3>& joint_positions)
{
	hand_joints_position_host.resize(joint_positions.size());

	for (int i = 0; i < joint_positions.size(); i++)
	{
		hand_joints_position_host[i] = make_float4(joint_positions[i][0], -joint_positions[i][1], joint_positions[i][2], 1);
	}

	//upload to GPU
	hand_joints_position_device.create(hand_joints_position_host.size());
	hand_joints_position_device.upload(hand_joints_position_host.data(), hand_joints_position_host.size());
}

void Interaction::store_hand_joint_position_before(std::vector<glm::vec3>& joint_positions)
{
	hand_joints_position_before_host.resize(joint_positions.size());

	for (int i = 0; i < joint_positions.size(); i++)
	{
		hand_joints_position_before_host[i] = make_float4(joint_positions[i][0], -joint_positions[i][1], joint_positions[i][2], 1);
	}

	//upload to GPU
	hand_joints_position_before_device.create(hand_joints_position_before_host.size());
	hand_joints_position_before_device.upload(hand_joints_position_before_host.data(), hand_joints_position_before_host.size());
}

mat34 Interaction::Eigen2mat(Eigen::Matrix4f matrix)
{
	mat34 transform_m34;
	transform_m34.rot.m00() = matrix(0, 0); transform_m34.rot.m01() = matrix(0, 1); transform_m34.rot.m02() = matrix(0, 2);
	transform_m34.rot.m10() = matrix(1, 0); transform_m34.rot.m11() = matrix(1, 1); transform_m34.rot.m12() = matrix(1, 2);
	transform_m34.rot.m20() = matrix(2, 0); transform_m34.rot.m21() = matrix(2, 1); transform_m34.rot.m22() = matrix(2, 2);

	transform_m34.trans.x = matrix(0, 3); transform_m34.trans.y = matrix(1, 3); transform_m34.trans.z = matrix(2, 3);

	return transform_m34;
}

void Interaction::set_phalange_transformation_global2local(std::vector<Eigen::Matrix4f> phalange_global)
{
	//std::cout << "phalange size:" << phalange_global.size() << std::endl;
	for (int i = 0; i < phalange_num; i++)
	{
		/*std::cout << i << ":" << std::endl;
		std::cout << phalange_global[i] << std::endl;*/
		Eigen::Matrix4f global2local_temp = phalange_global[i].inverse();
		global2local_phalange_host[i] = Eigen2mat(global2local_temp);
	}

	global2local_phalange_device.upload(global2local_phalange_host.data(), global2local_phalange_host.size());
}

void Interaction::set_phalange_transformation_local2global_before(std::vector<Eigen::Matrix4f> phalange_global)
{
	for (int i = 0; i < phalange_num; i++)
	{
		Eigen::Matrix4f local2global_temp = phalange_global[i];
		local2global_phalange_before_host[i] = Eigen2mat(local2global_temp);
	}

	local2global_phalange_before_device.upload(local2global_phalange_before_host.data(), local2global_phalange_before_host.size());
}

void Interaction::set_phalange_transformation_local2global(std::vector<Eigen::Matrix4f> phalange_global)
{
	for (int i = 0; i < phalange_num; i++)
	{
		Eigen::Matrix4f local2global_temp = phalange_global[i];
		local2global_phalange_host[i] = Eigen2mat(local2global_temp);
	}

	local2global_phalange_device.upload(local2global_phalange_host.data(), local2global_phalange_host.size());
}

void Interaction::obtain_touch_status(mat34 &object_rigid_motion)
{
	valid_interaction_warped_vertex.download(valid_interaction_warped_vertex_host);
	valid_interaction_warped_normal.download(valid_interaction_warped_normal_host);
	valid_interaction_finger_idx.download(valid_interaction_finger_idx_host);

	//
	std::vector<std::vector<float3>> interaction_vertex_each_joint_temp;
	std::vector<std::vector<float3>> interaction_normal_each_joint_temp;
	interaction_vertex_each_joint_temp.resize(joint_num);
	interaction_normal_each_joint_temp.resize(joint_num);

	//clear
	use_interaction_vertex.clear();
	use_interaction_normal.clear();
	use_interaction_joint_idx.clear();

	//detach the interaction vertex
	for (int i = 0; i < valid_interaction_finger_idx_host.size(); i++)
	{
		unsigned char joint_id = valid_interaction_finger_idx_host[i];
		float4 vertex_temp = valid_interaction_warped_vertex_host[i];
		float4 normal_temp = valid_interaction_warped_normal_host[i];

		float radius = joint_radius_host[joint_id];
		float4 joint_pos = joint_position_host[joint_id];

		float3 live_v = object_rigid_motion.rot*make_float3(vertex_temp.x, vertex_temp.y, vertex_temp.z) + object_rigid_motion.trans;
		float3 live_n = object_rigid_motion.rot*make_float3(normal_temp.x, normal_temp.y, normal_temp.z);
		
		if (norm(live_v - make_float3(joint_pos.x, joint_pos.y, joint_pos.z)) < radius)
		{
			interaction_vertex_each_joint_temp[joint_id].push_back(make_float3(vertex_temp.x, vertex_temp.y, vertex_temp.z));
			interaction_normal_each_joint_temp[joint_id].push_back(make_float3(normal_temp.x, normal_temp.y, normal_temp.z));

			use_interaction_vertex.push_back(make_float3(live_v.x, -live_v.y, live_v.z));
			use_interaction_normal.push_back(make_float3(live_n.x, -live_n.y, live_n.z));
			use_interaction_joint_idx.push_back(joint_id);
		}

		
	}

	//change the touch status based on the prior status and interaction detection
	for (int joint_id = 0; joint_id < joint_num; joint_id++)
	{
		if (finger_touch_status[joint_id])//for those joints touching on the object, check if they touch off the object
		{
			if (interaction_vertex_each_joint_temp[joint_id].size() < touch_off)//
			{
				interaction_vertex_each_joint[joint_id].clear();
				interaction_normal_each_joint[joint_id].clear();
				finger_touch_status[joint_id] = false;
			}
		}
		else//for those joints touching off the object, check if they touch on the object, if yes, give the interaction vertex and set the touch status on
		{
			if (interaction_vertex_each_joint_temp[joint_id].size() > touch_on)
			{
				interaction_vertex_each_joint[joint_id] = interaction_vertex_each_joint_temp[joint_id];
				interaction_normal_each_joint[joint_id] = interaction_normal_each_joint_temp[joint_id];
				finger_touch_status[joint_id] = true;
			}
		}

		/*if (interaction_vertex_each_joint_temp[joint_id].size() > touch_on)
		{
			interaction_vertex_each_joint[joint_id] = interaction_vertex_each_joint_temp[joint_id];
			interaction_normal_each_joint[joint_id] = interaction_normal_each_joint_temp[joint_id];
			finger_touch_status[joint_id] = true;
		}
		else
		{
			interaction_vertex_each_joint[joint_id].clear();
			interaction_normal_each_joint[joint_id].clear();
			finger_touch_status[joint_id] = false;
		}*/
	}
}

std::vector<float4> Interaction::get_valid_interaction_corrs_warped_vertex()
{
	valid_interaction_warped_vertex.download(valid_interaction_warped_vertex_host);

	return valid_interaction_warped_vertex_host;
}

std::vector<float4> Interaction::get_valid_interaction_corrs_warped_normal()
{
	valid_interaction_warped_normal.download(valid_interaction_warped_normal_host);

	return valid_interaction_warped_normal_host;
}

std::vector<float4> Interaction::get_valid_interaction_corrs_cano_vertex()
{
	valid_interaction_cano_vertex.download(valid_interaction_cano_vertex_host);

	return valid_interaction_cano_vertex_host;
}

std::vector<float4> Interaction::get_valid_interaction_corrs_cano_normal()
{
	valid_interaction_cano_normal.download(valid_interaction_cano_normal_host);

	return valid_interaction_cano_normal_host;
}

std::vector<unsigned char> Interaction::get_valid_interaction_finger_idx()
{
	valid_interaction_finger_idx.download(valid_interaction_finger_idx_host);

	return valid_interaction_finger_idx_host;
}

std::vector<unsigned char> Interaction::get_valid_interaction_block_idx()
{
	valid_interaction_block_idx.download(valid_interaction_block_idx_host);

	return valid_interaction_block_idx_host;
}

std::vector<int3> Interaction::get_valid_interaction_sphere_block()
{
	valid_interaction_sphere_block.download(valid_interaction_sphere_block_host);

	return valid_interaction_sphere_block_host;
}

std::vector<float3> Interaction::get_valid_interaction_sphere_coordinate()
{
	valid_interaction_sphere_coordinate.download(valid_interaction_sphere_coordinate_host);

	return valid_interaction_sphere_coordinate_host;
}