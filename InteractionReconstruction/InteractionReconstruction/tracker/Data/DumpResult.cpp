#include "DumpResult.h"

void dumpObjModel(std::vector<float4> &valid_vertices_host, const std::string& file_path)
{
	if (valid_vertices_host.size() % 3 != 0)
	{
		std::cout << "vertice number doesn't right" << std::endl;
		exit(0);
	}

	std::ofstream model_file(file_path);
	if (!model_file.is_open())
	{
		std::cerr << "(DumpResult) Error: fail to open file " << file_path << std::endl;
		return;
	}

	std::cout << "dump object model to " << file_path << std::endl;

	for (int i = 0; i < valid_vertices_host.size(); i++)
		model_file << "v " << valid_vertices_host[i].x / 1000.0 << " " << valid_vertices_host[i].y / 1000.0 << " " << valid_vertices_host[i].z / 1000.0 << std::endl;

	int tri_num = valid_vertices_host.size() / 3;
	for (int i = 0; i < tri_num; i++)
		model_file << "f " << 3 * i + 1 << " " << 3 * i + 2 << " " << 3 * i + 3 << std::endl;

	std::cout << "dump object model successfully!" << std::endl;
}

void dumpObjModel(std::vector<float4> &valid_vertices_host, std::vector<float4> &valid_normal_host,
	const std::string& file_path)
{
	if (valid_vertices_host.size() % 3 != 0)
	{
		std::cout << "vertice number doesn't right" << std::endl;
		exit(0);
	}

	std::ofstream model_file(file_path);
	if (!model_file.is_open())
	{
		std::cerr << "(DumpResult) Error: fail to open file " << file_path << std::endl;
		return;
	}

	std::cout << "dump object model to " << file_path << std::endl;

	for (int i = 0; i < valid_vertices_host.size(); i++)
	{
		model_file << "v " << valid_vertices_host[i].x / 1000.0 << " " << valid_vertices_host[i].y / 1000.0 << " " << valid_vertices_host[i].z / 1000.0 << std::endl;
		model_file << "vn " << valid_normal_host[i].x << " " << valid_normal_host[i].y << " " << valid_normal_host[i].z << std::endl;
	}

	int tri_num = valid_vertices_host.size() / 3;
	for (int i = 0; i < tri_num; i++)
		model_file << "f " << 3 * i + 1 << " " << 3 * i + 2 << " " << 3 * i + 3 << std::endl;

	std::cout << "dump object model successfully" << std::endl;
}

void outputContactInfo(std::string file_prefix, Tracker& tracker)
{
	// output contact points and object pose
	auto contact_points = tracker.interaction_datas.get_valid_interaction_corrs_warped_vertex();
	auto contact_norm = tracker.interaction_datas.get_valid_interaction_corrs_warped_normal();

	Json::Value frame_data;
	Json::Value obj_motion_data;
	Json::Value contact_data;
	Json::Value contact_point_data;
	Json::Value contact_norm_data;

	Json::Value keypoint_data;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
			obj_motion_data.append(tracker.object_pose(i, j));
	}

	for (int i = 0; i < contact_points.size(); i++)
	{
		contact_point_data.append(contact_points[i].x);
		contact_point_data.append(contact_points[i].y);
		contact_point_data.append(contact_points[i].z);

		contact_norm_data.append(contact_norm[i].x);
		contact_norm_data.append(contact_norm[i].y);
		contact_norm_data.append(contact_norm[i].z);
	}
	contact_data["point"] = contact_point_data;
	contact_data["norm"] = contact_norm_data;

	for (std::string& name : tracker.joint_names_output_full)
	{
		auto joint_id = tracker.worker->model->centers_name_to_id_map[name];
		glm::vec3 center_temp = tracker.worker->model->centers[joint_id];
		float radius = tracker.worker->model->radii[joint_id];
		Json::Value joint_data;
		joint_data["joint_name"] = name;
		joint_data["center"].append(center_temp[0]);
		joint_data["center"].append(center_temp[1]);
		joint_data["center"].append(center_temp[2]);
		joint_data["radius"] = radius;
		keypoint_data.append(joint_data);
	}

	frame_data["frame_id"] = tracker.current_frame;
	frame_data["obj_motion"] = obj_motion_data;
	frame_data["contacts"] = contact_data;
	frame_data["keypoints"] = keypoint_data;

	Json::StyledWriter sw;
	ofstream os;
	os.open(file_prefix + std::to_string(tracker.current_frame) + ".json");
	if (!os.is_open())
	{
		std::cerr << "(DumpResult) Error: Can not open file for obj motion and contact info record!" << std::endl;
		return;
	}
	os << sw.write(frame_data);
	os.close();
}

void outputHandObjMotion(std::string filename, Tracker& tracker)
{
	static std::set<std::string> file_set;

	Json::StyledWriter sw;

	// convert motion info into json format
	Json::Value hand_motion;
	Json::Value obj_motion;
	Json::Value tips_target_pos;
	Json::Value tips_conf;
	Json::Value tips_org_conf;
	Json::Value motion_info;

	for (std::string& name : tracker.joint_names_output_full)
	{
		auto joint_id = tracker.worker->model->centers_name_to_id_map[name];
		glm::vec3 center_temp = tracker.worker->model->centers[joint_id];

		hand_motion.append(center_temp[0]);
		hand_motion.append(center_temp[1]);
		hand_motion.append(center_temp[2]);
	}
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
			obj_motion.append(tracker.object_pose(i, j));
	}
	for (auto p : tracker.tip_3D_keypoint_pos)
	{
		tips_target_pos.append(p.x());
		tips_target_pos.append(p.y());
		tips_target_pos.append(p.z());
	}
	for (auto c : tracker.worker->tips_rel_conf)
	{
		tips_conf.append(c);
	}
	for (auto c : tracker.worker->tips_org_conf)
	{
		tips_org_conf.append(c);
	}

	motion_info["frame_id"] = tracker.current_frame;
	motion_info["hand_motion"] = hand_motion;
	motion_info["tips_target_pos"] = tips_target_pos;
	motion_info["obj_motion"] = obj_motion;
	motion_info["tips_conf"] = tips_conf;
	motion_info["tips_org_conf"] = tips_org_conf;

	if (file_set.find(filename) == file_set.end())
	{
		ofstream os;
		os.open(filename);

		Json::Value json_root;
		Json::Value hand_info;
		Json::Value joint_name;
		Json::Value joint_radius;

		for (std::string& name : tracker.joint_names_output_full)
		{
			auto center_id = tracker.worker->model->centers_name_to_id_map[name];
			float radius = tracker.worker->model->radii[center_id];

			joint_name.append(name);
			joint_radius.append(radius);
		}
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
				obj_motion.append(tracker.object_pose(i, j));
		}

		hand_info["joint_name"] = joint_name;
		hand_info["joint_radius"] = joint_radius;
		json_root["hand_info"] = hand_info;
		json_root["motion_info"].append(motion_info);

		os << sw.write(json_root);
		os.close();

		file_set.insert(filename);
	}
	else
	{
		std::fstream os;
		os.open(filename);

		os.seekp(-11, std::ios::end);
		os << ",\n" << sw.write(motion_info) << "   ]\n}\n";

		os.close();
	}
}