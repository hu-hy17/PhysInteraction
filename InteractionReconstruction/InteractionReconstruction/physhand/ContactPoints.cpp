#include"ContactPoints.h"
#include"Defs.h"

#include<iostream>
#include<fstream>
#include<set>

#include<igl/signed_distance.h>

using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;

ContactPoints::ContactPoints()
{
	m_all_contact_points = std::vector<Eigen::Vector3f>();
	m_all_contact_norms = std::vector<Eigen::Vector3f>();

	m_classified_contact_points = std::vector<Eigen::Vector3f>();
	m_classified_contact_norms = std::vector<Eigen::Vector3f>();

	m_joint_pos = std::vector<Eigen::Vector3f>();
	m_joint_radius = std::vector<float>();
}

void ContactPoints::setObjectInfo(const InertiaSolver* p_obj_info)
{
	if (m_object_info)
	{
		cout << m_class_name << "Warning : existing object info will be overwritten!" << endl;
		m_object_info = nullptr;
	}
	m_object_info = p_obj_info;
}

void ContactPoints::reset(bool reset_obj_info /* = false */)
{
	m_all_contact_points.clear();
	m_all_contact_norms.clear();
	m_cp_joint_correspond.clear();
	m_classified_contact_points.clear();
	m_classified_contact_norms.clear();
	m_joint_pos.clear();
	m_joint_radius.clear();
	m_joint_dists.clear();

	m_all_contact_points.swap(std::vector<Eigen::Vector3f>());
	m_all_contact_norms.swap(std::vector<Eigen::Vector3f>());
	m_cp_joint_correspond.swap(std::vector<int>());
	m_classified_contact_points.swap(std::vector<Eigen::Vector3f>());
	m_classified_contact_norms.swap(std::vector<Eigen::Vector3f>());
	m_joint_pos.swap(std::vector<Eigen::Vector3f>());
	m_joint_radius.swap(std::vector<float>());
	m_joint_dists.swap(std::vector<double>());

	m_set_data = false;

	if (reset_obj_info)
		m_object_info = nullptr;
}

void ContactPoints::loadData(const std::vector<Eigen::Vector3f>& contact_points,
							 const std::vector<Eigen::Vector3f>& contact_normals,
							 const std::vector<Eigen::Vector3f>& joint_pos,
							 const std::vector<float>& joint_radii,
							 const Eigen::Matrix4f& object_motion)
{
	m_obj_motion = object_motion;
	for (int r = 1; r < 3; r++)
	{
		for (int c = 0; c < 4; c++)
			m_obj_motion(r, c) = -m_obj_motion(r, c);
	}
	for (int r = 0; r < 4; r++)
	{
		for (int c = 0; c < 4; c++)
			m_obj_motion_d(r, c) = m_obj_motion(r, c);
	}

	if (joint_pos.size() != joint_radii.size())
	{
		cout << m_class_name << "Error : Joints position number does not match joints radius number!" << endl;
		goto fail;
	}

	if (contact_points.size() != contact_normals.size())
	{
		cout << m_class_name << "Error : Contact points number does not match contact normals number!" << endl;
		goto fail;
	}

	int keypoint_num = joint_pos.size();
	m_joint_pos = joint_pos;
	for (int i = 0; i < keypoint_num; i++)
	{
		m_joint_pos[i].z() = -m_joint_pos[i].z();
	}
	m_joint_radius = joint_radii;

	int contact_point_num = contact_points.size();
	for (int i = 0; i < contact_point_num; i++)
	{
		float px = contact_points[i].x();
		float py = contact_points[i].y();
		float pz = contact_points[i].z();

		Eigen::Vector4f tmp_contact_point(px, py, pz, 1.0);
		tmp_contact_point = m_obj_motion * tmp_contact_point;
		Eigen::Vector3f tmp_contact_norm = m_obj_motion.block(0, 0, 3, 3) * contact_normals[i];
		tmp_contact_norm.normalize();

		m_all_contact_points.push_back(Eigen::Vector3f(tmp_contact_point(0), tmp_contact_point(1), tmp_contact_point(2)));

		m_all_contact_norms.push_back(-Eigen::Vector3f(tmp_contact_norm(0), tmp_contact_norm(1), tmp_contact_norm(2)));
	}

	m_set_data = true;

	return;

fail:
	system("pause");
	exit(-1);
}

void ContactPoints::classify()
{
	if (!m_set_data)
	{
		cerr << m_class_name << "Error: You have not set enough data for solution!" << endl;
		system("pause");
		exit(-1);
	}

	int joint_num = m_joint_pos.size();
	int contact_point_num = m_all_contact_points.size();
	std::vector<int> classification = std::vector<int>(joint_num, 0);
	std::vector<Eigen::Vector3f> pos_sum = std::vector<Eigen::Vector3f>(joint_num);
	std::vector<Eigen::Vector3f> norm_sum = std::vector<Eigen::Vector3f>(joint_num);
	std::vector<int> cp_class(contact_point_num, -1);	
	std::set<int> contact_joints;	

	for (int i = 0; i < joint_num; i++)
	{
		pos_sum[i] = Eigen::Vector3f::Zero();
		norm_sum[i] = Eigen::Vector3f::Zero();
	}

	for (int i = 0; i < contact_point_num; i++)
	{
		const Eigen::Vector3f& tmp_contact_point = m_all_contact_points[i];

		for (int j = 0; j < joint_num; j++)
		{
			float dist = (tmp_contact_point - m_joint_pos[j]).norm();
			if (dist <= CONTACT_CONTROL::g_radius_expand_rate * m_joint_radius[j])
			{
				++classification[j];
				cp_class[i] = j;
				pos_sum[j] += tmp_contact_point;
				norm_sum[j] += m_all_contact_norms[i];
				break;
			}
		}
	}

	auto isTips = [&](int joint_idx) {
		for (int i = 0; i < 5; i++)
		{
			if (joint_idx == TIPS_JOINT_IDX[i])
				return true;
		}
		return false;
	};

	for (int i = 0; i < joint_num; i++)
	{
		if (classification[i] < CONTACT_CONTROL::g_contact_point_lower_bound)
			continue;

		if (!isTips(i))
			continue;

		contact_joints.insert(i);
		m_classified_contact_points.push_back(pos_sum[i] / classification[i]);
		norm_sum[i] /= classification[i];
		norm_sum[i].normalize();
		m_classified_contact_norms.push_back(norm_sum[i]);
		m_cp_joint_correspond.push_back(i);
	}

	if (!m_object_info)
	{
		cout << m_class_name << "Warning : object info is not set!" << endl;
		return;
	}
	
	Eigen::Matrix4f obj_to_global;

	obj_to_global = m_obj_motion;

	m_joint_dists.resize(m_classified_contact_points.size(), 0);

	std::vector<int> potential_cp_joint_idx;
	for (int i = 0; i < 5; i++)
	{
		if (contact_joints.find(TIPS_JOINT_IDX[i]) == contact_joints.end())
		{
			potential_cp_joint_idx.push_back(TIPS_JOINT_IDX[i]);
		}
	}
	_addPotentialCp_(potential_cp_joint_idx, obj_to_global);
}

void ContactPoints::_addPotentialCp_(const std::vector<int>& joint_idx_arr, Eigen::Matrix4f obj_to_global)
{
	if (joint_idx_arr.size() <= 0)
		return;

	Eigen::Matrix4f global_to_obj = obj_to_global.inverse();

	Eigen::MatrixXf query_points(joint_idx_arr.size(), 3);

	Eigen::VectorXf result_dist;
	Eigen::VectorXi element_idx;
	Eigen::MatrixXf closest_point;

	for (int i = 0; i < joint_idx_arr.size(); i++)
	{
		int joint_idx = joint_idx_arr[i];
		assert(joint_idx < TOTAL_JOINT_NUM);
		Eigen::Vector3f qPoint = (global_to_obj * 
								  Eigen::Vector4f(m_joint_pos[joint_idx](0), m_joint_pos[joint_idx](1), m_joint_pos[joint_idx](2), 1.0f)
								  ).head<3>();
		query_points.block(i, 0, 1, 3) = qPoint.transpose();
	}

	m_object_info->m_obj_tree->squared_distance(m_object_info->m_vertices, m_object_info->m_indices, query_points,
												result_dist, element_idx, closest_point);

	for (int i = 0; i < joint_idx_arr.size(); i++)
	{
		int joint_idx = joint_idx_arr[i];
		float dist = sqrtf(result_dist(i));
		// cout << JOINT_NAME[joint_idx] << " distance : " << dist << endl;

		if (dist - m_joint_radius[joint_idx] > CONTACT_CONTROL::g_max_potential_cp_dist)
			continue;

		int eIdx = element_idx(i);
		Eigen::Vector3f potential_cp = (obj_to_global * Eigen::Vector4f(closest_point(i, 0), closest_point(i, 1), closest_point(i, 2), 1.0f)).head<3>();
		Eigen::Vector3f potential_norm = obj_to_global.block(0, 0, 3, 3) * m_object_info->getTriangleNormal(eIdx);
		m_classified_contact_points.push_back(potential_cp);
		m_classified_contact_norms.push_back(-potential_norm);
		m_joint_dists.push_back(dist - m_joint_radius[joint_idx]);

		m_cp_joint_correspond.push_back(joint_idx);
	}

	// cout << endl;
}

void ContactPoints::_jointBallInterpolation_(int add_num)
{
	float ratio = 1.0 / (add_num + 1);
	int joint_num = m_joint_pos.size();
	for (int i = 0; i < joint_num; i++)
	{
		int child_joint_idx = CHILD_JOINT[i];
		if (child_joint_idx == -1)
			continue;

		float parent_rate = 1.0;
		float child_rate = 0;
		for (int _ = 0; _ < add_num; _++)
		{
			parent_rate -= ratio;
			child_rate += ratio;
			Eigen::Vector3f new_joint_ball_pos = parent_rate * m_joint_pos[i] + child_rate * m_joint_pos[child_joint_idx];
			float new_joint_ball_radius = parent_rate * m_joint_radius[i] + child_rate * m_joint_radius[child_joint_idx];

			m_joint_pos.push_back(new_joint_ball_pos);
			m_joint_radius.push_back(new_joint_ball_radius);
		}
	}
}
