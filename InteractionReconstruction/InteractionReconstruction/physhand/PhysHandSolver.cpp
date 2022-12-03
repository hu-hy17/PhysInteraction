#include"PhysHandSolver.h"
#include"Defs.h"
#include"Utils.h"

#include<iostream>
#include<windows.h>

using std::cout;
using std::endl;

void PhysHandSolver::loadObjectMesh(const std::vector<Eigen::Vector3f>& vertices,
									const std::vector<Eigen::Vector3f>& normals)
{
	m_inertia_solver.loadPointCloud(vertices, normals);
	m_inertia_solver.solve();
	m_inertia_solver.m_inertia_tensor *= m_obj_mass;
	m_contact_points.setObjectInfo(&m_inertia_solver);

	m_set_obj_data = true;
}

bool PhysHandSolver::isReadyToSolve()
{
	return m_set_obj_data;
}

void PhysHandSolver::initForceSolver(bool nonrigid /* = false */)
{
	m_obj_mass = 0.2;
	m_friction_factor = 0.7;
	m_delta_t = 1.0 / 30;
	Eigen::Vector3d edge_len(0.058, 0.058, 0.058);
	Eigen::Vector3d gravity(0, -9.8, 0);
	m_force_solver.setPhysData(m_friction_factor, m_obj_mass, gravity, m_delta_t);

	double mass_offset = (0.2 / m_obj_mass) * (0.2 / m_obj_mass);
	double weight_force = 100.0 * mass_offset;
	double weight_moment = 2e6 * mass_offset;
	double weight_friction_cone = 1e6;
	double weight_val = 2 * mass_offset;
	double weight_contact_dist = 1e5;			// 5e5
	double weight_no_contact_no_force = 1e8 * mass_offset;	// 1e10
	m_force_solver.setWeightData(weight_force, weight_moment, weight_friction_cone, weight_val,
								 weight_contact_dist, weight_no_contact_no_force);

	m_is_sticky_tip.resize(5, false);
	m_last_tip_pos_in_obj_coord.resize(5);
	m_last_normal.resize(5);

	m_acc_conf.resize(5, 1.0);

	m_slide_to_target.resize(5, false);

	m_nonrigid = nonrigid;
}

std::vector<Eigen::Vector3f> PhysHandSolver::solve(const std::vector<Eigen::Vector3f>& contact_points,
												   const std::vector<Eigen::Vector3f>& contact_normals,
												   const std::vector<Eigen::Vector3f>& joint_pos,
												   const std::vector<float>& joint_radii,
												   const std::vector<double>& tips_conf,
												   const Eigen::Matrix4f object_motion)
{
	m_contact_points.reset(false);
	m_contact_points.loadData(contact_points, contact_normals, joint_pos, joint_radii, object_motion);
	m_contact_points.classify();

	m_force_solver.reset();

	Eigen::Vector3d cur_vel, next_vel, cur_ang_vel, next_ang_vel;
	Eigen::Matrix4d obj_motion_global = m_contact_points.m_obj_motion_d * m_inertia_solver.m_obj_to_cano_d;
	Eigen::Vector3d cur_pos = obj_motion_global.block(0, 3, 3, 1);
	m_force_solver.setCurPos(1e-3 * cur_pos);
	_calTarForceAndMoment_(m_contact_points.m_obj_motion * m_inertia_solver.m_obj_to_cano);

	std::vector<Eigen::Vector3d> vec_cp = std::vector<Eigen::Vector3d>();
	std::vector<Eigen::Vector3d> vec_norm = std::vector<Eigen::Vector3d>();
	std::vector<double> joint_dists = std::vector<double>();

	auto& cp_arr = m_contact_points.m_classified_contact_points;
	auto& norm_arr = m_contact_points.m_classified_contact_norms;
	auto& dist_arr = m_contact_points.m_joint_dists;

	//if (1)
	//{
	//	int icount = 0;
	//	for (auto d : dist_arr)
	//	{
	//		if (d <= CONTACT_CONTROL::g_min_potential_cp_dist) 
	//		{
	//			++icount;
	//		}
	//	}
	//	std::ofstream ofs("../../../../result/contact_num.txt", std::ios_base::app);
	//	ofs << icount << endl;
	//	ofs.close();
	//	std::vector<Eigen::Vector3f> ret;
	//	return ret;
	//}

	for (int i = 0; i < cp_arr.size(); i++)
	{
		vec_cp.push_back(
			1e-3 * Eigen::Vector3d(cp_arr[i](0), cp_arr[i](1), cp_arr[i](2))
		);
		vec_norm.push_back(
			Eigen::Vector3d(norm_arr[i](0), norm_arr[i](1), norm_arr[i](2))
		);
		joint_dists.push_back(1e-3 * dist_arr[i]);
	}

	m_force_solver.setContactData(vec_cp, vec_norm, joint_dists);

	m_force_solver.solve(false);

	for (int i = 0; i < 5; i++)
	{
		if (tips_conf[i] < m_acc_conf[i])
		{
			m_acc_conf[i] = CONF_CONTROL::g_conf_ratio_dec * m_acc_conf[i] + 
				(1 - CONF_CONTROL::g_conf_ratio_dec) * tips_conf[i];
		}
		else
		{
			m_acc_conf[i] = CONF_CONTROL::g_conf_ratio_inc * m_acc_conf[i] + 
				(1 - CONF_CONTROL::g_conf_ratio_inc) * tips_conf[i];
		}
	}

	std::vector<Eigen::Vector3f> ret = _getRefinedTipPos_();

	_setCpAttributes_();
	_updateStickyTips_();

	if (0)
	{
		int icount = 0;
		for (auto b : m_cp_is_final)
		{
			if (b)
				++icount;
		}
		std::ofstream ofs("../../../../result/contact_num.txt", std::ios_base::app);
		ofs << icount << endl;
		ofs.close();
	}

	return ret;
}

void PhysHandSolver::getPhysInfo(std::vector<Eigen::Vector3f>& contact_points,
								 std::vector<Eigen::Vector3f>& contact_forces,
								 std::vector<int>& contact_corr)
{
	contact_points = m_force_solver.getContactPoints();
	contact_forces = m_force_solver.getContactForces();
	contact_corr = m_contact_points.m_cp_joint_correspond;
	assert(contact_forces.size() == contact_points.size());
	assert(contact_forces.size() == contact_corr.size());
}

void PhysHandSolver::getPhysInfo(std::vector<Eigen::Vector3f>& contact_points,
	std::vector<Eigen::Vector3f>& contact_forces,
	std::vector<int>& contact_corr,
	Eigen::Vector3f& tar_force,
	Eigen::Vector3f& tar_moment,
	Eigen::Vector3f& obj_center,
	Eigen::Vector3f& obj_vel,
	Eigen::Vector3f& obj_ang_vel,
	Eigen::Matrix3f& obj_rot)
{
	getPhysInfo(contact_points, contact_forces, contact_corr);
	tar_force = m_force_solver.get_tar_force();
	tar_moment = m_force_solver.get_tar_moment();
	obj_center = m_force_solver.get_cur_pos();
	obj_vel = Eigen::Vector3f(m_vel.x(), m_vel.y(), -m_vel.z());
	obj_ang_vel = Eigen::Vector3f(m_ang_vel.x(), m_ang_vel.y(), -m_ang_vel.z());
	obj_rot = m_obj_rot;
	for (int c = 0; c < 3; ++c)
		obj_rot(2, c) = -obj_rot(2, c);
}

void PhysHandSolver::setTipsFinalPos(const std::vector<Eigen::Vector3f> tips_final_pos)
{
	Eigen::Matrix4f obj_to_global = m_contact_points.m_obj_motion * m_inertia_solver.m_obj_to_cano;
	for (int i = 0; i < 5; i++)
	{
		Eigen::Vector3f cur_tips_pos = (obj_to_global.inverse() *
										Eigen::Vector4f(tips_final_pos[i].x(), tips_final_pos[i].y(), -tips_final_pos[i].z(), 1.0f)).head(3);

		m_last_tip_pos_in_obj_coord[i] = cur_tips_pos;
	}
}

void PhysHandSolver::_calTarForceAndMoment_(const Eigen::Matrix4f& obj_motion)
{
	m_obj_rot = obj_motion.block(0, 0, 3, 3);
	if (m_obj_motions_buf.size() < 3) 
	{
		m_force_solver.setTarForceAndTorque(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));
		m_obj_motions_buf.push(obj_motion);
		return;
	}
	
	m_obj_motions_buf.pop();

	auto pre_motion = m_obj_motions_buf.back();
	auto pre_pre_motion = m_obj_motions_buf.front();

	Eigen::Vector3f vel = 1e-3 / m_delta_t * (obj_motion.block(0, 3, 3, 1) - pre_motion.block(0, 3, 3, 1));
	Eigen::Vector3f acc = 1e-3 / (m_delta_t * m_delta_t) * (obj_motion.block(0, 3, 3, 1) + pre_pre_motion.block(0, 3, 3, 1)
		- 2 * pre_motion.block(0, 3, 3, 1));

	Eigen::Vector3f vel_ang = getRotAngle(pre_motion.block(0, 0, 3, 3), obj_motion.block(0, 0, 3, 3)) / m_delta_t;
	Eigen::Vector3f vel_ang_pre = getRotAngle(pre_pre_motion.block(0, 0, 3, 3), pre_motion.block(0, 0, 3, 3)) / m_delta_t;
	Eigen::Vector3f acc_ang = (vel_ang - vel_ang_pre) / m_delta_t;

	if (m_smooth_buf.size() == 0)
	{
		m_smooth_buf.push_back(vel);
		m_smooth_buf.push_back(acc);
		m_smooth_buf.push_back(vel_ang);
		m_smooth_buf.push_back(acc_ang);
	}
	else
	{
		float s = m_movement_smooth_ratio;
		m_smooth_buf[0] = s * m_smooth_buf[0] + (1 - s) * vel;
		m_smooth_buf[1] = s * m_smooth_buf[1] + (1 - s) * acc;
		m_smooth_buf[2] = s * m_smooth_buf[2] + (1 - s) * vel_ang;
		m_smooth_buf[3] = s * m_smooth_buf[3] + (1 - s) * acc_ang;
		vel = m_smooth_buf[0];
		acc = m_smooth_buf[1];
		vel_ang = m_smooth_buf[2];
		acc_ang = m_smooth_buf[3];
	}
	
	m_vel = vel;
	m_ang_vel = vel_ang;

	Eigen::Vector3f tar_force = m_obj_mass * acc;

	Eigen::Matrix3f global_to_obj = obj_motion.block(0, 0, 3, 3).inverse();
	vel_ang = global_to_obj * vel_ang;
	acc_ang = global_to_obj * acc_ang;
	Eigen::Vector3f tar_moment = m_inertia_solver.m_inertia_tensor * acc_ang + vel_ang.cross(m_inertia_solver.m_inertia_tensor * vel_ang);

	Eigen::Matrix3f obj_to_global = obj_motion.block(0, 0, 3, 3);
	tar_moment = obj_to_global * tar_moment;

	cout << "Target Force: " << tar_force.norm() << "N\n";
	cout << "Target Moment: " << tar_moment.norm() << "Nm\n";

	m_force_solver.setTarForceAndTorque(
		Eigen::Vector3d(tar_force.x(), tar_force.y(), tar_force.z()),
		Eigen::Vector3d(tar_moment.x(), tar_moment.y(), tar_moment.z())
	);

	m_obj_motions_buf.push(obj_motion);
}

std::vector<Eigen::Vector3f> PhysHandSolver::_getRefinedTipPos_()
{
	std::vector<Eigen::Vector3f> tip_joint_pos(5);

	int tips_cp_idx[5];
	int contact_num = 0;
	for (int tip_idx = 0; tip_idx < 5; ++tip_idx)
	{
		int joint_idx = TIPS_JOINT_IDX[tip_idx];
		int cp_idx = -1;

		for (int i = 0; i < m_contact_points.m_cp_joint_correspond.size(); i++)
		{
			if (m_contact_points.m_cp_joint_correspond[i] == joint_idx)
			{
				cp_idx = i;
				if (tip_idx != 4) { ++contact_num; }
				break;
			}
		}

		tips_cp_idx[tip_idx] = cp_idx;
	}

	for (int tip_idx = 0; tip_idx < 5; ++tip_idx)
	{
		int joint_idx = TIPS_JOINT_IDX[tip_idx];
		int cp_idx = tips_cp_idx[tip_idx];

		if (cp_idx == -1)
		{
			tip_joint_pos[tip_idx] = m_contact_points.m_joint_pos[joint_idx];
		}
		else
		{
			if (m_contact_points.m_joint_dists[cp_idx] <= CONTACT_CONTROL::g_min_potential_cp_dist)
			{
				tip_joint_pos[tip_idx] = m_contact_points.m_joint_pos[joint_idx];
			}
			else
			{
				if (1e3 * m_force_solver.m_contact_result_dists[cp_idx] <= CONTACT_CONTROL::g_min_potential_cp_dist)
				{
					Eigen::Vector3f refine_pos = m_contact_points.m_classified_contact_points[cp_idx]
						- 0.85f * m_contact_points.m_classified_contact_norms[cp_idx] * m_contact_points.m_joint_radius[joint_idx];

					if (!g_use_conf_on_contact_status)
					{
						tip_joint_pos[tip_idx] = refine_pos;
					}
					else
					{
						float diff = (refine_pos - m_contact_points.m_joint_pos[joint_idx]).norm();
						float allow_slide_diff = INT_MAX;

						if (m_acc_conf[tip_idx] > CONF_CONTROL::g_allow_slide_conf_lower_bound)
						{
							allow_slide_diff = CONF_CONTROL::g_allow_slide_diff_lower_bound;
						}
						else
						{
							if (m_acc_conf[tip_idx] > 1e-5)
								allow_slide_diff = (CONF_CONTROL::g_allow_slide_conf_lower_bound * CONF_CONTROL::g_allow_slide_diff_lower_bound) / m_acc_conf[tip_idx];
						}
						tip_joint_pos[tip_idx] = (diff > allow_slide_diff) ? m_contact_points.m_joint_pos[joint_idx] : refine_pos;
					}
				}
				else
				{
					tip_joint_pos[tip_idx] = m_contact_points.m_joint_pos[joint_idx];
				}
			}
		}

		if (g_use_friction)
		{
			_getTipPosWithFriction_(tip_joint_pos[tip_idx], tip_idx, cp_idx, contact_num);
		}

		tip_joint_pos[tip_idx].z() = -tip_joint_pos[tip_idx].z();
	}

	return tip_joint_pos;
}

void PhysHandSolver::_getTipPosWithFriction_(Eigen::Vector3f& tar_tip_pos, const int tip_idx, const int cp_idx, const int contact_num /*=1*/)
{
	Eigen::Matrix4f obj_to_global = m_contact_points.m_obj_motion * m_inertia_solver.m_obj_to_cano;
	Eigen::Matrix4f global_to_obj = obj_to_global.inverse();

	if (!m_first_frame && m_is_sticky_tip[tip_idx])
	{
		double cf_val = m_force_solver.m_contact_forces[cp_idx].norm();		

		Eigen::Vector3f prev_tip_pos = m_last_tip_pos_in_obj_coord[tip_idx];	
		Eigen::Vector3f prev_tip_normal = m_last_normal[tip_idx];			
		Eigen::Vector3f cur_tip_pos = (global_to_obj * Eigen::Vector4f(tar_tip_pos.x(), tar_tip_pos.y(), tar_tip_pos.z(), 1.0)).head(3);	

		if (m_nonrigid)
		{
			Eigen::Vector3f deltax_normal = (prev_tip_normal.dot(cur_tip_pos - prev_tip_pos)) * prev_tip_normal;
			prev_tip_pos += deltax_normal;
		}

		Eigen::Vector4f tip_pos_global = m_contact_points.m_obj_motion * m_inertia_solver.m_obj_to_cano *
			Eigen::Vector4f(prev_tip_pos.x(), prev_tip_pos.y(), prev_tip_pos.z(), 1.0f);

		Eigen::Vector3f kinematic_pos = tar_tip_pos;
		Eigen::Vector3f physical_pos = tip_pos_global.head(3);
		float diff = (kinematic_pos - physical_pos).norm();

		if (!g_use_conf_on_friction)
		{
			tar_tip_pos = physical_pos;
			return;
		}

		float allow_slide_diff = INT_MAX;
		if (m_acc_conf[tip_idx] > CONF_CONTROL::g_allow_slide_conf_lower_bound)
		{
			allow_slide_diff = CONF_CONTROL::g_allow_slide_diff_lower_bound;
		}
		else
		{
			if (m_acc_conf[tip_idx] > 1e-5)
				allow_slide_diff = (CONF_CONTROL::g_allow_slide_conf_lower_bound * CONF_CONTROL::g_allow_slide_diff_lower_bound) / m_acc_conf[tip_idx];
		}

		if (tip_idx != 4) { allow_slide_diff /= contact_num; }
		std::cout << contact_num << std::endl;

		if (!m_slide_to_target[tip_idx])
		{
			if (diff > allow_slide_diff)
			{
				tar_tip_pos = CONF_CONTROL::g_slide_ratio * kinematic_pos + (1 - CONF_CONTROL::g_slide_ratio) * physical_pos;
				m_slide_to_target[tip_idx] = true;
			}
			else
			{
				tar_tip_pos = physical_pos;
			}
		}
		else
		{
			if (m_acc_conf[tip_idx] < CONF_CONTROL::g_allow_slide_conf_lower_bound)
			{
				tar_tip_pos = physical_pos;
				m_slide_to_target[tip_idx] = false;
			}
			else
			{
				if (diff < CONF_CONTROL::g_stop_slide_diff_upper_bound)
				{
					tar_tip_pos = kinematic_pos;
					m_slide_to_target[tip_idx] = false;
				}
				else
				{
					tar_tip_pos = CONF_CONTROL::g_slide_ratio * kinematic_pos + (1 - CONF_CONTROL::g_slide_ratio) * physical_pos;
				}
			}
		}
	}
}

void PhysHandSolver::_updateStickyTips_()
{
	Eigen::Matrix4f obj_to_global = m_contact_points.m_obj_motion * m_inertia_solver.m_obj_to_cano;

	m_last_frame_sticky_tips = m_is_sticky_tip;
	for (int i = 0; i < 5; i++)
	{
		m_is_sticky_tip[i] = false;
	}

	const auto& cp_joint_correspond = m_contact_points.m_cp_joint_correspond;
	int cp_num = cp_joint_correspond.size();

	for (int cp_idx = 0; cp_idx < cp_num; cp_idx++)
	{
		int joint_idx = cp_joint_correspond[cp_idx];
		int tip_idx = JOINT_IDX_TO_TIP[joint_idx];
		if (tip_idx == -1)
			continue;

		double cf_val = m_force_solver.m_contact_forces[cp_idx].norm();	
		if (cf_val > CONTACT_CONTROL::g_sticky_tip_force_min_rate * m_obj_mass * 9.8)
		{
			// printf("%d-%f\n", tip_idx, cf_val);
			m_is_sticky_tip[tip_idx] = true;
		}
	}

	for (int i = 0; i < 5; ++i) 
	{
		m_last_normal[i] = Eigen::Vector3f(0, 0, 0);
	}

	Eigen::Matrix4f global_to_object = (m_contact_points.m_obj_motion * m_inertia_solver.m_obj_to_cano).inverse();
	Eigen::Matrix3f global_to_object_rot = global_to_object.block(0, 0, 3, 3);

	for (int cp_idx = 0; cp_idx < cp_num; cp_idx++)
	{
		int joint_idx = cp_joint_correspond[cp_idx];
		int tip_idx = JOINT_IDX_TO_TIP[joint_idx];
		if (tip_idx == -1)
			continue;

		m_last_normal[tip_idx] = global_to_object_rot * m_contact_points.m_classified_contact_norms[cp_idx];
	}

	if (m_first_frame)
		m_first_frame = false;
}

void PhysHandSolver::_setCpAttributes_()
{
	const auto& org_dists = m_contact_points.m_joint_dists;
	const auto& min_dists = m_force_solver.m_contact_result_dists;
	assert(org_dists.size() == min_dists.size());

	int cp_num = org_dists.size();
	m_cp_is_final.resize(cp_num, false);
	m_cp_is_cadidate.resize(cp_num, false);

	for (int i = 0; i < cp_num; i++)
	{
		m_cp_is_cadidate[i] = org_dists[i] > CONTACT_CONTROL::g_min_potential_cp_dist;
		m_cp_is_final[i] = 1e3 * min_dists[i] <= CONTACT_CONTROL::g_min_potential_cp_dist;
	}
}

