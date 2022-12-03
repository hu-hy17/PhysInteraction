#pragma once

#include"InertiaSolver.h"
#include"ForceSolver.h"
#include"ContactPoints.h"

#include<queue>

class PhysHandSolver
{
private:
	const std::string m_class_name = "(PhysHandSolver)";
	bool m_set_obj_data = false;
	bool m_first_frame = true;
	bool m_nonrigid = false;

	double m_obj_mass;				
	double m_friction_factor;		
	double m_delta_t;				

	Eigen::Vector3f m_vel;
	Eigen::Vector3f m_ang_vel;
	Eigen::Matrix3f m_obj_rot;

	std::vector<Eigen::Vector3f> m_smooth_buf;			
	const float m_movement_smooth_ratio = 0.8f;			

	std::queue<Eigen::Matrix4f> m_obj_motions_buf;		

	std::vector<bool> m_is_sticky_tip;							
	std::vector<bool> m_last_frame_sticky_tips;					
	std::vector<Eigen::Vector3f> m_last_tip_pos_in_obj_coord;	
	std::vector<Eigen::Vector3f> m_last_normal;					

	std::vector<bool> m_cp_is_cadidate;				
	std::vector<bool> m_cp_is_final;				

	std::vector<double> m_acc_conf;							
	std::vector<bool> m_slide_to_target;	

public:
	InertiaSolver m_inertia_solver;
	ForceSolver m_force_solver;
	ContactPoints m_contact_points;
	std::vector<double> m_result_dists;

public:
	void loadObjectMesh(const std::vector<Eigen::Vector3f>& vertices,
						const std::vector<Eigen::Vector3f>& normals);

	void initForceSolver(bool nonrigid = false);

	bool isReadyToSolve();

	std::vector<Eigen::Vector3f> solve(const std::vector<Eigen::Vector3f>& contact_points,
									   const std::vector<Eigen::Vector3f>& contact_normals,
									   const std::vector<Eigen::Vector3f>& joint_pos,
									   const std::vector<float>& joint_radii,
									   const std::vector<double>& tips_conf,
									   const Eigen::Matrix4f object_motion);

	void getPhysInfo(std::vector<Eigen::Vector3f>& contact_points,
					 std::vector<Eigen::Vector3f>& contact_forces,
					 std::vector<int>& contact_corr);

	void getPhysInfo(std::vector<Eigen::Vector3f>& contact_points,
					 std::vector<Eigen::Vector3f>& contact_forces,
					 std::vector<int>& contact_corr,
					 Eigen::Vector3f& tar_force,
					 Eigen::Vector3f& tar_moment,
				     Eigen::Vector3f& obj_center,
				     Eigen::Vector3f& obj_vel,
					 Eigen::Vector3f& obj_ang_vel,
					 Eigen::Matrix3f& obj_rot);

	void setTipsFinalPos(const std::vector<Eigen::Vector3f> tips_final_pos);

private:
	void _calTarForceAndMoment_(const Eigen::Matrix4f& obj_motion);
	std::vector<Eigen::Vector3f> _getRefinedTipPos_();
	void _setCpAttributes_();
	void _updateStickyTips_();
	void _getTipPosWithFriction_(Eigen::Vector3f& tip_pos, const int tip_idx, const int cp_idx, const int contact_num = 1);
};