#pragma once

#include<vector>
#include<string>
#include<memory>

#include<Eigen/Dense>
#include<Eigen/Core>

#include"InertiaSolver.h"

class ContactPoints
{
private:
	std::string m_class_name = "(ContactPoints)";

	bool m_set_data = false;								

	const InertiaSolver* m_object_info = nullptr;			

public:
	std::vector<Eigen::Vector3f> m_all_contact_points;		
	std::vector<Eigen::Vector3f> m_all_contact_norms;		

	std::vector<Eigen::Vector3f> m_classified_contact_points;	
	std::vector<Eigen::Vector3f> m_classified_contact_norms;	

	std::vector<int> m_cp_joint_correspond;	

	std::vector<double> m_joint_dists;		

	Eigen::Matrix4f m_obj_motion;					
	Eigen::Matrix4d m_obj_motion_d;					

	std::vector<Eigen::Vector3f> m_joint_pos;		
	std::vector<float> m_joint_radius;				

private:

	void _jointBallInterpolation_(int add_num = 1);

	void _addPotentialCp_(const std::vector<int>& joint_idx, Eigen::Matrix4f obj_to_global);
	
public:
	ContactPoints();

	void loadData(const std::vector<Eigen::Vector3f>& contact_points,
				  const std::vector<Eigen::Vector3f>& contact_normals,
				  const std::vector<Eigen::Vector3f>& joint_pos,
				  const std::vector<float>& joint_radii,
				  const Eigen::Matrix4f& object_motion);

	void setObjectInfo(const InertiaSolver* p_obj_info);

	void reset(bool reset_obj_info = false);

	void classify();
};