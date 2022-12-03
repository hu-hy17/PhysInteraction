#pragma once

#include<Eigen/Dense>
#include<string>

void getOrthognalBase(const Eigen::Vector3d& norm_vec,
					  Eigen::Vector3d& ret_vec1,
					  Eigen::Vector3d& ret_vec2);

Eigen::Vector3f getRotAngle(const Eigen::Matrix3f& from_rot_mat, const Eigen::Matrix3f& to_rot_mat);

Eigen::Matrix3f getRotMat(const Eigen::Vector3f& from_vec, const Eigen::Vector3f& to_vec);

Eigen::Quaternionf quatSlerp(Eigen::Quaternionf &start_q, Eigen::Quaternionf &end_q, float t);