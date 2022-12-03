#include"Utils.h"
#include"ContactPoints.h"
#include"InertiaSolver.h"

#include<iostream>
#include<fstream>
#include<cmath>

using std::cout;
using std::endl;

void getOrthognalBase(const Eigen::Vector3d& norm_vec,
					  Eigen::Vector3d& ret_vec1,
					  Eigen::Vector3d& ret_vec2)
{
	const double zero_limit = 1e-4;		

	int choose_idx = -1;
	for (int i = 0; i < 3; i++)
	{
		if (abs(norm_vec(i)) > zero_limit)
		{
			choose_idx = i;
			break;
		}
	}

	assert(choose_idx != -1);

	double deno = 0;
	for (int i = 0; i < 3; i++)
	{
		if (i != choose_idx)
		{
			deno -= norm_vec(i);
			ret_vec1(i) = 1;
		}
	}
	ret_vec1(choose_idx) = deno / norm_vec(choose_idx);

	ret_vec2 = norm_vec.cross(ret_vec1);

	ret_vec1.normalize();
	ret_vec2.normalize();
}

Eigen::Vector3f getRotAngle(const Eigen::Matrix3f& from_rot_mat, const Eigen::Matrix3f& to_rot_mat)
{

	Eigen::Matrix3f tmp = from_rot_mat.inverse();
	Eigen::Matrix3f rot_mat = to_rot_mat * tmp;
	Eigen::AngleAxisf rot_vec;
	rot_vec.fromRotationMatrix(rot_mat);
	Eigen::Vector3f ret = rot_vec.angle() * rot_vec.axis();
	return ret;

	//Eigen::Matrix3f om = (to_rot_mat - from_rot_mat) * to_rot_mat.transpose();
	//return Eigen::Vector3f(om(2, 1), om(0, 2), om(1, 0));
}


Eigen::Matrix3f getRotMat(const Eigen::Vector3f& from_vec, const Eigen::Vector3f& to_vec)
{
	Eigen::Quaternionf quat;
	return quat.setFromTwoVectors(from_vec, to_vec).toRotationMatrix();
}

Eigen::Quaternionf quatSlerp(Eigen::Quaternionf &start_q, Eigen::Quaternionf &end_q, float t)
{
	Eigen::Quaternionf lerp_q;

	float cos_angle = start_q.x() * end_q.x()
		+ start_q.y() * end_q.y()
		+ start_q.z() * end_q.z()
		+ start_q.w() * end_q.w();

	// If the dot product is negative, the quaternions have opposite handed-ness and slerp won't take
	// the shorter path. Fix by reversing one quaternion.
	if (cos_angle < 0) {
		end_q.x() = -end_q.x();
		end_q.y() = -end_q.y();
		end_q.z() = -end_q.z();
		end_q.w() = -end_q.w();
		cos_angle = -cos_angle;
	}

	float ratio_A, ratio_B;
	//If the inputs are too close for comfort, linearly interpolate
	if (cos_angle > 0.99995f) {
		ratio_A = 1.0f - t;
		ratio_B = t;
	}
	else {
		float sin_angle = sqrt(1.0f - cos_angle * cos_angle);
		float angle = atan2(sin_angle, cos_angle);
		ratio_A = sin((1.0f - t) * angle) / sin_angle;
		ratio_B = sin(t * angle) / sin_angle;
	}

	lerp_q.x() = ratio_A * start_q.x() + ratio_B * end_q.x();
	lerp_q.y() = ratio_A * start_q.y() + ratio_B * end_q.y();
	lerp_q.z() = ratio_A * start_q.z() + ratio_B * end_q.z();
	lerp_q.w() = ratio_A * start_q.w() + ratio_B * end_q.w();

	return lerp_q.normalized();
}
