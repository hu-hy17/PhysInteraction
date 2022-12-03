#include "arcball.h"
#include <iostream>

void arcball::init(float _fx, float _fy, float _cx, float _cy,
				   int _xoff, int _yoff, int _width, int _height,
				   float _dist)
{
	m_fx = _fx;
	m_fy = _fy;
	m_cx = _cx;
	m_cy = _cy;
	m_dist = _dist;
	m_xoff = _xoff;
	m_yoff = _yoff;
	m_width = _width;
	m_height = _height;
}

void arcball::set_prev_view_rot_pos(float xpos, float ypos)
{
	track_mode = VIEW_ROT;
	float radius = 0.309f*std::min(m_width, m_height);
	float x = (xpos - m_xoff - 0.5f*m_width) / radius;
	float y = (ypos - m_yoff - 0.5f*m_height) / radius;
	float z = 0.0f;

	if (x*x + y*y > 0.5f)
	{
		z = -0.5f / std::sqrt(x*x + y*y);
	}
	else
	{
		z = -std::sqrt(1.0f - x*x - y*y);
	}

	m_prev_view_rot_pos = Eigen::Vector3f(x, y, z);
}

void arcball::set_curr_view_rot_pos(float xpos, float ypos)
{
	float radius = 0.309f*std::min(m_width, m_height);
	float x = (xpos - m_xoff - 0.5f*m_width) / radius;
	float y = (ypos - m_yoff - 0.5f*m_height) / radius;
	float z = 0.0f;

	if (x*x + y*y > 0.5f)
	{
		z = -0.5f / std::sqrt(x*x + y*y);
	}
	else
	{
		z = -std::sqrt(1.0f - x*x - y*y);
	}

	m_curr_view_rot_pos = Eigen::Vector3f(x, y, z);
}

void arcball::set_prev_trans_pos(float xwin, float ywin)
{
	track_mode = TRANS;
	float depth = m_dist+m_scroll;
	float xc = depth*(xwin-m_xoff-m_cx)/m_fx;
	float yc = depth*(ywin-m_yoff-m_cy)/m_fy;

	m_prev_trans_pos = Eigen::Vector3f(xc, yc, depth);
}

void arcball::set_curr_trans_pos(float xwin, float ywin)
{
	float depth = m_dist+m_scroll;
	float xc = depth*(xwin-m_xoff-m_cx)/m_fx;
	float yc = depth*(ywin-m_yoff-m_cy)/m_fy;

	m_curr_trans_pos = Eigen::Vector3f(xc, yc, depth);
}

Eigen::Matrix3f arcball::get_view_rot(void)
{
	//	Calculate rotation matrix
	Eigen::Vector3f v0 = m_prev_view_rot_pos.normalized();
	Eigen::Vector3f v1 = m_curr_view_rot_pos.normalized();
	Eigen::Quaternionf delta_quaternion;
	delta_quaternion.setFromTwoVectors(v0, v1);


	//	Update last position
	m_prev_view_rot_pos = m_curr_view_rot_pos;

	return delta_quaternion.matrix();
}

Eigen::Vector3f arcball::get_view_trans(void)
{
	//	Calculate translation vector
	m_translation += m_curr_trans_pos - m_prev_trans_pos;

	//	Update last position
	m_prev_trans_pos = m_curr_trans_pos;

	//	Update view_matrix
	Eigen::Vector3f view_trans = m_translation;
	view_trans(2) = m_scroll;
	return view_trans;
}

void arcball::set_prev_light_rot_pos(float xpos, float ypos)
{
	track_mode = LIGHT_ROT;
	float radius = 0.309f*std::min(m_width, m_height);
	float x = (xpos - m_xoff - 0.5f*m_width) / radius;
	float y = (ypos - m_yoff - 0.5f*m_height) / radius;
	float z = 0.0f;

	if (x*x+y*y > 0.5f)
	{
		z = 0.5f / std::sqrt(x*x + y*y);
	}
	else
	{
		z = std::sqrt(1.0f - x*x - y*y);
	}

	m_prev_light_rot_pos = Eigen::Vector3f(x, y, z);
}

void arcball::set_curr_light_rot_pos(float xpos, float ypos)
{
	float radius = 0.309f*std::min(m_width, m_height);
	float x = (xpos - m_xoff - 0.5f*m_width) / radius;
	float y = (ypos - m_yoff - 0.5f*m_height) / radius;
	float z = 0.0f;

	if (x*x + y*y > 0.5f)
	{
		z = 0.5f / std::sqrt(x*x + y*y);
	}
	else
	{
		z = std::sqrt(1.0f - x*x - y*y);
	}

	m_curr_light_rot_pos = Eigen::Vector3f(x, y, z);
}

Eigen::Matrix4f arcball::get_light_matrix(void)
{
	//	Calculate rotation matrix
	Eigen::Vector3f v0 = m_prev_light_rot_pos.normalized();
	Eigen::Vector3f v1 = m_curr_light_rot_pos.normalized();
	Eigen::Quaternionf delta_quaternion;
	delta_quaternion.setFromTwoVectors(v0, v1);

	//	Update last position
	m_prev_light_rot_pos = m_curr_light_rot_pos;

	//	Update light rotation matrix
	Eigen::Matrix4f light_rot_update = Eigen::Matrix4f::Identity();
	light_rot_update.topLeftCorner(3, 3) = delta_quaternion.matrix();
	return light_rot_update;
}

void arcball::set_prev_model_rot_pos(float xpos, float ypos)
{
	track_mode = MODEL_ROT;
	float radius = 0.309f*std::min(m_width, m_height);
	float x = (xpos - m_xoff - 0.5f*m_width) / radius;
	float y = (ypos - m_yoff - 0.5f*m_height) / radius;
	float z = 0.0f;

	if (x*x + y*y > 0.5f)
	{
		z = -0.5f / std::sqrt(x*x + y*y);
	}
	else
	{
		z = -std::sqrt(1.0f - x*x - y*y);
	}

	m_prev_model_rot_pos = Eigen::Vector3f(x, y, z);
}

void arcball::set_curr_model_rot_pos(float xpos, float ypos)
{
	float radius = 0.309f*std::min(m_width, m_height);
	float x = (xpos - m_xoff - 0.5f*m_width) / radius;
	float y = (ypos - m_yoff - 0.5f*m_height) / radius;
	float z = 0.0f;

	if (x*x + y*y > 0.5f)
	{
		z = -0.5f / std::sqrt(x*x + y*y);
	}
	else
	{
		z = -std::sqrt(1.0f - x*x - y*y);
	}

	m_curr_model_rot_pos = Eigen::Vector3f(x, y, z);
}

Eigen::Matrix4f arcball::get_model_matrix(void)
{
	//	Calculate rotation matrix
	Eigen::Vector3f v0 = m_prev_model_rot_pos.normalized();
	Eigen::Vector3f v1 = m_curr_model_rot_pos.normalized();
	Eigen::Quaternionf delta_quaternion;
	delta_quaternion.setFromTwoVectors(v0, v1);

	//	Update last position
	m_prev_model_rot_pos = m_curr_model_rot_pos;

	//	Update model rotation matrix
	Eigen::Matrix4f model_rot_update = Eigen::Matrix4f::Identity();
	model_rot_update.topLeftCorner(3, 3) = delta_quaternion.matrix();
	return model_rot_update;
}