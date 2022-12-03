#ifndef _ARCBALL_H_
#define _ARCBALL_H_

#include <Eigen/Eigen>

class arcball
{
public:
	arcball() {}
	~arcball() {}

	void init(float _fx, float _fy, float _cx, float _cy,
			  int _xoff, int _yoff, int _width, int _height,
			  float _dist);

	void set_prev_view_rot_pos(float xpos, float ypos);
	void set_curr_view_rot_pos(float xpos, float ypos);
	void set_prev_trans_pos(float xwin, float ywin);
	void set_curr_trans_pos(float xwin, float ywin);
	void set_prev_light_rot_pos(float xpox, float ypos);
	void set_curr_light_rot_pos(float xpos, float ypos);
	void set_prev_model_rot_pos(float xpos, float ypos);
	void set_curr_model_rot_pos(float xpos, float ypos);
	void resize(int _new_xoff, int _new_yoff, int _new_width, int _new_height);
	void reset(void);
	void reset_trans_pos(void);
	void reset_view_rot_pos(void);
	void reset_light_rot_pos(void);
	void reset_model_rot_pos(void);
	void zoomin(void) { m_scroll -= 0.05f; }
	void zoomout(void) { m_scroll += 0.05f; }
	Eigen::Matrix3f get_view_rot();
	Eigen::Vector3f get_view_trans();
	Eigen::Matrix4f get_light_matrix();
	Eigen::Matrix4f get_model_matrix();

	enum { NONE, VIEW_ROT, TRANS, LIGHT_ROT, MODEL_ROT } track_mode = NONE;

private:
	float m_fx = 0;
	float m_fy = 0;
	float m_cx = 0;
	float m_cy = 0;
	int m_xoff = 0;
	int m_yoff = 0;
	int m_width = 0;
	int m_height = 0;
	float m_scroll = 0;
	float m_dist = 1.0f;
	Eigen::Vector3f m_prev_view_rot_pos = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
	Eigen::Vector3f m_curr_view_rot_pos = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
	Eigen::Vector3f m_prev_trans_pos = Eigen::Vector3f::Zero();
	Eigen::Vector3f m_curr_trans_pos = Eigen::Vector3f::Zero();
	Eigen::Vector3f m_prev_light_rot_pos = Eigen::Vector3f::Zero();
	Eigen::Vector3f m_curr_light_rot_pos = Eigen::Vector3f::Zero();
	Eigen::Vector3f m_prev_model_rot_pos = Eigen::Vector3f::Zero();
	Eigen::Vector3f m_curr_model_rot_pos = Eigen::Vector3f::Zero();
	Eigen::Quaternionf m_view_quaternion = Eigen::Quaternionf::Identity();
	Eigen::Vector3f m_translation = Eigen::Vector3f::Zero();
};

inline void arcball::resize(int _new_xoff, int _new_yoff, int _new_width, int _new_height)
{
	m_xoff = _new_xoff;
	m_yoff = _new_yoff;
	m_width = _new_width;
	m_height = _new_height;
}

inline void arcball::reset(void)
{
	reset_trans_pos();
	reset_view_rot_pos();
	reset_light_rot_pos();
	reset_model_rot_pos();
	m_scroll = 0;
	m_translation = Eigen::Vector3f::Zero();
	m_view_quaternion = Eigen::Quaternionf::Identity();
}

inline void arcball::reset_trans_pos(void)
{
	track_mode = NONE;
	m_prev_trans_pos = Eigen::Vector3f::Zero();
	m_curr_trans_pos = Eigen::Vector3f::Zero();
}

inline void arcball::reset_view_rot_pos(void)
{
	track_mode = NONE;
	m_prev_view_rot_pos = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
	m_curr_view_rot_pos = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
}

inline void arcball::reset_light_rot_pos(void)
{
	track_mode = NONE;
	m_prev_light_rot_pos = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
	m_curr_light_rot_pos = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
}
inline void arcball::reset_model_rot_pos(void)
{
	track_mode = NONE;
	m_prev_model_rot_pos = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
	m_curr_model_rot_pos = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
}

#endif