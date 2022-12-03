#pragma once
#include <QOpenGLWidget>
#include "tracker/ForwardDeclarations.h"
#include "tracker/OpenGL/KinectDataRenderer/KinectDataRenderer.h"
#include "tracker/OpenGL/ConvolutionRenderer/ConvolutionRenderer.h"
#include "tracker/OpenGL/MarkerRender/MarkerRenderer.h"
#include "CommonVariances.h"

#include "SphereMesh2Mano/ModelConvertor.h"

class GLWidget : public QGLWidget, public QOpenGLFunctions {
public:
	Worker * worker;
	DataStream * const datastream;
	SolutionStream * const solutions;

	Camera*const _camera;
	KinectDataRenderer kinect_renderer;
	ConvolutionRenderer convolution_renderer;
	std::unique_ptr<GLArrow> arrow_render;
	std::unique_ptr<GLMesh> mano_render;
	ModelConvertor model_convertor;

	bool playback;
	bool real_color;
	bool use_mano;

	std::string data_path;

	bool show_render = false;
	bool store_render = false;
	std::string render_store_path = "D:/Project/HandReconstruction/hmodel-master_vs2015_MultiCamera_s/result/render/";

	//ZH
	GLuint Render_PBO;
	GLuint Render_texture;
	GLuint Show_texture;
	int frame_id;

	Eigen::Matrix4f pose_camera0;
	Eigen::Matrix4f pose_camera1;
	Eigen::Matrix4f invpose_camera0;
	Eigen::Matrix4f invpose_camera1;
	Eigen::Matrix4f view_camera0;
	Eigen::Matrix4f view_camera1;
	Eigen::Matrix4f object_motion;

	Eigen::Matrix4f projection_matrix_c0;
	Eigen::Matrix4f projection_matrix_c1;

	LARGE_INTEGER time_stmp;
	double count_freq, count_interv, time_inter;

public:

	GLWidget(Worker* worker, DataStream * datastream, SolutionStream * solutions, bool playback, bool real_color, bool use_mano, std::string data_path);

	~GLWidget();

	void set_store_file(bool is_show, bool is_store, std::string store_path);

	void initialProjectionMatrix(int _width, int _height, CamerasParameters camera_par);

	void initializeGL();

	void paintGL();

	void paintGLCtrl_Skeleton();

	void paintPredictedHand();

	void render_texture_rigid();

	void render_texture_nonrigid();

	void set_object_motion(Eigen::Matrix4f object_mot);

	void reinitView();

private:
	Eigen::Vector3f camera_center = Eigen::Vector3f(0, 0, 0);
	Eigen::Vector3f image_center = Eigen::Vector3f(0, 0, 400);
	Eigen::Vector3f camera_up = Eigen::Vector3f(0, 1, 0);
	Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

	bool mouse_button_pressed = true;
	Eigen::Vector2f cursor_position = Eigen::Vector2f(640, 480);
	Eigen::Vector2f euler_angles = Eigen::Vector2f(-6.411, -1.8);
	Eigen::Vector2f initial_euler_angles = Eigen::Vector2f(-6.411, -1.8);
	float cursor_sensitivity = 0.003f;

	void process_mouse_movement(GLfloat cursor_x, GLfloat cursor_y);

	void process_mouse_button_pressed(GLfloat cursor_x, GLfloat cursor_y);

	void process_mouse_button_released();
	
	void mouseMoveEvent(QMouseEvent *event);

	void mousePressEvent(QMouseEvent *event);

	void wheelEvent(QWheelEvent * event);

	void keyPressEvent(QKeyEvent *event);
};
