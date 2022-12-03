
#include "util/gl_wrapper.h"
#include "util/OpenGL32Format.h" 

#include "TwSettings.h"
#include "GLWidget.h"
#include "tracker/Data/Camera.h"
#include "tracker/Data/DataStream.h"
#include "tracker/Worker.h"
#include "tracker/Data/SolutionStream.h"
#include "tracker/Data/TextureColor8UC3.h"
#include "tracker/Data/TextureDepth16UC1.h"
#include "tracker/HandFinder/HandFinder.h"

#include "tracker/OpenGL/DebugRenderer/DebugRenderer.h"

#define M_PI 3.14159265358979323846

GLWidget::GLWidget(Worker* worker, DataStream * datastream, SolutionStream * solutions, bool playback, bool real_color, bool use_mano, std::string data_path) :
QGLWidget(OpenGL32Format()),
worker(worker),
datastream(datastream),
solutions(solutions),
_camera(worker->camera),
use_mano(use_mano),
convolution_renderer(worker->model, real_color, data_path) {
	this->playback = playback;
	this->data_path = data_path;
	this->resize(640 * 2, 480 * 2);
	this->move(0, 20);
	convolution_renderer.window_width = 640*2;// this->width();
	convolution_renderer.window_height = 480*2;// this->height();

	if (use_mano)
	{
		model_convertor.init();
	}

	std::cout << "Started OpenGL " << this->format().majorVersion() << "." << this->format().minorVersion() << std::endl;
	// this->installEventFilter(new AntTweakBarEventFilter(this)); ///< all actions pass through filter
	// initialProjectionMatrix();
}

GLWidget::~GLWidget() {
	worker->cleanup_graphic_resources();
	tw_settings->tw_cleanup();

	glDeleteBuffers(1, &(this->Render_PBO));
	glDeleteTextures(1, &(this->Render_texture));
}

void GLWidget::set_store_file(bool is_show, bool is_store, std::string store_path)
{
	show_render = is_show;
	store_render = is_store;
	render_store_path = store_path;
}

void GLWidget::initialProjectionMatrix(int _width, int _height, CamerasParameters camera_par)
{
	pose_camera0 = camera_par.camerapose_c0;// Eigen::Matrix4f::Identity();
	pose_camera1 = camera_par.camerapose_c1;
	invpose_camera0 = pose_camera0.inverse();
	invpose_camera1 = pose_camera1.inverse();

	view_camera0 = invpose_camera0;// Eigen::Matrix4f::Identity();
	view_camera1 = invpose_camera1;
	object_motion = Eigen::Matrix4f::Identity();

	float width = _width;
	float height = _height;

	float clip_far = 800.0f;//800.0f   
	float clip_near = 10.0f;

	/*float fx_c0 = 473.297f*width / 640;
	float fy_c0 = 473.297f*height / 480;
	float cx_c0 = 316.561f*width / 640;
	float cy_c0 = 245.293f*height / 480;*/
	float fx_c0 = camera_par.c0.fx*width / 640;
	float fy_c0 = camera_par.c0.fy*height / 480;
	float cx_c0 = camera_par.c0.cx*width / 640;
	float cy_c0 = camera_par.c0.cy*height / 480;

	projection_matrix_c0 = Eigen::Matrix4f::Identity();

	projection_matrix_c0(0, 0) = 2 * fx_c0 / width;
	projection_matrix_c0(0, 2) = (2 * cx_c0 - width) / width;

	projection_matrix_c0(1, 1) = 2 * fy_c0 / height;
	projection_matrix_c0(1, 2) = (2 * cy_c0 - height) / height;

	projection_matrix_c0(2, 2) = (clip_far + clip_near) / (clip_far - clip_near);
	projection_matrix_c0(2, 3) = -2 * clip_far*clip_near / (clip_far - clip_near);

	projection_matrix_c0(3, 2) = 1.0f;
	projection_matrix_c0(3, 3) = 0.0f;

	/*float fx_c1 = 474.984f*width / 640;
	float fy_c1 = 474.984f*height / 480;
	float cx_c1 = 310.504f*width / 640;
	float cy_c1 = 245.546f*height / 480;*/
	float fx_c1 = camera_par.c1.fx*width / 640;
	float fy_c1 = camera_par.c1.fy*height / 480;
	float cx_c1 = camera_par.c1.cx*width / 640;
	float cy_c1 = camera_par.c1.cy*height / 480;

	projection_matrix_c1 = Eigen::Matrix4f::Identity();

	projection_matrix_c1(0, 0) = 2 * fx_c1 / width;
	projection_matrix_c1(0, 2) = (2 * cx_c1 - width) / width;

	projection_matrix_c1(1, 1) = 2 * fy_c1 / height;
	projection_matrix_c1(1, 2) = (2 * cy_c1 - height) / height;

	projection_matrix_c1(2, 2) = (clip_far + clip_near) / (clip_far - clip_near);
	projection_matrix_c1(2, 3) = -2 * clip_far*clip_near / (clip_far - clip_near);

	projection_matrix_c1(3, 2) = 1.0f;
	projection_matrix_c1(3, 3) = 0.0f;

	//initial time counter
	QueryPerformanceFrequency(&time_stmp);
	count_freq = (double)time_stmp.QuadPart;
}

void GLWidget::initializeGL() {
	this->initializeOpenGLFunctions();

	std::cout << "GLWidget::initializeGL()" << std::endl;
	initialize_glew();
	tw_settings->tw_init(this->width(), this->height()); ///< FIRST!!

	glEnable(GL_DEPTH_TEST);

	kinect_renderer.init(_camera);

	///--- Initialize other graphic resources
	this->makeCurrent();
	worker->init_graphic_resources();

	///--- Setup with data from worker
	kinect_renderer.setup(worker->sensor_color_texture->texid(), worker->sensor_depth_texture->texid());

	convolution_renderer.projection = _camera->view_projection_matrix();
	convolution_renderer.init(ConvolutionRenderer::NORMAL);

	arrow_render = std::make_unique<GLArrow>();
	arrow_render->setProjMat(_camera->view_projection_matrix());
	arrow_render->setViewMat(_camera->view_matrix(), Eigen::Vector3f(0, 0, 0));
	// arrow_render->setProjMat(projection_matrix_c0);
	// arrow_render->setViewMat(view_camera0);
	arrow_render->setZRatio(0.5f);
	// arrow_render->setColor(QVector4D(47 / 255.0, 79 / 255.0, 79 / 255.0, 1.0));

	// 生成纹理对象、PBO
	glGenBuffers(1, &(Render_PBO));
	glBindBuffer(GL_ARRAY_BUFFER, Render_PBO);
	glBufferData(GL_ARRAY_BUFFER, this->width()* this->height() * 3, NULL, GL_STREAM_COPY);
	glBindBuffer(GL_ARRAY_BUFFER, NULL);

	// 初始化纹理属性
	glGenTextures(1, &(Render_texture));
	glBindTexture(GL_TEXTURE_2D, Render_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, this->width(), this->height(), 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, NULL);

	// 初始化纹理属性，分割图显示
	glGenTextures(1, &(Show_texture));
	glBindTexture(GL_TEXTURE_2D, Show_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 320, 240, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, NULL);

	frame_id = 1;
}

void GLWidget::paintGL() {

	/*LONGLONG count_start_render, count_end_render;
	double count_interv, time_inter;
	QueryPerformanceCounter(&time_stmp);
	count_start_render = time_stmp.QuadPart;*/

	Eigen::Matrix4f view_org = Eigen::Matrix4f::Identity();
	Eigen::Vector3f camera_center = Eigen::Vector3f(0, 0, 0);
	Eigen::Matrix4f view_projection;
	Eigen::Matrix4f camera_projection;

	std::vector<float3> point_cloud;
	Eigen::Vector3f color_value;
	char render_image_file[512] = { 0 };


	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, this->width(), this->height());

	/***********************************************************************/
	/*                render the ho results of view0                       */
	/***********************************************************************/
	if (show_render || store_render)
	{
		glClearColor(1, 1, 1, 0);//glClearColor(1, 1, 1, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		view_org = view_camera0;// Eigen::Matrix4f::Identity();
		camera_center = pose_camera0.block(0, 3, 3, 1);
		camera_projection = projection_matrix_c0;
		view_projection = camera_projection * view_org;

		bool _debug_render_point_cloud = false;
		if (_debug_render_point_cloud)
		{

			kinect_renderer.set_projection_point(view_projection);
			point_cloud.clear();
			point_cloud = worker->handfinder->point_cloud;
			color_value[0] = 0.0; color_value[1] = 0.0; color_value[2] = 1.0;
			kinect_renderer.render_point(point_cloud, color_value);
		}

		//render the object nodes
		bool _debug_render_obj_node = false;
		if (_debug_render_obj_node)
		{
			color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
			kinect_renderer.render_object_nodes(worker->object_nodes, color_value);
			kinect_renderer.render_point(worker->interaction_corrs_vertex, color_value);
			kinect_renderer.render_object_nodes(worker->fingertips, color_value);
		}

		//kinect_renderer.render_interaction_correspondences(worker->interaction_corrs_vertex, worker->interaction_corrs_finger_idx);

		point_cloud.clear();
		point_cloud = worker->interaction_corrs_vertex;
		color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
		//kinect_renderer.render_point(point_cloud, color_value);

		point_cloud.clear();
		point_cloud = worker->contact_spheremesh;
		color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
		//kinect_renderer.render_point(point_cloud, color_value);

		//render the object model
		kinect_renderer.set_view_projection_object(view_org, view_projection, object_motion);
		kinect_renderer.render_object_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

		glDisable(GL_BLEND);
		// render hand (spheremesh or mano)
		if (!use_mano)
		{
			convolution_renderer.render3(camera_projection, view_org, camera_center);
		}
		else
		{
			auto center_float3 = worker->model->centers;
			std::vector<unsigned int> indices{
				25,
				15, 14, 13,
				11, 10, 9,
				3, 2, 1,
				7, 6, 5,
				19, 18, 17,
				12, 8, 0, 4, 16
			};
			std::vector<float> keypoints(63, 0);

			for (int i = 0; i < indices.size(); i++)
			{
				keypoints[3 * i + 0] = center_float3[indices[i]].x / 1000;
				keypoints[3 * i + 1] = -center_float3[indices[i]].y / 1000;
				keypoints[3 * i + 2] = center_float3[indices[i]].z / 1000;
			}

			model_convertor.convert(keypoints, worker->frame_id, 10);
			mano_render = std::make_unique<GLMesh>(model_convertor.getMesh());
			mano_render->setProjMat(projection_matrix_c0);
			mano_render->setViewMat(view_camera0, -view_camera0.block(0, 3, 3, 1));
			mano_render->setColor(QVector4D(1.0f, 0.8f, 0.6f, 1.0f));
			mano_render->paint();
		}

		// render forces
		arrow_render->setColor(QVector4D(0.2f, 0.2f, 0.2f, 1.0f));
		arrow_render->setProjMat(projection_matrix_c0);
		arrow_render->setViewMat(view_camera0, -view_camera0.block(0, 3, 3, 1));

		for (int i = 0; i < worker->contact_forces.size(); i++)
		{
			Eigen::Vector3f& p = worker->contact_points[i];
			Eigen::Vector3f& f = worker->contact_forces[i];
			if (f.norm() < 0.1) continue;
			QVector3D qp(p.x(), p.y(), p.z());
			QVector3D qf(f.x(), f.y(), f.z());
			arrow_render->paint(1.8, qf.length() * 35, qf, qp);
		}

		//worker->model->render_outline();
		//DebugRenderer::instance().set_uniform("view_projection", view_projection);
		//DebugRenderer::instance().render();

		//tw_settings->tw_draw();

		//ZH added
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cv::Mat image = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		cv::Mat image_flipped;
		cv::flip(image, image_flipped, 0);
		cv::Mat image640;
		cv::resize(image_flipped, image640, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("hand-object view0", image640);

		if (store_render)
		{
			sprintf(render_image_file, "handobject_view0_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640);
		}
	}

	/***********************************************************************/
	/*               render the hand results of view0                      */
	/***********************************************************************/
#if 1		// Modify by hhy
	if (show_render || store_render)
	{
		glClearColor(1, 1, 1, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		kinect_renderer.set_projection_point(view_projection);

		point_cloud = worker->handfinder->point_cloud;
		color_value[0] = 0.0; color_value[1] = 0.0; color_value[2] = 1.0;
		//		kinect_renderer.render_point(point_cloud, color_value);

		point_cloud.clear();
		point_cloud = worker->handfinder->point_cloud2;
		color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
		//	kinect_renderer.set_point_color(color_value);
		//		kinect_renderer.render_point(point_cloud, color_value);

		//render the object nodes
		color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
		//		kinect_renderer.render_object_nodes(worker->object_nodes, color_value);

		//render the object model
		//kinect_renderer.set_view_projection_object(view_org, view_projection, object_motion);
		//kinect_renderer.render_object_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

		point_cloud.clear();
		point_cloud = worker->contact_spheremesh;
		color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
		//kinect_renderer.render_point(point_cloud, color_value);

		glDisable(GL_BLEND);
		convolution_renderer.render3(camera_projection, view_org, camera_center);

		//ZH added
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cv::Mat image = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		cv::Mat image_flipped;
		cv::flip(image, image_flipped, 0);
		cv::Mat image640;
		cv::resize(image_flipped, image640, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("hand view0", image640);

		if (store_render)
		{
			sprintf(render_image_file, "hand_view0_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640);
		}

	}

	if (show_render || store_render)
	{
		glClearColor(1, 1, 1, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//render the object model
		kinect_renderer.set_view_projection_object(view_org, view_projection, Eigen::Matrix4f::Identity());//object_motion
		kinect_renderer.render_canonical_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

		//glDisable(GL_BLEND);
		//convolution_renderer.render3(view_org, camera_center);

		//ZH added
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cv::Mat image = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		cv::Mat image_flipped;
		cv::flip(image, image_flipped, 0);
		cv::Mat image640;
		cv::resize(image_flipped, image640, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("cano object view0", image640);

		if (store_render)
		{
			sprintf(render_image_file, "cano_object_view0_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640);
		}


		//render the live object model
		glClearColor(1, 1, 1, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		kinect_renderer.set_view_projection_object(view_org, view_projection, object_motion);
		kinect_renderer.render_object_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

		//kinect_renderer.render_interaction_correspondences(worker->interaction_corrs_vertex, worker->interaction_corrs_finger_idx);
		/*point_cloud.clear();
		point_cloud = worker->contact_spheremesh;
		color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
		kinect_renderer.render_point(point_cloud, color_value);*/

		point_cloud.clear();
		point_cloud = worker->interaction_corrs_vertex;
		color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
		//kinect_renderer.render_point(point_cloud, color_value);

		//ZH added
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		//		cv::Mat image = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		//		cv::Mat image_flipped;
		cv::flip(image, image_flipped, 0);
		//		cv::Mat image640;
		cv::resize(image_flipped, image640, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("live object view0", image640);

		if (store_render)
		{
			sprintf(render_image_file, "live_object_view0_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640);
		}
	}

#endif

	/***********************************************************************/
	/*                render the ho results of view1					   */
	/***********************************************************************/
#if 1		
	if (show_render || store_render)
	{
		//view_org(0, 0) = -0.995791316; view_org(0, 1) = 0.000000000; view_org(0, 2) = 0.0916499496; view_org(0, 3) = 52.6652985;
		//view_org(1, 0) = 0.0140768168; view_org(1, 1) = 0.988134205; view_org(1, 2) = 0.152946860; view_org(1, 3) = -49.3974419;
		//view_org(2, 0) = -0.0905624405; view_org(2, 1) = 0.153593287; view_org(2, 2) = -0.983975351; view_org(2, 3) = 695.281067;

		view_org = view_camera1;
		//		static Eigen::Matrix4f view_org_inv = view_org.inverse();

		camera_projection = projection_matrix_c1;
		camera_center(0) = pose_camera1(0, 3); camera_center(1) = pose_camera1(1, 3); camera_center(2) = pose_camera1(2, 3);
		view_projection = camera_projection * view_org;

		glClearColor(1, 1, 1, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//kinect_renderer.set_projection_point(view_projection);

		//point_cloud = worker->handfinder->point_cloud;
		//color_value[0] = 0.0; color_value[1] = 0.0; color_value[2] = 1.0;
		//kinect_renderer.render_point(point_cloud, color_value);

		/*point_cloud.clear();
		point_cloud = worker->handfinder->point_cloud2;
		color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
		kinect_renderer.render_point(point_cloud, color_value);*/

		//render the object nodes
		// color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
		//		kinect_renderer.render_object_nodes(worker->object_nodes, color_value);
		//		kinect_renderer.render_point(worker->interaction_corrs_vertex, color_value);
		//		kinect_renderer.render_object_nodes(worker->fingertips, color_value);
		//kinect_renderer.render_interaction_correspondences(worker->interaction_corrs_vertex, worker->interaction_corrs_finger_idx);

		//point_cloud.clear();
		//point_cloud = worker->interaction_corrs_vertex;
		//color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
		//kinect_renderer.render_point(point_cloud, color_value);

		//point_cloud.clear();
		//point_cloud = worker->contact_spheremesh;
		//color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
		//kinect_renderer.render_point(point_cloud, color_value);

		//render the object model
		kinect_renderer.set_view_projection_object(view_org, view_projection, object_motion);
		kinect_renderer.render_object_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

		glDisable(GL_BLEND);
		convolution_renderer.render3(camera_projection, view_org, camera_center);

		//ZH added
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cv::Mat image_view1 = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image_view1.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		cv::Mat image_view1_flipped;
		cv::flip(image_view1, image_view1_flipped, 0);
		cv::Mat image640_view1;
		cv::resize(image_view1_flipped, image640_view1, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("hand-object view1", image640_view1);

		if (store_render)
		{
			sprintf(render_image_file, "handobject_view1_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640_view1);
		}
	}
#endif

#if 0
	/***********************************************************************/
	/*               render the hand results of view1                      */
	/***********************************************************************/
	if (show_render || store_render)
	{

		glClearColor(1, 1, 1, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		kinect_renderer.set_projection_point(view_projection);

		point_cloud = worker->handfinder->point_cloud;
		color_value[0] = 0.0; color_value[1] = 0.0; color_value[2] = 1.0;
		//		kinect_renderer.render_point(point_cloud, color_value);

		point_cloud.clear();
		point_cloud = worker->handfinder->point_cloud2;
		color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
		//		kinect_renderer.render_point(point_cloud, color_value);

		//render the object nodes
		color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
		//		kinect_renderer.render_object_nodes(worker->object_nodes, color_value);

		point_cloud.clear();
		point_cloud = worker->contact_spheremesh;
		color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
		kinect_renderer.render_point(point_cloud, color_value);

		//render the object model
		//kinect_renderer.set_view_projection_object(view_org, view_projection, object_motion);
		//kinect_renderer.render_object_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

		glDisable(GL_BLEND);
		convolution_renderer.render3(camera_projection, view_org, camera_center);

		//ZH added
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cv::Mat image_view1 = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image_view1.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		cv::Mat image_view1_flipped;
		cv::flip(image_view1, image_view1_flipped, 0);
		cv::Mat image640_view1;
		cv::resize(image_view1_flipped, image640_view1, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("hand view1", image640_view1);

		if (store_render)
		{
			sprintf(render_image_file, "hand_view1_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640_view1);
		}
	}

	if (show_render || store_render)
	{

		glClearColor(1, 1, 1, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//render the cano object model
		kinect_renderer.set_view_projection_object(view_org, view_projection, Eigen::Matrix4f::Identity());//object_motion
		kinect_renderer.render_canonical_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

		//ZH added
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cv::Mat image_view1 = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image_view1.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		cv::Mat image_view1_flipped;
		cv::flip(image_view1, image_view1_flipped, 0);
		cv::Mat image640_view1;
		cv::resize(image_view1_flipped, image640_view1, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("cano object view1", image640_view1);

		if (store_render)
		{
			sprintf(render_image_file, "cano_object_view1_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640_view1);
		}


		//render the live object model
		glClearColor(1, 1, 1, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		kinect_renderer.set_view_projection_object(view_org, view_projection, object_motion);
		kinect_renderer.render_object_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

		//kinect_renderer.render_interaction_correspondences(worker->interaction_corrs_vertex, worker->interaction_corrs_finger_idx);
		point_cloud.clear();
		point_cloud = worker->interaction_corrs_vertex;
		color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
		kinect_renderer.render_point(point_cloud, color_value);

		//ZH added
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		//		cv::Mat image_view1 = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image_view1.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		//		cv::Mat image_view1_flipped;
		cv::flip(image_view1, image_view1_flipped, 0);
		//		cv::Mat image640_view1;
		cv::resize(image_view1_flipped, image640_view1, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("live object view1", image640_view1);

		if (store_render)
		{
			sprintf(render_image_file, "live_object_view1_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640_view1);
		}
	}
#endif

#if 0
	/**********************************************************************/
	/*         render the result from virtual camera1                     */
	/**********************************************************************/

	if (show_render || store_render)
	{
		Eigen::Matrix4f virtual_pose_camera1 = Eigen::Matrix4f::Zero();
		virtual_pose_camera1(0, 2) = -1; virtual_pose_camera1(0, 3) = 380;
		virtual_pose_camera1(2, 0) = 1; virtual_pose_camera1(2, 3) = 380;
		virtual_pose_camera1(1, 1) = 1; virtual_pose_camera1(3, 3) = 1;

		Eigen::Matrix4f virtual_view_camera1 = virtual_pose_camera1.inverse();

		view_org = virtual_view_camera1;

		camera_projection = projection_matrix_c0;
		camera_center(0) = virtual_pose_camera1(0, 3); camera_center(1) = virtual_pose_camera1(1, 3); camera_center(2) = virtual_pose_camera1(2, 3);
		view_projection = camera_projection * view_org;

		glClearColor(1, 1, 1, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		kinect_renderer.set_projection_point(view_projection);

		//render the object model
		kinect_renderer.set_view_projection_object(view_org, view_projection, object_motion);
		kinect_renderer.render_object_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

		glDisable(GL_BLEND);
		convolution_renderer.render3(camera_projection, view_org, camera_center);

		//ZH added
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cv::Mat image_view1 = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image_view1.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		cv::Mat image_view1_flipped;
		cv::flip(image_view1, image_view1_flipped, 0);
		cv::Mat image640_view1;
		cv::resize(image_view1_flipped, image640_view1, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("hand-object virtual view1", image640_view1);

		if (store_render)
		{
			sprintf(render_image_file, "handobject_virtual_view1_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640_view1);
		}

	}

	/**********************************************************************/
	/*         render the result from virtual camera2                     */
	/**********************************************************************/

	if (show_render || store_render)
	{
		Eigen::Matrix4f virtual_pose_camera2 = Eigen::Matrix4f::Zero();

		//from top to downward
		/*virtual_pose_camera2(0, 1) = -1;
		virtual_pose_camera2(1, 2) = -1; virtual_pose_camera2(1, 3) = 380;
		virtual_pose_camera2(2, 0) = 1; virtual_pose_camera2(2, 3) = 380;
		virtual_pose_camera2(3, 3) = 1;*/

		//from the other view
		virtual_pose_camera2(0, 2) = 1; virtual_pose_camera2(0, 3) = -380;
		virtual_pose_camera2(1, 1) = 1;
		virtual_pose_camera2(2, 0) = -1; virtual_pose_camera2(2, 3) = 380;
		virtual_pose_camera2(3, 3) = 1;

		Eigen::Matrix4f virtual_view_camera2 = virtual_pose_camera2.inverse();

		view_org = virtual_view_camera2;

		camera_projection = projection_matrix_c0;
		camera_center(0) = virtual_pose_camera2(0, 3); camera_center(1) = virtual_pose_camera2(1, 3); camera_center(2) = virtual_pose_camera2(2, 3);
		view_projection = camera_projection * view_org;

		glClearColor(1, 1, 1, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		kinect_renderer.set_projection_point(view_projection);

		//render the object model
		kinect_renderer.set_view_projection_object(view_org, view_projection, object_motion);
		kinect_renderer.render_object_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

		glDisable(GL_BLEND);
		convolution_renderer.render3(camera_projection, view_org, camera_center);

		//ZH added
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cv::Mat image_view1 = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image_view1.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		cv::Mat image_view1_flipped;
		cv::flip(image_view1, image_view1_flipped, 0);
		cv::Mat image640_view1;
		cv::resize(image_view1_flipped, image640_view1, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("hand-object virtual view2", image640_view1);

		if (store_render)
		{
			sprintf(render_image_file, "handobject_virtual_view2_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640_view1);
		}

	}
#endif

	/***********************************************************************/
	/*          render the results of live view                            */
	/***********************************************************************/

	view_projection = _camera->view_projection_matrix() * view;

	glClearColor(1, 1, 1, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	bool _debug_render_hand_point_cloud = true;
	if (_debug_render_hand_point_cloud)
	{
		// left
		kinect_renderer.set_projection_point(view_projection);
		point_cloud = worker->handfinder->point_cloud;
		color_value[0] = 0.0; color_value[1] = 0.0; color_value[2] = 1.0;
		kinect_renderer.render_point(point_cloud, color_value);

		// right
		point_cloud.clear();
		point_cloud = worker->handfinder->point_cloud2;
		color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
		kinect_renderer.render_point(point_cloud, color_value);
	}

	//render the object nodes
	//color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
	//kinect_renderer.render_object_index_nodes(worker->object_nodes, worker->node_close2tip_idx, color_value);
	//kinect_renderer.render_object_nodes2tips(worker->object_nodes, worker->node_tip_idx, worker->variant_smooth);
	//kinect_renderer.render_point(worker->interaction_corrs_vertex, color_value);
	//	kinect_renderer.render_object_nodes(worker->fingertips, color_value);

	/*kinect_renderer.render_interaction_correspondences(worker->interaction_corrs_vertex, worker->interaction_corrs_finger_idx);
	std::cout << "interaction corr size:" << worker->interaction_corrs_vertex.size() << std::endl;*/

	/*kinect_renderer.render_interaction_correspondences(worker->contact_spheremesh, worker->interaction_corrs_finger_idx);
	std::cout << "interaction corr size:" << worker->contact_spheremesh.size() << std::endl;*/

	/*point_cloud.clear();
	point_cloud = worker->interaction_corrs_vertex;
	color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
	kinect_renderer.render_point(point_cloud, color_value);

	point_cloud.clear();
	point_cloud = worker->contact_spheremesh;
	color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
	kinect_renderer.render_point(point_cloud, color_value);

	std::cout << "interaction corr size:" << worker->contact_spheremesh.size() << std::endl;*/

	//render the object model
	kinect_renderer.set_view_projection_object(view, view_projection, object_motion);
	kinect_renderer.render_object_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

	glDisable(GL_BLEND);

	// render hand
	if (!use_mano)
	{
		convolution_renderer.render();
	}
	else
	{
		auto center_float3 = worker->model->centers;
		std::vector<unsigned int> indices{
			25,
			15, 14, 13,
			11, 10, 9,
			3, 2, 1,
			7, 6, 5,
			19, 18, 17,
			12, 8, 0, 4, 16
		};
		std::vector<float> keypoints(63, 0);

		for (int i = 0; i < indices.size(); i++)
		{
			keypoints[3 * i + 0] = center_float3[indices[i]].x / 1000;
			keypoints[3 * i + 1] = -center_float3[indices[i]].y / 1000;
			keypoints[3 * i + 2] = center_float3[indices[i]].z / 1000;
		}

		model_convertor.convert(keypoints, worker->frame_id, 10);
		mano_render = std::make_unique<GLMesh>(model_convertor.getMesh());
		mano_render->setProjMat(_camera->view_projection_matrix());
		mano_render->setViewMat(convolution_renderer.camera.view, convolution_renderer.camera.camera_center);
		mano_render->setColor(QVector4D(1.0f, 0.8f, 0.6f, 1.0f));
		mano_render->paint();
	}

	// render forces
	arrow_render->setProjMat(_camera->view_projection_matrix());
	arrow_render->setViewMat(convolution_renderer.camera.view, convolution_renderer.camera.camera_center);
	arrow_render->setColor(QVector4D(0.2f, 0.2f, 0.2f, 1.0f));
	for (int i = 0; i < worker->contact_forces.size(); i++)
	{
		Eigen::Vector3f& p = worker->contact_points[i];
		Eigen::Vector3f& f = worker->contact_forces[i];
		if (f.norm() < 0.1) continue;
		QVector3D qp(p.x(), p.y(), p.z());
		QVector3D qf(f.x(), f.y(), f.z());
		arrow_render->paint(1.8, qf.length() * 35, qf, qp);
	}

	// render target tip pos
	bool _debug_render_target_tip_pos = false;
	if (_debug_render_target_tip_pos && worker->tar_tip_point.size())
	{
		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		kinect_renderer.set_projection_point(view_projection);
		kinect_renderer.render_object_nodes(worker->tar_tip_point, Eigen::Vector3f(0, 1.0f, 0));
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);
	}

	// render target force
	bool _debug_render_target_force = false;
	if (_debug_render_target_force)
	{
		Eigen::Vector3f& oc = worker->object_center;
		QVector3D qoc(oc.x(), oc.y(), oc.z());
		Eigen::Vector3f& tf = worker->target_force;
		QVector3D qtf(tf.x(), tf.y(), tf.z());
		Eigen::Vector3f& tm = worker->target_moment;
		QVector3D qtm(tm.x(), tm.y(), tm.z());
		Eigen::Vector3f& ov = worker->object_vel;
		QVector3D qov(ov.x(), ov.y(), ov.z());
		Eigen::Vector3f& oav = worker->object_ang_vel;
		QVector3D qoav(oav.x(), oav.y(), oav.z());

		Eigen::Vector3f x_axis = worker->object_rot * Eigen::Vector3f(1, 0, 0);
		Eigen::Vector3f y_axis = worker->object_rot * Eigen::Vector3f(0, 1, 0);
		Eigen::Vector3f z_axis = worker->object_rot * Eigen::Vector3f(0, 0, 1);
		QVector3D qx(x_axis.x(), x_axis.y(), x_axis.z());
		QVector3D qy(y_axis.x(), y_axis.y(), y_axis.z());
		QVector3D qz(z_axis.x(), z_axis.y(), z_axis.z());

		arrow_render->setColor(QVector4D(0, 1.0f, 1.0f, 1.0f));
		arrow_render->paint(1, qtf.length() * 200, qtf, qoc);	// force
		arrow_render->paint(1, qov.length() * 200, qov, qoc);		// velocity

		arrow_render->setColor(QVector4D(1.0f, 0, 1.0f, 1.0f));
		arrow_render->paint(1, qtm.length() * 500, qtm, qoc);	// moment
		arrow_render->paint(1, qoav.length() * 50, qoav, qoc);		// angle velocity
	}

	glDisable(GL_BLEND);
	//ZH added
#if 0:
	if (show_render || store_render)
	{
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cv::Mat image_live = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image_live.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		cv::Mat image_live_flipped;
		cv::flip(image_live, image_live_flipped, 0);
		//	cv::flip(image_flipped, image_flipped, 1);
		cv::Mat image640_live;
		cv::resize(image_live_flipped, image640_live, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("rendered image live view", image640_live);

		//char render_image_file[512] = { 0 };

		// Modify by hhy
		if (store_render)
		{
			sprintf(render_image_file, "live_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640_live);
		}
	}

	//
	/*sprintf(render_image_file, "render/live_%04d.png", worker->frame_id);
	cv::imwrite(render_image_file, image640_live);*/
#endif
}

void GLWidget::paintGLCtrl_Skeleton()
{
	/*LONGLONG count_start_render, count_end_render;
	double count_interv, time_inter;
	QueryPerformanceCounter(&time_stmp);
	count_start_render = time_stmp.QuadPart;*/

	Eigen::Matrix4f view_org = Eigen::Matrix4f::Identity();
	Eigen::Vector3f camera_center = Eigen::Vector3f(0, 0, 0);
	Eigen::Matrix4f view_projection;
	Eigen::Matrix4f camera_projection;

	std::vector<float3> point_cloud;
	Eigen::Vector3f color_value;
	char render_image_file[512] = { 0 };

	view_projection = _camera->view_projection_matrix() * view;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/////////////////////////////////////////////////////////////////////////////
	//show keypoints
	glViewport(640 * 2, 480, 640, 480);
	
	/*color_value[0] = 0.0; color_value[1] = 0.0; color_value[2] = 1.0;
	kinect_renderer.render_object_nodes(worker->get_hand_keypoints(), color_value);*/

	/*kinect_renderer.set_projection_line(view_projection);
	kinect_renderer.render_line(worker->get_pred_skeleton(), worker->get_hand_keypoints_color());*/

	kinect_renderer.render_img(worker->keypt_pred_img);

	// show images
	glViewport(640*2, 0, 640, 480);
	//show segmentation
	kinect_renderer.render_img(worker->segmentation);

	/***********************************************************************/
	/*          render the results of live view                            */
	/***********************************************************************/
	glViewport(0, 0, 640 * 2, 480 * 2);

	kinect_renderer.set_projection_point(view_projection);

	point_cloud = worker->handfinder->point_cloud;
	color_value[0] = 0.0; color_value[1] = 0.0; color_value[2] = 1.0;
	//kinect_renderer.render_point(point_cloud, color_value);

	/*point_cloud.clear();
	point_cloud = worker->handfinder->point_cloud2;
	color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
	kinect_renderer.render_point(point_cloud, color_value);*/

	//render the object nodes
	//color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
	//kinect_renderer.render_object_index_nodes(worker->object_nodes, worker->node_close2tip_idx, color_value);
	//kinect_renderer.render_object_nodes2tips(worker->object_nodes, worker->node_tip_idx, worker->variant_smooth);
	//kinect_renderer.render_point(worker->interaction_corrs_vertex, color_value);
	//	kinect_renderer.render_object_nodes(worker->fingertips, color_value);

	/*kinect_renderer.render_interaction_correspondences(worker->interaction_corrs_vertex, worker->interaction_corrs_finger_idx);
	std::cout << "interaction corr size:" << worker->interaction_corrs_vertex.size() << std::endl;*/

	/*kinect_renderer.render_interaction_correspondences(worker->contact_spheremesh, worker->interaction_corrs_finger_idx);
	std::cout << "interaction corr size:" << worker->contact_spheremesh.size() << std::endl;*/

	/*point_cloud.clear();
	point_cloud = worker->interaction_corrs_vertex;
	color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
	kinect_renderer.render_point(point_cloud, color_value);

	point_cloud.clear();
	point_cloud = worker->contact_spheremesh;
	color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
	kinect_renderer.render_point(point_cloud, color_value);

	std::cout << "interaction corr size:" << worker->contact_spheremesh.size() << std::endl;*/

	//render the object model
	kinect_renderer.set_view_projection_object(view, view_projection, object_motion);
	kinect_renderer.render_object_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

	/*QueryPerformanceCounter(&time_stmp);
	count_end_render = time_stmp.QuadPart;
	count_interv = (double)(count_end_render - count_start_render);
	time_inter = count_interv * 1000 / count_freq;
	std::cout << "render object model" << "	" << time_inter << std::endl;


	QueryPerformanceCounter(&time_stmp);
	count_start_render = time_stmp.QuadPart;*/

	glDisable(GL_BLEND);
	convolution_renderer.render();

	/*QueryPerformanceCounter(&time_stmp);
	count_end_render = time_stmp.QuadPart;
	count_interv = (double)(count_end_render - count_start_render);
	time_inter = count_interv * 1000 / count_freq;
	std::cout << "render hand model" << "	" << time_inter << std::endl;*/


	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	{
		glLoadIdentity();
		glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		{
			glEnable(GL_TEXTURE_2D);
			{// Draw each sub window
				glBindTexture(GL_TEXTURE_2D, Show_texture);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, worker->segmentation.cols, worker->segmentation.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, worker->segmentation.data);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glBegin(GL_QUAD_STRIP); {
					glTexCoord2f(0, 1);  glVertex2f(0.665, 0.529);//glVertex2f(winReg[w][0], winReg[w][1]); //
					glTexCoord2f(1, 1);  glVertex2f(1, 0.529);//glVertex2f(winReg[w][2], winReg[w][1]); //
					glTexCoord2f(1, 0);  glVertex2f(1, 1);//glVertex2f(winReg[w][2], winReg[w][3]); //
					glTexCoord2f(0, 0);  glVertex2f(0.665, 1);//glVertex2f(winReg[w][0], winReg[w][3]); //
				}
				glEnd();
			}
			glDisable(GL_TEXTURE_2D);
		}
		glPopMatrix();
	}
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glFlush();

	//ZH added
	if (show_render || store_render)
	{
		// copy to Render_PBO
		glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
		glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

		// copy to Texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cv::Mat image_live = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
		glBindTexture(GL_TEXTURE_2D, Render_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image_live.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		cv::Mat image_live_flipped;
		cv::flip(image_live, image_live_flipped, 0);
		//	cv::flip(image_flipped, image_flipped, 1);
		cv::Mat image640_live;
		cv::resize(image_live_flipped, image640_live, cv::Size(640*1.5, 480), (0, 0), (0, 0), cv::INTER_LINEAR);

		if (show_render)
			cv::imshow("rendered image live view", image640_live);

		//char render_image_file[512] = { 0 };
		if (store_render)
		{
			sprintf(render_image_file, "live_%04d.png", worker->frame_id);
			cv::imwrite(render_store_path + render_image_file, image640_live);
		}
	}

	//
	/*sprintf(render_image_file, "render/live_%04d.png", worker->frame_id);
	cv::imwrite(render_image_file, image640_live);*/

}

void GLWidget::paintPredictedHand()
{
	/***********************************************************************/
	/*          render the results of live view                            */
	/***********************************************************************/
	Eigen::Matrix4f view_projection;
	std::vector<float3> point_cloud;
	Eigen::Vector3f color_value;

	view_projection = _camera->view_projection_matrix() * view;

	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	kinect_renderer.set_projection_point(view_projection);

	point_cloud = worker->handfinder->point_cloud;
	color_value[0] = 0.0; color_value[1] = 0.0; color_value[2] = 1.0;
//	kinect_renderer.render_point(point_cloud, color_value);

	point_cloud.clear();
	point_cloud = worker->handfinder->point_cloud2;
	color_value[0] = 0.0; color_value[1] = 1.0; color_value[2] = 0.0;
//	kinect_renderer.render_point(point_cloud, color_value);

	//render the object nodes
	color_value[0] = 1.0; color_value[1] = 0.0; color_value[2] = 0.0;
	//kinect_renderer.render_object_index_nodes(worker->object_nodes, worker->node_close2tip_idx, color_value);
	//kinect_renderer.render_object_nodes2tips(worker->object_nodes, worker->node_tip_idx, worker->variant_smooth);
	kinect_renderer.render_point(worker->interaction_corrs_vertex, color_value);
	//	kinect_renderer.render_object_nodes(worker->fingertips, color_value);
	kinect_renderer.render_interaction_correspondences(worker->interaction_corrs_vertex, worker->interaction_corrs_finger_idx);

	//render the object model
	kinect_renderer.set_view_projection_object(view, view_projection, object_motion);
	kinect_renderer.render_object_model(worker->object_live_points, worker->object_live_normals, worker->vertex_number);

	/*QueryPerformanceCounter(&time_stmp);
	count_end_render = time_stmp.QuadPart;
	count_interv = (double)(count_end_render - count_start_render);
	time_inter = count_interv * 1000 / count_freq;
	std::cout << "render object model" << "	" << time_inter << std::endl;


	QueryPerformanceCounter(&time_stmp);
	count_start_render = time_stmp.QuadPart;*/

	glDisable(GL_BLEND);
	convolution_renderer.render();

	/*QueryPerformanceCounter(&time_stmp);
	count_end_render = time_stmp.QuadPart;
	count_interv = (double)(count_end_render - count_start_render);
	time_inter = count_interv * 1000 / count_freq;
	std::cout << "render hand model" << "	" << time_inter << std::endl;*/

	//ZH added
	// copy to Render_PBO
	glBindBuffer(GL_PIXEL_PACK_BUFFER, Render_PBO);
	glReadPixels(0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, NULL);

	// copy to Texture from PBO
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Render_PBO);
	glBindTexture(GL_TEXTURE_2D, Render_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	cv::Mat image_live = cv::Mat(this->height(), this->width(), CV_8UC3, cv::Scalar(0));
	glBindTexture(GL_TEXTURE_2D, Render_texture);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, image_live.data);
	glBindTexture(GL_TEXTURE_2D, 0);

	cv::Mat image_live_flipped;
	cv::flip(image_live, image_live_flipped, 0);
	//	cv::flip(image_flipped, image_flipped, 1);
	cv::Mat image640_live;
	cv::resize(image_live_flipped, image640_live, cv::Size(640, 480), (0, 0), (0, 0), cv::INTER_LINEAR);
	cv::imshow("rendered preidcted hand", image640_live);

	char render_image_file[512] = { 0 };
	sprintf(render_image_file, "predicted_live_%04d.png", worker->frame_id);
	cv::imwrite(render_store_path + render_image_file, image640_live);
}

void GLWidget::set_object_motion(Eigen::Matrix4f object_mot)
{
	object_motion = object_mot;
}

void GLWidget::reinitView()
{
	// set view matrix to init value
	view = Eigen::Matrix4f::Identity();
	camera_center = Eigen::Vector3f::Zero();

	convolution_renderer.camera.view = view;
	convolution_renderer.camera.camera_center = camera_center;

	worker->offscreen_renderer.convolution_renderer_ptr->camera.view = view;
	worker->offscreen_renderer.convolution_renderer_ptr->camera.camera_center = camera_center;

	initial_euler_angles = Eigen::Vector2f(-6.411, -1.8);
}

void GLWidget::render_texture_rigid()
{
	glEnable(GL_CULL_FACE);
	// cull front face
	glCullFace(GL_FRONT);

	//render the texture of object model for view0
	glBindFramebuffer(GL_FRAMEBUFFER, kinect_renderer.fbo_rigid_motion_c0);
	glViewport(0, 0, 320, 240);
	GLfloat m_vmap_clear[4] = { 0,0,0,0 };
	GLfloat m_zbuffer_clear = 1;
	glClearBufferfv(GL_COLOR, 0, m_vmap_clear);
	glClearBufferfv(GL_COLOR, 1, m_vmap_clear);
	glClearBufferfv(GL_DEPTH, 0, &m_zbuffer_clear);

//	Eigen::Matrix4f view_projection_camera0 = projection_matrix_c0;//the projection matrix is not precise and should be modified
	kinect_renderer.set_view_projection_texture(projection_matrix_c0, view_camera0, object_motion);
	kinect_renderer.render_object_model_texture( worker->vertex_number);

	//cv::Mat live_vertex_image = cv::Mat(240, 320, CV_32FC4, cv::Scalar(0.0, 0.0, 0.0, 0.0));
	//glBindTexture(GL_TEXTURE_2D, kinect_renderer.tex_live_vmap_c0);
	//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, live_vertex_image.data);
	//glBindTexture(GL_TEXTURE_2D, 0);
	//cv::imshow("live vertex image", live_vertex_image);

	//cv::Mat live_normal_image = cv::Mat(240, 320, CV_32FC4, cv::Scalar(0.0, 0.0, 0.0, 0.0));
	//glBindTexture(GL_TEXTURE_2D, kinect_renderer.tex_live_nmap_c0);
	//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, live_normal_image.data);
	//glBindTexture(GL_TEXTURE_2D, 0);
	//cv::imshow("live normal image", live_normal_image);

	// cull front face
	glCullFace(GL_FRONT);
	//render the texture of object model for view1
	glBindFramebuffer(GL_FRAMEBUFFER, kinect_renderer.fbo_rigid_motion_c1);
	glViewport(0, 0, 320, 240);
	glClearBufferfv(GL_COLOR, 0, m_vmap_clear);
	glClearBufferfv(GL_COLOR, 1, m_vmap_clear);
	glClearBufferfv(GL_DEPTH, 0, &m_zbuffer_clear);

	kinect_renderer.set_view_projection_texture( projection_matrix_c1, view_camera1, object_motion);
	kinect_renderer.render_object_model_texture(worker->vertex_number);
	
	glDisable(GL_CULL_FACE);
}

void GLWidget::render_texture_nonrigid()
{
	glEnable(GL_CULL_FACE);
	// cull front face
	glCullFace(GL_FRONT);

	//render the texture of object model
	glBindFramebuffer(GL_FRAMEBUFFER, kinect_renderer.fbo_nonrigid_motion_c0);
	glViewport(0, 0, 320, 240);
	GLfloat m_vmap_clear[4] = { 0,0,0,0 };
	GLfloat m_zbuffer_clear = 1;
	glClearBufferfv(GL_COLOR, 0, m_vmap_clear);
	glClearBufferfv(GL_COLOR, 1, m_vmap_clear);
	glClearBufferfv(GL_DEPTH, 0, &m_zbuffer_clear);

	//	Eigen::Matrix4f view_projection_camera0 = projection_matrix_c0;//the projection matrix is not precise and should be modified
	kinect_renderer.set_view_projection_texture_nonrigid( projection_matrix_c0, view_camera0, object_motion );
	kinect_renderer.render_object_model_texture_nonrigid(worker->vertex_number);

	//cv::Mat can_vertex_image = cv::Mat(240, 320, CV_32FC4, cv::Scalar(0.0, 0.0, 0.0, 0.0));
	//glBindTexture(GL_TEXTURE_2D, kinect_renderer.tex_can_vmap_c0);
	//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, can_vertex_image.data);
	//glBindTexture(GL_TEXTURE_2D, 0);
	//cv::imshow("can vertex image c0", can_vertex_image);

	//cv::Mat can_normal_image = cv::Mat(240, 320, CV_32FC4, cv::Scalar(0.0, 0.0, 0.0, 0.0));
	//glBindTexture(GL_TEXTURE_2D, kinect_renderer.tex_can_nmap_c0);
	//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, can_normal_image.data);
	//glBindTexture(GL_TEXTURE_2D, 0);
	//cv::imshow("can normal image c0", can_normal_image);

	/*cv::Mat live_normal_image = cv::Mat(240, 320, CV_32FC4, cv::Scalar(0.0, 0.0, 0.0, 0.0));
	glReadBuffer(GL_COLOR_ATTACHMENT1);
	glReadPixels(0, 0, 320, 240, GL_RGBA, GL_FLOAT, live_normal_image.data);*/

	//render the texture of object model
	glBindFramebuffer(GL_FRAMEBUFFER, kinect_renderer.fbo_nonrigid_motion_c1);
	glViewport(0, 0, 320, 240);
	glClearBufferfv(GL_COLOR, 0, m_vmap_clear);
	glClearBufferfv(GL_COLOR, 1, m_vmap_clear);
	glClearBufferfv(GL_DEPTH, 0, &m_zbuffer_clear);

	//	Eigen::Matrix4f view_projection_camera0 = projection_matrix_c0;//the projection matrix is not precise and should be modified
	kinect_renderer.set_view_projection_texture_nonrigid(projection_matrix_c1, view_camera1, object_motion);
	kinect_renderer.render_object_model_texture_nonrigid(worker->vertex_number);

	//cv::Mat can_vertex_image_c1 = cv::Mat(240, 320, CV_32FC4, cv::Scalar(0.0, 0.0, 0.0, 0.0));
	//glBindTexture(GL_TEXTURE_2D, kinect_renderer.tex_can_vmap_c1);
	//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, can_vertex_image_c1.data);
	//glBindTexture(GL_TEXTURE_2D, 0);
	//cv::imshow("can vertex image c1", can_vertex_image_c1);

	//cv::Mat can_normal_image_c1 = cv::Mat(240, 320, CV_32FC4, cv::Scalar(0.0, 0.0, 0.0, 0.0));
	//glBindTexture(GL_TEXTURE_2D, kinect_renderer.tex_can_nmap_c1);
	//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, can_normal_image_c1.data);
	//glBindTexture(GL_TEXTURE_2D, 0);
	//cv::imshow("can normal image c1", can_normal_image_c1);

	glDisable(GL_CULL_FACE);
}

void GLWidget::process_mouse_movement(GLfloat cursor_x, GLfloat cursor_y) {
	glm::vec3 image_center_glm = worker->model->centers[worker->model->centers_name_to_id_map["palm_back"]] +
		worker->model->centers[worker->model->centers_name_to_id_map["palm_middle"]];
	image_center = Eigen::Vector3f(image_center_glm[0] / 2, image_center_glm[1] / 2 + 30, image_center_glm[2] / 2);
	float d = (camera_center - image_center).norm();

	float delta_x = cursor_x - cursor_position[0];
	float delta_y = cursor_y - cursor_position[1];

	float theta = initial_euler_angles[0] + cursor_sensitivity * delta_x;
	float phi = initial_euler_angles[1] + cursor_sensitivity * delta_y;

	Eigen::Vector3f x = sin(theta) * sin(phi) * Eigen::Vector3f::UnitX();
	Eigen::Vector3f y = cos(phi) * Eigen::Vector3f::UnitY();
	Eigen::Vector3f z = cos(theta) * sin(phi) * Eigen::Vector3f::UnitZ();

	camera_center = image_center + d * (x + y + z);
	euler_angles = Eigen::Vector2f(theta, phi);

	Vector3 f, u, s;
	f = (image_center - camera_center).normalized();
	u = camera_up.normalized();
	s = u.cross(f).normalized();
	u = f.cross(s);
	view.block(0, 0, 1, 3) = s.transpose();
	view(0, 3) = -s.dot(camera_center);
	view.block(1, 0, 1, 3) = u.transpose();
	view(1, 3) = -u.dot(camera_center);
	view.block(2, 0, 1, 3) = f.transpose();
	view(2, 3) = -f.dot(camera_center);

	// set view matrix 
	convolution_renderer.camera.view = view;
	convolution_renderer.camera.camera_center = camera_center;

	worker->offscreen_renderer.convolution_renderer_ptr->camera.view = view;
	worker->offscreen_renderer.convolution_renderer_ptr->camera.camera_center = camera_center;
}

void GLWidget::process_mouse_button_pressed(GLfloat cursor_x, GLfloat cursor_y) {
	mouse_button_pressed = true;
	cursor_position = Eigen::Vector2f(cursor_x, cursor_y);
}

void GLWidget::process_mouse_button_released() {
	initial_euler_angles = euler_angles;
}

void GLWidget::mouseMoveEvent(QMouseEvent *event) {
	if (event->buttons() == Qt::LeftButton) {
		process_mouse_movement(event->x(), event->y());
	}
	else {
		if (mouse_button_pressed == true) {
			process_mouse_button_released();
			mouse_button_pressed = false;
		}
	}
}

void GLWidget::mousePressEvent(QMouseEvent *event) {
	process_mouse_button_pressed(event->x(), event->y());
}

void GLWidget::wheelEvent(QWheelEvent * event) {

//	printf("------------>wheel data:%d\n", event->delta());


}

void GLWidget::keyPressEvent(QKeyEvent *event) {
	GLWidget* qglviewer = this;
	switch (event->key()){
	case Qt::Key_Escape: {
		this->close();
	}
	break;
	case Qt::Key_S: {
		cout << "set up path for saving images" << std::endl;
		//datastream->save_as_images(data_path);
	}
	break;
	case Qt::Key_1: {
		cout << "uniform scaling up" << endl;
		worker->model->resize_model(1.05, 1.0, 1.0);
	}
	break;
	case Qt::Key_2: {
		cout << "uniform scaling down" << endl;
		worker->model->resize_model(0.95, 1.0, 1.0);
	}
	break;
	case Qt::Key_3: {
		cout << "width scaling up" << endl;
		worker->model->resize_model(1.0, 1.05, 1.0);
	}
	break;
	case Qt::Key_4: {
		cout << "width scaling down" << endl;
		worker->model->resize_model(1.0, 0.95, 1.0);
	}
	break;
	case Qt::Key_5: {
		cout << "thickness scaling up" << endl;
		worker->model->resize_model(1.0, 1.0, 1.05);
	}
	break;
	case Qt::Key_6: {
		cout << "thickness scaling down" << endl;
		worker->model->resize_model(1.0, 1.0, 0.95);
	}
	break;
	case Qt::Key_W: {
		worker->is_watch = !worker->is_watch;
		cout << "is watch:"<<worker->is_watch << endl;
	}
	break;
	case Qt::Key_R: {
		worker->is_reconstruction = !worker->is_reconstruction;
		cout << "is reconstruction:" << worker->is_reconstruction << endl;
	}
	break;
	case Qt::Key_T: {
		worker->is_only_track = !worker->is_only_track;
		cout << "is only track:" << worker->is_only_track << endl;
	}
	break;
	case Qt::Key_V: {
		reinitView();
		cout << "view reinitialize" << endl;
	}
	}
}

