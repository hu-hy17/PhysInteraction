#include "Worker.h"
#include "util/gl_wrapper.h"
#include "util/tictoc.h"

#include <QElapsedTimer>
#include <QGLWidget>

#include "cudax/externs.h"

#include "tracker/ForwardDeclarations.h"
// #include "tracker/Sensor/Sensor.h"
#include "tracker/Data/TextureColor8UC3.h"
#include "tracker/Data/TextureDepth16UC1.h"
#include "tracker/HandFinder/HandFinder.h"
#include "tracker/Energy/Energy.h"
#include "tracker/TwSettings.h"
#include "tracker/HModel/Model.h"

#include "physhand/Defs.h"

#include <ctime>

void Worker::updateGL() { if (glarea != NULL) glarea->updateGL(); }

void Worker::render_predicted_hand() { if (gl_resrc_ptr != NULL) gl_resrc_ptr->paintPredictedHand(); }

void Worker::render_models() { if (gl_resrc_ptr != NULL) gl_resrc_ptr->paintGL(); }

void Worker::render_texture_rigid() { if (gl_resrc_ptr != NULL) gl_resrc_ptr->render_texture_rigid(); }

void Worker::render_texture_nonrigid() { if (gl_resrc_ptr != NULL)gl_resrc_ptr->render_texture_nonrigid(); }

void Worker::initialRendererCamMatrix(int _width, int _height, CamerasParameters camera_par)
{
	if (gl_resrc_ptr != NULL) 
		gl_resrc_ptr->initialProjectionMatrix(_width,_height, camera_par);
}

void Worker::initialWorkerCamMatrix(int _width, int _height, CamerasParameters camera_par)
{
	para_camera0 = camera_par.c0;
	para_camera1 = camera_par.c1;

	pose_camera0 = camera_par.camerapose_c0;
	pose_camera1 = camera_par.camerapose_c1;

	view_camera0 = pose_camera0.inverse();
	view_camera1 = pose_camera1.inverse();

	kernel_set_camera_para(para_camera0.fx / 2, para_camera0.fy / 2);
}

void Worker::set_camera_object_motion(Eigen::Matrix4f object_motion)
{
	if (gl_resrc_ptr != NULL)
	{
		gl_resrc_ptr->set_object_motion(object_motion);
	}
}

Worker::Worker(Camera *camera, bool test, bool benchmark, bool save_rasotrized_model, int user_name, std::string data_path) {

	this->camera = camera;
	this->benchmark = benchmark;
	this->test = test;
	this->save_rastorized_model = save_rasotrized_model;
	this->user_name = user_name;
	this->data_path = data_path;

	this->model = new Model();
	this->model->init(user_name, data_path);
	std::vector<float> theta_initial = std::vector<float>(num_thetas, 0);
	theta_initial[1] = -70; theta_initial[2] = 400;
	model->move(theta_initial);

	model->update_centers();
	model->compute_outline();

	if (user_name == 0) model->manually_adjust_initial_transformations();

	//set keypoint to block
	Keypoint_block.resize(21);
	{
		Keypoint_block[0] = 12;//thumb top
		Keypoint_block[1] = 13;//thumb middle
		Keypoint_block[2] = 14;//thumb bottom
		Keypoint_block[3] = 17;//thumb base

		Keypoint_block[4] = 17;//root
		
		Keypoint_block[5] = 9;//index top
		Keypoint_block[6] = 10;//index middle
		Keypoint_block[7] = 11;//index bottom
		Keypoint_block[8] = 19;//index base

		Keypoint_block[9] = 6;//middle top
		Keypoint_block[10] = 7;//middle middle
		Keypoint_block[11] = 8;//middle bottom
		Keypoint_block[12] = 18;//middle base

		Keypoint_block[13] = 3;//ring top
		Keypoint_block[14] = 4;//ring middle
		Keypoint_block[15] = 5;//ring bottom
		Keypoint_block[16] = 16;//ring base

		Keypoint_block[17] = 0;//pinky top
		Keypoint_block[18] = 1;//pinky middle
		Keypoint_block[19] = 2;//pinky bottom
		Keypoint_block[20] = 15;//pinky base
	}

	//set keypoint to sphere center
	Keypoint2SphereCenter.resize(21);
	{
		Keypoint2SphereCenter[0] = 16;//thumb top
		Keypoint2SphereCenter[1] = 17;//thumb middle
		Keypoint2SphereCenter[2] = 18;//thumb bottom
		Keypoint2SphereCenter[3] = 19;//thumb base

		Keypoint2SphereCenter[4] = 25;//root

		Keypoint2SphereCenter[5] = 12;//index top
		Keypoint2SphereCenter[6] = 13;//index middle
		Keypoint2SphereCenter[7] = 14;//index bottom
		Keypoint2SphereCenter[8] = 15;// 23;// 15;//index base

		Keypoint2SphereCenter[9] = 8;//middle top
		Keypoint2SphereCenter[10] = 9;//middle middle
		Keypoint2SphereCenter[11] = 10;//middle bottom
		Keypoint2SphereCenter[12] = 11;// 22;// 11;//middle base

		Keypoint2SphereCenter[13] = 4;//ring top
		Keypoint2SphereCenter[14] = 5;//ring middle
		Keypoint2SphereCenter[15] = 6;//ring bottom
		Keypoint2SphereCenter[16] = 7;// 21;// 7;//ring base

		Keypoint2SphereCenter[17] = 0;//pinky top
		Keypoint2SphereCenter[18] = 1;//pinky middle
		Keypoint2SphereCenter[19] = 2;//pinky bottom
		Keypoint2SphereCenter[20] = 3;//pinky base
	}

	//set using keypoint 
	using_keypoint_2D.clear();
	{
		//thumb
		using_keypoint_2D.push_back(0);
		using_keypoint_2D.push_back(1);
		using_keypoint_2D.push_back(2);
		using_keypoint_2D.push_back(3);
		
		//root
		using_keypoint_2D.push_back(4);

		////index
		using_keypoint_2D.push_back(5);
		using_keypoint_2D.push_back(6);
		using_keypoint_2D.push_back(7);
		using_keypoint_2D.push_back(8);

		//middle
		using_keypoint_2D.push_back(9);
		using_keypoint_2D.push_back(10);
		using_keypoint_2D.push_back(11);
		using_keypoint_2D.push_back(12);

		//ring
		using_keypoint_2D.push_back(13);
		using_keypoint_2D.push_back(14);
		using_keypoint_2D.push_back(15);
		using_keypoint_2D.push_back(16);

		//pinky
		using_keypoint_2D.push_back(17);
		using_keypoint_2D.push_back(18);
		using_keypoint_2D.push_back(19);
		using_keypoint_2D.push_back(20);
	}

	using_keypoint_tips.resize(5);
	{
		using_keypoint_tips[0] = 5;		// index top
		using_keypoint_tips[1] = 9;		// middle top
		using_keypoint_tips[2] = 17;	// pinky top
		using_keypoint_tips[3] = 13;	// ring top
		using_keypoint_tips[4] = 0;		// thumb top
	}
	tips_rel_conf.resize(5);
	tips_org_conf.resize(5);

	E_KeyPoint3D.keypoint_3D_debug_file.open(E_KeyPoint3D.keypoint_3D_debug_file_path);

	//initialize OneEuroFilter
	OEFilter.init(22);
}

/// @note any initialization that has to be done once GL context is active
void Worker::init_graphic_resources() {
	offscreen_renderer.init(camera, model, data_path, true);
	if (save_rastorized_model) rastorizer.init(camera, model, data_path, false);
	sensor_color_texture = new ColorTexture8UC3(camera->width(), camera->height());
	sensor_depth_texture = new DepthTexture16UC1(camera->width(), camera->height());
	camera_depth_texture = new DepthTexture16UC1Recon(camera->width(), camera->height());

	tw_settings->tw_add(settings->termination_max_iters, "#iters", "group=Tracker");
	tw_settings->tw_add(settings->termination_max_rigid_iters, "#iters (rigid)", "group=Tracker");

	///--- Initialize the energies modules
	using namespace energy;
	trivial_detector = new TrivialDetector(camera, &offscreen_renderer);
	handfinder = new HandFinder(camera);
	E_fitting.init(this);

	E_limits.init(model);
	E_limits_second.init(model);
	E_collision.init(model);
	E_collision_second.init(model);
	E_pose.init(this);
	E_pose_second.init(this);
	E_temporal.init(model);
	E_damping.init(model);

	init_graphic_map_resources();
}

void Worker::init_graphic_map_resources()
{
	// initialize cudaResourceDesc
	memset(&m_res_desc, 0, sizeof(cudaResourceDesc));
	m_res_desc.resType = cudaResourceTypeArray;

	// initialize cudaTextureDesc
	memset(&m_tex_desc, 0, sizeof(cudaTextureDesc));
	m_tex_desc.addressMode[0] = cudaAddressModeBorder;
	m_tex_desc.addressMode[1] = cudaAddressModeBorder;
	m_tex_desc.filterMode = cudaFilterModePoint;
	m_tex_desc.readMode = cudaReadModeElementType;
	m_tex_desc.normalizedCoords = 0;

	// register textures for rigid icp
	cudaSafeCall(cudaGraphicsGLRegisterImage(&m_tex_res[0], gl_resrc_ptr->kinect_renderer.tex_live_vmap_c0, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&m_tex_res[1], gl_resrc_ptr->kinect_renderer.tex_live_nmap_c0, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&m_tex_res[2], gl_resrc_ptr->kinect_renderer.tex_live_vmap_c1, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&m_tex_res[3], gl_resrc_ptr->kinect_renderer.tex_live_nmap_c1, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	// register textures for nonrigid icp
	cudaSafeCall(cudaGraphicsGLRegisterImage(&m_tex_res_nonrigid[0], gl_resrc_ptr->kinect_renderer.tex_can_vmap_c0, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&m_tex_res_nonrigid[1], gl_resrc_ptr->kinect_renderer.tex_can_nmap_c0, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&m_tex_res_nonrigid[2], gl_resrc_ptr->kinect_renderer.tex_can_vmap_c1, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&m_tex_res_nonrigid[3], gl_resrc_ptr->kinect_renderer.tex_can_nmap_c1, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

	//register buffer obejcts
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_buffer_res[0], gl_resrc_ptr->kinect_renderer.can_vertex_buffer, cudaGraphicsRegisterFlagsWriteDiscard));
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_buffer_res[1], gl_resrc_ptr->kinect_renderer.can_normal_buffer, cudaGraphicsRegisterFlagsWriteDiscard));
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_buffer_res[2], gl_resrc_ptr->kinect_renderer.warp_vertex_buffer, cudaGraphicsRegisterFlagsWriteDiscard));
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_buffer_res[3], gl_resrc_ptr->kinect_renderer.warp_normal_buffer, cudaGraphicsRegisterFlagsWriteDiscard));
}

std::tuple<cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t> Worker::map_rigid_icp_texobj()
{
	// map graphics resource
	cudaSafeCall(cudaGraphicsMapResources(4, m_tex_res));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&m_live_vmap_array_dv_c0, m_tex_res[0], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&m_live_nmap_array_dv_c0, m_tex_res[1], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&m_live_vmap_array_dv_c1, m_tex_res[2], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&m_live_nmap_array_dv_c1, m_tex_res[3], 0, 0));

	// create texture objects
	m_res_desc.res.array.array = m_live_vmap_array_dv_c0;
	cudaSafeCall(cudaCreateTextureObject(&m_live_vmap_texobj_dv_c0, &m_res_desc, &m_tex_desc, NULL));
	m_res_desc.res.array.array = m_live_nmap_array_dv_c0;
	cudaSafeCall(cudaCreateTextureObject(&m_live_nmap_texobj_dv_c0, &m_res_desc, &m_tex_desc, NULL));
	m_res_desc.res.array.array = m_live_vmap_array_dv_c1;
	cudaSafeCall(cudaCreateTextureObject(&m_live_vmap_texobj_dv_c1, &m_res_desc, &m_tex_desc, NULL));
	m_res_desc.res.array.array = m_live_nmap_array_dv_c1;
	cudaSafeCall(cudaCreateTextureObject(&m_live_nmap_texobj_dv_c1, &m_res_desc, &m_tex_desc, NULL));

	return std::tuple<cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t>
	{m_live_vmap_texobj_dv_c0, m_live_nmap_texobj_dv_c0, m_live_vmap_texobj_dv_c1, m_live_nmap_texobj_dv_c1};
}

void Worker::unmap_rigid_icp_texobj()
{
	// destroy texture objects
	cudaSafeCall(cudaDestroyTextureObject(m_live_vmap_texobj_dv_c0));
	cudaSafeCall(cudaDestroyTextureObject(m_live_nmap_texobj_dv_c0));
	cudaSafeCall(cudaDestroyTextureObject(m_live_vmap_texobj_dv_c1));
	cudaSafeCall(cudaDestroyTextureObject(m_live_nmap_texobj_dv_c1));

	// unmap cuda resources
	cudaSafeCall(cudaGraphicsUnmapResources(4, m_tex_res));
}

std::tuple<cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t> Worker::map_nonrigid_icp_texobj()
{
	// map graphics resource
	cudaSafeCall(cudaGraphicsMapResources(4, m_tex_res_nonrigid));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&m_can_vmap_array_dv_c0, m_tex_res_nonrigid[0], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&m_can_nmap_array_dv_c0, m_tex_res_nonrigid[1], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&m_can_vmap_array_dv_c1, m_tex_res_nonrigid[2], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&m_can_nmap_array_dv_c1, m_tex_res_nonrigid[3], 0, 0));

	// create texture objects
	m_res_desc.res.array.array = m_can_vmap_array_dv_c0;
	cudaSafeCall(cudaCreateTextureObject(&m_can_vmap_texobj_dv_c0, &m_res_desc, &m_tex_desc, NULL));
	m_res_desc.res.array.array = m_can_nmap_array_dv_c0;
	cudaSafeCall(cudaCreateTextureObject(&m_can_nmap_texobj_dv_c0, &m_res_desc, &m_tex_desc, NULL));
	m_res_desc.res.array.array = m_can_vmap_array_dv_c1;
	cudaSafeCall(cudaCreateTextureObject(&m_can_vmap_texobj_dv_c1, &m_res_desc, &m_tex_desc, NULL));
	m_res_desc.res.array.array = m_can_nmap_array_dv_c1;
	cudaSafeCall(cudaCreateTextureObject(&m_can_nmap_texobj_dv_c1, &m_res_desc, &m_tex_desc, NULL));

	return std::tuple<cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t, cudaTextureObject_t>
	{m_can_vmap_texobj_dv_c0, m_can_nmap_texobj_dv_c0, m_can_vmap_texobj_dv_c1, m_can_nmap_texobj_dv_c1};
}

void Worker::unmap_nonrigid_icp_texobj()
{
	// destroy texture objects
	cudaSafeCall(cudaDestroyTextureObject(m_can_vmap_texobj_dv_c0));
	cudaSafeCall(cudaDestroyTextureObject(m_can_nmap_texobj_dv_c0));
	cudaSafeCall(cudaDestroyTextureObject(m_can_vmap_texobj_dv_c1));
	cudaSafeCall(cudaDestroyTextureObject(m_can_nmap_texobj_dv_c1));

	// unmap cuda resources
	cudaSafeCall(cudaGraphicsUnmapResources(4, m_tex_res_nonrigid));
}

std::tuple<	pcl::gpu::DeviceArray<float4>,
	pcl::gpu::DeviceArray<float4>,
	pcl::gpu::DeviceArray<float4>,
	pcl::gpu::DeviceArray<float4>> Worker::map_vertex_attributes_to_CUDA()
{
	// map resource of vertex attributes
	cudaSafeCall(cudaGraphicsMapResources(4, m_buffer_res));

	// get pointers to vertex attribute buffers
	void *can_vertices_dptr;
	void *can_normals_dptr;
	void *warp_vertices_dptr;
	void *warp_normals_dptr;

	size_t can_vertices_buffer_size;
	size_t can_normals_buffer_size;
	size_t warp_vertices_buffer_size;
	size_t warp_normals_buffer_size;

	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&can_vertices_dptr, &can_vertices_buffer_size, m_buffer_res[0]));
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&can_normals_dptr, &can_normals_buffer_size, m_buffer_res[1]));
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&warp_vertices_dptr, &warp_vertices_buffer_size, m_buffer_res[2]));
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&warp_normals_dptr, &warp_normals_buffer_size, m_buffer_res[3]));

	// unmap resource of vertex attributes
	cudaSafeCall(cudaGraphicsUnmapResources(4, m_buffer_res));

	return std::tuple<pcl::gpu::DeviceArray<float4>,
		pcl::gpu::DeviceArray<float4>,
		pcl::gpu::DeviceArray<float4>,
		pcl::gpu::DeviceArray<float4>>
	{pcl::gpu::DeviceArray<float4>((float4*)can_vertices_dptr, can_vertices_buffer_size / sizeof(float4)),
		pcl::gpu::DeviceArray<float4>((float4*)can_normals_dptr, can_normals_buffer_size / sizeof(float4)),
		pcl::gpu::DeviceArray<float4>((float4*)warp_vertices_dptr, warp_vertices_buffer_size / sizeof(float4)),
		pcl::gpu::DeviceArray<float4>((float4*)warp_normals_dptr, warp_normals_buffer_size / sizeof(float4))};
}

void Worker::map_vertex_attributes()
{
	// map resource of vertex attributes
	cudaSafeCall(cudaGraphicsMapResources(4, m_buffer_res));
}

void Worker::unmap_vertex_attributes()
{
	// unmap resource of vertex attributes
	cudaSafeCall(cudaGraphicsUnmapResources(4, m_buffer_res));
}

void Worker::cleanup_graphic_resources() {

	// unregister textures
	cudaSafeCall(cudaGraphicsUnregisterResource(m_tex_res[0]));
	cudaSafeCall(cudaGraphicsUnregisterResource(m_tex_res[1]));
	cudaSafeCall(cudaGraphicsUnregisterResource(m_tex_res[2]));
	cudaSafeCall(cudaGraphicsUnregisterResource(m_tex_res[3]));
	cudaSafeCall(cudaGraphicsUnregisterResource(m_tex_res_nonrigid[0]));
	cudaSafeCall(cudaGraphicsUnregisterResource(m_tex_res_nonrigid[1]));
	cudaSafeCall(cudaGraphicsUnregisterResource(m_tex_res_nonrigid[2]));
	cudaSafeCall(cudaGraphicsUnregisterResource(m_tex_res_nonrigid[3]));

	// unregister input buffers
	cudaSafeCall(cudaGraphicsUnregisterResource(m_buffer_res[0]));
	cudaSafeCall(cudaGraphicsUnregisterResource(m_buffer_res[1]));
	cudaSafeCall(cudaGraphicsUnregisterResource(m_buffer_res[2]));
	cudaSafeCall(cudaGraphicsUnregisterResource(m_buffer_res[3]));

	delete sensor_color_texture;
	delete sensor_depth_texture;
	delete camera_depth_texture;
	E_fitting.cleanup();
}

Worker::~Worker() {
	delete trivial_detector;
	delete handfinder;
	delete model;
}

void Worker::load_2D_keypoint_GT(std::string file_path)
{
	cout << "loading keypoints GT" << endl;
	std::ifstream in(file_path); //Keypoint_2D_GT_path + "/left_2D_keypoints.txt"
	if (!in.is_open()) {
		cout << "cannot open keypoints GT file" << endl;
		exit(0);
	}

	int max_num_frames = 5000;
	int num_keypoints = 21;
	//resize the 2D keypoint ground truth vector 
	Keypoint_2D_GT_vec_left.resize(max_num_frames);

	///--- Read in the matrix
	int row = 0;
	for (std::string line; std::getline(in, line)&&row<max_num_frames; ) {
		stringstream str(line);
		std::string elem;
		str >> elem;
		row = std::stoi(elem);

		std::vector<float2> keypoint_2D_frame;
		keypoint_2D_frame.resize(num_keypoints);
		for (int col = 0; col < num_keypoints; ++col) {
			float2 pixel_pos;
			str >> elem;
			pixel_pos.x = std::stof(elem);
			str >> elem;
			pixel_pos.y = std::stof(elem);
			keypoint_2D_frame[col] = pixel_pos;
		}

		Keypoint_2D_GT_vec_left[row] = keypoint_2D_frame;
	}

	in.close();
}

void Worker::load_3D_keypoint_GT(std::string file_path)
{
	cout << "loading 3D keypoint GT" << endl;
	std::ifstream in(file_path);//Keypoint_2D_GT_path + "/left_3D_keypoints.txt"
	if (!in.is_open()) {
		cout << "cannot open 3D keypoint GT file" << endl;
		exit(0);
	}

	int max_num_frames = 5000;
	int num_keypoints = 21;
	//resize the 2D keypoint ground truth vector 
	Keypoint_3D_GT_vec_left.resize(max_num_frames);

	///--- Read in the matrix
	int row = 0;
	for (std::string line; std::getline(in, line) && row < max_num_frames; ) {
		stringstream str(line);
		std::string elem;
		str >> elem;
		row = std::stoi(elem);

		std::vector<float3> keypoint_3D_frame;
		keypoint_3D_frame.resize(num_keypoints);
		for (int col = 0; col < num_keypoints; ++col) {
			float3 keypoint_pos;
			str >> elem;
			keypoint_pos.x = std::stof(elem);
			str >> elem;
			keypoint_pos.y = std::stof(elem);
			str >> elem;
			keypoint_pos.z = std::stof(elem);
			keypoint_3D_frame[col] = keypoint_pos;
		}

		Keypoint_3D_GT_vec_left[row] = keypoint_3D_frame;
	}

	in.close();
}

void Worker::load_3D_keypoint_combine(std::string file_path)
{
	cout << "loading 3D keypoint combine" << endl;
	std::ifstream in(file_path);//Keypoint_2D_GT_path + "/left_3D_keypoints.txt"
	if (!in.is_open()) {
		cout << "cannot open 3D keypoint combine file" << endl;
		exit(0);
	}

	int max_num_frames = 5000;
	int num_keypoints = 21;
	//resize the 2D keypoint ground truth vector 
	Keypoint_3D_combine_vec_left.resize(max_num_frames);

	///--- Read in the matrix
	int row = 0;
	for (std::string line; std::getline(in, line) && row < max_num_frames; ) {
		stringstream str(line);
		std::string elem;
		str >> elem;
		row = std::stoi(elem);

		std::vector<float3> keypoint_3D_frame;
		keypoint_3D_frame.resize(num_keypoints);
		for (int col = 0; col < num_keypoints; ++col) {
			float3 keypoint_pos;
			str >> elem;
			keypoint_pos.x = std::stof(elem);
			str >> elem;
			keypoint_pos.y = std::stof(elem);
			str >> elem;
			keypoint_pos.z = std::stof(elem);
			keypoint_3D_frame[col] = keypoint_pos;
		}

		Keypoint_3D_combine_vec_left[row] = keypoint_3D_frame;
	}

	in.close();
}

void Worker::load_3D_keypoint_visible(std::string file_path)
{
	cout << "loading 3D keypoint visible" << endl;
	std::ifstream in(file_path);//Keypoint_2D_GT_path + "/left_3D_keypoints.txt"
	if (!in.is_open()) {
		cout << "cannot open 3D keypoint visible file" << endl;
		exit(0);
	}

	int max_num_frames = 5000;
	int num_keypoints = 21;
	//resize the 2D keypoint ground truth vector 
	Keypoint_3D_visible_vec_left.resize(max_num_frames);

	///--- Read in the matrix
	int row = 0;
	for (std::string line; std::getline(in, line) && row < max_num_frames; ) {
		stringstream str(line);
		std::string elem;
		str >> elem;
		row = std::stoi(elem);

		std::vector<float> keypoint_3D_frame;
		keypoint_3D_frame.resize(num_keypoints);
		for (int col = 0; col < num_keypoints; ++col) {
			str >> elem;
			keypoint_3D_frame[col] = std::stof(elem);
		}

		Keypoint_3D_visible_vec_left[row] = keypoint_3D_frame;
	}

	in.close();
}

void Worker::track(int iter,
	cv::Mat& silhouette_c0,
	std::vector<int>& real_ADT_c0,
	Interaction& interaction_data,
	std::vector<float3> joints_pred,
	std::vector<float2> joints_pred_2D,
	bool time_test) {
	bool eval_error = (iter == settings->termination_max_iters - 1);
	bool rigid_only = (iter < settings->termination_max_rigid_iters);
	bool count_data_points = (iter == settings->termination_max_iters - 1);

	std::vector<float> _thetas = model->get_theta();
	bool set_parameter = (iter == 0);

	float max_iters = settings->termination_max_iters;
	float weight_factor = (max_iters - iter) / max_iters;
	//std::cout << weight_factor << std::endl;

	///--- Serialize matrices for jacobian computation
	model->serializer.serialize_model();
	//model->compute_rendered_indicator(handfinder->sensor_silhouette, camera);

	///--- Optimization phases	
	LinearSystem system(num_thetas);
	//clear the sys
	system.lhs = Matrix_MxN::Zero(num_thetas, num_thetas);
	system.rhs = VectorN::Zero(num_thetas);

	//eval_error = true;
	kernel_set_camera_para(para_camera0.fx / 2, para_camera0.fy / 2);

	if (frame_id < start_frame_with_only_2D_keypoint)
	{
		if (time_test)
		{
			nvprofiler::start("hand track by c0", frame_id);
		}

		E_fitting.track(current_frame, system, rigid_only, eval_error,
			tracking_error.push_error, tracking_error.pull_error,
			iter, frame_id, silhouette_c0, real_ADT_c0, para_camera0, view_camera0, count_data_points);///<!!! MUST BE FIRST CALL	
		if (time_test)
		{
			nvprofiler::stop();
		}

		if (time_test)
		{
			nvprofiler::start("hand track by interaction", frame_id);
		}
		LinearSystem system_interaction(num_thetas);
		if (!run_surface2surface)
			E_interaction.track_joints(interaction_data, system_interaction, rigid_motion, set_parameter, eval_error, frame_id);
		else
			E_interaction.track_blocks(interaction_data, system_interaction, rigid_motion, set_parameter, eval_error, frame_id);
		{
			system.lhs += E_interaction.InteractionHand_weight * system_interaction.lhs;//0.5 (20200522) 1.0
			system.rhs += E_interaction.InteractionHand_weight * system_interaction.rhs;
		}
		if (time_test)
		{
			nvprofiler::stop();
		}

		E_collision.track(system, eval_error, tracking_error.collision_error);
		E_temporal.track(system, current_frame, eval_error, tracking_error.first_order_temporal_error, tracking_error.second_order_temporal_error);

		if (frame_id > reconstruction_frame)
			E_poseprediction.track(system, _thetas, predicted_pose);

		E_limits.track(system, _thetas, eval_error, tracking_error.limit_error);
		E_damping.track_full(system);
	}

	Eigen::Matrix4f camera_pose;
	camera_pose = pose_camera0;

	if (is_using_2D_keypoint)
	{
		E_KeyPoint3D.track(system, model, Keypoint_block, Keypoint2SphereCenter, joints_pred, using_keypoint_2D, camera_pose, frame_id, iter, weight_factor); //Keypoint_3D_GT_vec_left[frame_id]  weight_factor
	}

	if (frame_id < start_frame_with_only_2D_keypoint)
	{
		if (rigid_only)
			energy::Energy::rigid_only(system);
		else
			E_pose.track(system, _thetas); ///<!!! MUST BE LAST CALL	
	}

	///--- Solve 
	VectorN delta_thetas = energy::Energy::solve(system);

	///--- Update
	const vector<float> dt(delta_thetas.data(), delta_thetas.data() + num_thetas);
	_thetas = model->get_updated_parameters(_thetas, dt);

	if (iter == settings->termination_max_iters - 1)
	{
		std::vector<float> _thetas_input;
		for (int i = 7; i < 29; i++)
		{
			_thetas_input.push_back(_thetas[i]);
		}
		std::vector<float> _thetas_out = OEFilter.OneEuroFilter_run(_thetas_input);

		for (int i = 7; i < 29; i++)
		{
			_thetas[i] = _thetas_out[i - 7];
		}
	}

	model->move(_thetas);
	model->update_centers();
	model->compute_outline();
	E_temporal.update(current_frame.id, _thetas);
}

void Worker::track(int iter,
	cv::Mat& silhouette_c0, cv::Mat& silhouette_c1,
	std::vector<int>& real_ADT_c0, std::vector<int>& real_ADT_c1,
	Interaction& interaction_data,
	std::vector<float3> joints_pred,
	std::vector<float2> joints_pred_2D,
	bool time_test) {
	bool eval_error = (iter == settings->termination_max_iters - 1);
	bool rigid_only = (iter < settings->termination_max_rigid_iters);
	bool count_data_points = (iter == settings->termination_max_iters - 1);

	std::vector<float> _thetas = model->get_theta();
	bool set_parameter = (iter == 0);

	float max_iters = settings->termination_max_iters;
	float weight_factor = (max_iters - iter) / max_iters;
	//std::cout << weight_factor << std::endl;

	///--- Serialize matrices for jacobian computation
	model->serializer.serialize_model();
	//model->compute_rendered_indicator(handfinder->sensor_silhouette, camera);

	///--- Optimization phases	
	LinearSystem system(num_thetas);
	//clear the sys
	system.lhs = Matrix_MxN::Zero(num_thetas, num_thetas);
	system.rhs = VectorN::Zero(num_thetas);

	LinearSystem system2(num_thetas);
	//clear the sys
	system2.lhs = Matrix_MxN::Zero(num_thetas, num_thetas);
	system2.rhs = VectorN::Zero(num_thetas);

	//eval_error = true;
	kernel_set_camera_para(para_camera0.fx / 2, para_camera0.fy / 2);

	bool _use_fit_term = true;
	bool _use_reg_term = true;
	bool _use_pred_term = true;

	if (_use_fit_term)
	{
		if (time_test)
		{
			nvprofiler::start("hand track by c0", frame_id);
		}

		E_fitting.track(current_frame, system, rigid_only, eval_error,
			tracking_error.push_error, tracking_error.pull_error,
			iter, frame_id, silhouette_c0, real_ADT_c0, para_camera0, view_camera0, count_data_points);///<!!! MUST BE FIRST CALL	
		if (time_test)
		{
			nvprofiler::stop();
		}

		if (time_test)
		{
			nvprofiler::start("hand track by c1", frame_id);
		}
		E_fitting.track3dOnly(current_frame, system2, rigid_only, eval_error, tracking_error.push_error, tracking_error.pull_error, iter);
		{
			system.lhs += 1.0 * system2.lhs;
			system.rhs += 1.0 * system2.rhs;
		}
		if (time_test)
		{
			nvprofiler::stop();
		}
	}
	if (_use_pred_term)
	{
		Eigen::Matrix4f camera_pose;
		camera_pose = pose_camera0;
		if (is_using_2D_keypoint)
		{
			E_KeyPoint3D.track(system, model, Keypoint_block, Keypoint2SphereCenter, joints_pred, using_keypoint_2D, camera_pose, frame_id, iter, weight_factor); //Keypoint_3D_GT_vec_left[frame_id]  weight_factor
		}
		if (frame_id > reconstruction_frame)
		{
			E_poseprediction.track(system, _thetas, predicted_pose);
		}
	}
	if (_use_reg_term)
	{
		if (time_test)
		{
			nvprofiler::start("hand track by interaction", frame_id);
		}
		LinearSystem system_interaction(num_thetas);
		if (!run_surface2surface)
		{
			E_interaction.track_joints(interaction_data, system_interaction, rigid_motion, set_parameter, eval_error, frame_id);
		}
		else
		{
			E_interaction.track_blocks(interaction_data, system_interaction, rigid_motion, set_parameter, eval_error, frame_id);
		}
		system.lhs += E_interaction.InteractionHand_weight * system_interaction.lhs;//0.5 (20200522) 1.0
		system.rhs += E_interaction.InteractionHand_weight * system_interaction.rhs;
		if (time_test)
		{
			nvprofiler::stop();
		}

		E_limits.track(system, _thetas, eval_error, tracking_error.limit_error);
		E_collision.track(system, eval_error, tracking_error.collision_error);
		E_temporal.track(system, current_frame, eval_error, tracking_error.first_order_temporal_error, tracking_error.second_order_temporal_error);
	}

	E_damping.track_full(system);

	if (frame_id < start_frame_with_only_2D_keypoint)
	{
		if (rigid_only)
			energy::Energy::rigid_only(system);
		else
			E_pose.track(system, _thetas); ///<!!! MUST BE LAST CALL	
	}

	///--- Solve 
	VectorN delta_thetas = energy::Energy::solve(system);

	///--- Update
	const vector<float> dt(delta_thetas.data(), delta_thetas.data() + num_thetas);
	_thetas = model->get_updated_parameters(_thetas, dt);

	if (iter == settings->termination_max_iters - 1)
	{
		std::vector<float> _thetas_input;
		for (int i = 7; i < 29; i++)
		{
			_thetas_input.push_back(_thetas[i]);
		}
		std::vector<float> _thetas_out = OEFilter.OneEuroFilter_run(_thetas_input);

		for (int i = 7; i < 29; i++)
		{
			_thetas[i] = _thetas_out[i - 7];
		}
	}

	model->move(_thetas);
	model->update_centers();
	model->compute_outline();
	E_temporal.update(current_frame.id, _thetas);
}

bool Worker::track_till_convergence(cv::Mat& silhouette_c0,
									std::vector<int>& real_ADT_c0,
									Interaction& interaction_data, 
									Eigen::Matrix4f rigid_mot, 
									std::vector<float3> joints_pred, 
									std::vector<float2> joints_pred_2D, 
									bool time_test) {
	if (frame_id > reconstruction_frame + 10)
		E_fitting.settings->fit3D_exclude_outlier_enable = true;

	rigid_motion = rigid_mot;

	const std::vector<float> old_theta = model->get_theta();
	for (int i = 0; i < settings->termination_max_iters; ++i) 
	{
		track(i, silhouette_c0, real_ADT_c0, interaction_data, joints_pred, joints_pred_2D, time_test);
	}

	// calculate confidence based on the count of corresponding data points of each joint
	update_confidence();

	return monitor.is_failure_frame(tracking_error.pull_error, tracking_error.push_error, E_fitting.settings->fit2D_enable);
}

bool Worker::track_till_convergence(cv::Mat& silhouette_c0, cv::Mat& silhouette_c1,
									std::vector<int>& real_ADT_c0, std::vector<int>& real_ADT_c1,
									Interaction& interaction_data,
									Eigen::Matrix4f rigid_mot,
									std::vector<float3> joints_pred,
									std::vector<float2> joints_pred_2D,
									bool time_test) {
	if (frame_id > reconstruction_frame + 10)
		E_fitting.settings->fit3D_exclude_outlier_enable = true;

	rigid_motion = rigid_mot;

	const std::vector<float> old_theta = model->get_theta();
	for (int i = 0; i < settings->termination_max_iters; ++i) 
	{
		track(i, silhouette_c0, silhouette_c1, real_ADT_c0, real_ADT_c1, interaction_data, joints_pred, joints_pred_2D, time_test);
	}

	// calculate confidence based on the count of corresponding data points of each joint
	// update_confidence();

	return monitor.is_failure_frame(tracking_error.pull_error, tracking_error.push_error, E_fitting.settings->fit2D_enable);
}

void Worker::track_second_stage(const std::vector<Eigen::Vector3f>& tips_joint_pos, Interaction& interaction_data)
{
	int iter_times = settings->second_stage_iter_times;
	const std::vector<float> kin_theta = model->get_theta();
	std::vector<float> _thetas = model->get_theta();

	tar_tip_point.resize(5);
	for (int i = 0; i < 5; i++)
	{
		tar_tip_point[i] = make_float3(tips_joint_pos[i].x(), tips_joint_pos[i].y(), tips_joint_pos[i].z());
	}

	float limit_err;

	// ofstream e_ofs("../../../../result/energy_debug.txt", ios_base::app);

	for (int i = 0; i < iter_times; i++)
	{
		///--- Optimization phases	
		LinearSystem system(num_thetas);
		//clear the sys
		system.lhs = Matrix_MxN::Zero(num_thetas, num_thetas);
		system.rhs = VectorN::Zero(num_thetas);

		E_KeyPointTips.track(system, model, Keypoint_block, Keypoint2SphereCenter, tips_joint_pos, using_keypoint_tips);
		E_PhysKinDiff.track(system, _thetas, kin_theta);
		float err;
		E_limits_second.track(system, _thetas, false, err);
		E_collision_second.track(system, false, err);
		E_pose_second.track(system, _thetas);

		/// -- Solve
		VectorN delta_thetas = energy::Energy::solve(system);

		///--- Update
		const vector<float> dt(delta_thetas.data(), delta_thetas.data() + num_thetas);
		_thetas = model->get_updated_parameters(_thetas, dt);

		model->move(_thetas);
		model->update_centers();

		//{
		//	float max_diff = 0;
		//	for (int i = 0; i < 5; i++)
		//	{
		//		int center_id = model->centers_name_to_id_map[JOINT_NAME[TIPS_JOINT_IDX[i]]];
		//		auto c = model->centers[center_id];
		//		auto t = tar_tip_point[i];
		//		auto c1 = Eigen::Vector3f(c.x, c.y, c.z);
		//		auto t1 = Eigen::Vector3f(t.x, t.y, t.z);
		//		max_diff = std::max(max_diff, (c1 - t1).norm());
		//	}
		//	e_ofs << max_diff << ' ';
		//}
	}

	//e_ofs << endl;
	//e_ofs.close();

	E_temporal.update(current_frame.id, _thetas);
	model->compute_outline();
}

void Worker::update_confidence()
{
	vector<int> joint_conf(model->centers.size());
	kernel_get_conf(&(joint_conf[0]));

	for (int i = 0; i < 5; i++)
	{
		std::string joint_name = JOINT_NAME[TIPS_JOINT_IDX[i]];
		int center_idx = model->centers_name_to_id_map[joint_name];
		double rel_conf = (1.0 * joint_conf[TIPS_TO_JOINT_CONF_ID[i]]) / joint_conf[24];		// idx 24 is root joint
		double dep_ratio = abs(model->centers[center_idx].z / model->get_theta()[2]);
		rel_conf *= dep_ratio;
		rel_conf /= TIPS_CONF_REF[i];
		tips_rel_conf[i] = min(1.0, rel_conf);
		tips_org_conf[i] = joint_conf[TIPS_TO_JOINT_CONF_ID[i]];
		// std::cout << joint_name << ": " << tips_rel_conf[i] << std::endl;
	}
}

std::vector<float3> Worker::get_hand_keypoints()
{
	std::vector<float3> hand_keypoints;

	glm::vec3 center_temp;

	const std::vector<int> keypoints_id = { 16,17,18,19, 25, 12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3 };

	for (int i = 0; i < keypoints_id.size(); i++)
	{
		int id = keypoints_id[i];
		center_temp = model->centers[id];
		hand_keypoints.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}

	return hand_keypoints;
}

std::vector<float3> Worker::get_hand_skeleton()
{
	std::vector<float3> hand_skeleton;

	glm::vec3 center_temp;

	const std::vector<int> skeleton_id = { 16,17, 17,18, 18,19, 19,25,  12,13, 13,14, 14,15, 15,25,  8,9, 9,10, 10,11, 11,25,  4,5, 5,6, 6,7, 7,25,  0,1, 1,2, 2,3, 3,25 };

	for (int i = 0; i < skeleton_id.size(); i++)
	{
		int id = skeleton_id[i];
		center_temp = model->centers[id];
		hand_skeleton.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}

	return hand_skeleton;

}

std::vector<float3> Worker::get_pred_skeleton()
{
	std::vector<float3> hand_skeleton;

	glm::vec3 center_temp;

	const std::vector<int> skeleton_id = { 0,1, 1,2, 2,3, 3,4,  5,6, 6,7, 7,8, 8,4,  9,10, 10,11, 11,12, 12,4,  13,14, 14,15, 15,16, 16,4,  17,18, 18,19, 19,20, 20,4 };
	if (Keypoint_3D_pred.size() == 21)
	{
		for (int i = 0; i < skeleton_id.size(); i++)
		{
			int id = skeleton_id[i];
			float3 keypoint = Keypoint_3D_pred[id];
			hand_skeleton.push_back(make_float3(keypoint.x, -keypoint.y, keypoint.z));
		}
	}

	return hand_skeleton;

}

std::vector<float3> Worker::get_hand_keypoints_color()
{
	std::vector<float3> hand_skeleton_color;

	/*const std::vector<int> skeleton_id = { 16,17, 17,18, 18,19, 19,25,  12,13, 13,14, 14,15, 15,25,  8,9, 9,10, 10,11, 11,25,  4,5, 5,6, 6,7, 7,25,  0,1, 1,2, 2,3, 3,25 };

	std::vector<float3> hand_keypoints_color;

	hand_keypoints_color.resize(38);

	float3 base_color = make_float3(0.5, 0.5, 0.5);

	for (int i = 0; i < hand_keypoints_color.size(); i++)
		hand_keypoints_color[i] = base_color;

	hand_keypoints_color[25] = make_float3(0.5, 0.5, 0.5);

	float3 thumb_tip_color = make_float3(1.0, 0.0, 0.0);
	hand_keypoints_color[16] = base_color + 4 / 4.0*(thumb_tip_color - base_color);
	hand_keypoints_color[17] = base_color + 3 / 4.0*(thumb_tip_color - base_color);
	hand_keypoints_color[18] = base_color + 2 / 4.0*(thumb_tip_color - base_color);
	hand_keypoints_color[19] = base_color + 1 / 4.0*(thumb_tip_color - base_color);

	float3 index_tip_color = make_float3(1.0, 1.0, 0.0);
	hand_keypoints_color[12] = base_color + 4 / 4.0*(index_tip_color - base_color);
	hand_keypoints_color[13] = base_color + 3 / 4.0*(index_tip_color - base_color);
	hand_keypoints_color[14] = base_color + 2 / 4.0*(index_tip_color - base_color);
	hand_keypoints_color[15] = base_color + 1 / 4.0*(index_tip_color - base_color);

	float3 middle_tip_color = make_float3(0.0, 1.0, 0.0);
	hand_keypoints_color[8] = base_color + 4 / 4.0*(middle_tip_color - base_color);
	hand_keypoints_color[9] = base_color + 3 / 4.0*(middle_tip_color - base_color);
	hand_keypoints_color[10] = base_color + 2 / 4.0*(middle_tip_color - base_color);
	hand_keypoints_color[11] = base_color + 1 / 4.0*(middle_tip_color - base_color);

	float3 ring_tip_color = make_float3(0.0, 1.0, 1.0);
	hand_keypoints_color[4] = base_color + 4 / 4.0*(ring_tip_color - base_color);
	hand_keypoints_color[5] = base_color + 3 / 4.0*(ring_tip_color - base_color);
	hand_keypoints_color[6] = base_color + 2 / 4.0*(ring_tip_color - base_color);
	hand_keypoints_color[7] = base_color + 1 / 4.0*(ring_tip_color - base_color);

	float3 little_tip_color = make_float3(0.0, 0.0, 1.0);
	hand_keypoints_color[0] = base_color + 4 / 4.0*(little_tip_color - base_color);
	hand_keypoints_color[1] = base_color + 3 / 4.0*(little_tip_color - base_color);
	hand_keypoints_color[2] = base_color + 2 / 4.0*(little_tip_color - base_color);
	hand_keypoints_color[3] = base_color + 1 / 4.0*(little_tip_color - base_color);

	for (int i = 0; i < skeleton_id.size(); i++)
	{
		int id = skeleton_id[i];
		hand_skeleton_color.push_back(hand_keypoints_color[id]);
	}*/

	const std::vector<int> skeleton_id = { 0,1, 1,2, 2,3, 3,4,  5,6, 6,7, 7,8, 8,4,  9,10, 10,11, 11,12, 12,4,  13,14, 14,15, 15,16, 16,4,  17,18, 18,19, 19,20, 20,4 };

	std::vector<float3> hand_keypoints_color;

	hand_keypoints_color.resize(21);

	float3 base_color = make_float3(0.5, 0.5, 0.5);

	for (int i = 0; i < hand_keypoints_color.size(); i++)
		hand_keypoints_color[i] = base_color;

	hand_keypoints_color[4] = make_float3(0.5, 0.5, 0.5);

	float3 thumb_tip_color = make_float3(1.0, 0.0, 0.0);
	hand_keypoints_color[0] = base_color + 4 / 4.0*(thumb_tip_color - base_color);
	hand_keypoints_color[1] = base_color + 3 / 4.0*(thumb_tip_color - base_color);
	hand_keypoints_color[2] = base_color + 2 / 4.0*(thumb_tip_color - base_color);
	hand_keypoints_color[3] = base_color + 1 / 4.0*(thumb_tip_color - base_color);

	float3 index_tip_color = make_float3(1.0, 1.0, 0.0);
	hand_keypoints_color[5] = base_color + 4 / 4.0*(index_tip_color - base_color);
	hand_keypoints_color[6] = base_color + 3 / 4.0*(index_tip_color - base_color);
	hand_keypoints_color[7] = base_color + 2 / 4.0*(index_tip_color - base_color);
	hand_keypoints_color[8] = base_color + 1 / 4.0*(index_tip_color - base_color);

	float3 middle_tip_color = make_float3(0.0, 1.0, 0.0);
	hand_keypoints_color[9] = base_color + 4 / 4.0*(middle_tip_color - base_color);
	hand_keypoints_color[10] = base_color + 3 / 4.0*(middle_tip_color - base_color);
	hand_keypoints_color[11] = base_color + 2 / 4.0*(middle_tip_color - base_color);
	hand_keypoints_color[12] = base_color + 1 / 4.0*(middle_tip_color - base_color);

	float3 ring_tip_color = make_float3(0.0, 1.0, 1.0);
	hand_keypoints_color[13] = base_color + 4 / 4.0*(ring_tip_color - base_color);
	hand_keypoints_color[14] = base_color + 3 / 4.0*(ring_tip_color - base_color);
	hand_keypoints_color[15] = base_color + 2 / 4.0*(ring_tip_color - base_color);
	hand_keypoints_color[16] = base_color + 1 / 4.0*(ring_tip_color - base_color);

	float3 little_tip_color = make_float3(0.0, 0.0, 1.0);
	hand_keypoints_color[17] = base_color + 4 / 4.0*(little_tip_color - base_color);
	hand_keypoints_color[18] = base_color + 3 / 4.0*(little_tip_color - base_color);
	hand_keypoints_color[19] = base_color + 2 / 4.0*(little_tip_color - base_color);
	hand_keypoints_color[20] = base_color + 1 / 4.0*(little_tip_color - base_color);

	for (int i = 0; i < skeleton_id.size(); i++)
	{
		int id = skeleton_id[i];
		hand_skeleton_color.push_back(hand_keypoints_color[id]);
	}

	return hand_skeleton_color;
}