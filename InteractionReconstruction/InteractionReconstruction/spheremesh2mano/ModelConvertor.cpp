#include <iostream>
#include "ModelConvertor.h"
#include "ceres/ceres.h"
#include "glog/logging.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CENTRAL;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

LARGE_INTEGER time_stmp;
double count_freq, count_interv;
double time_inter;

void ModelConvertor::init()
{
	/**************************************************************/
	/*                    load mano model                         */
	/**************************************************************/
	m_mano = std::make_unique<MANOModel>("../../../data/mano/mano_r.json");
	m_shape = Eigen::VectorXd::Zero(10);// (10);
	
	// calibrated shape params
	bool _use_calibrated_shape = true;
	if (_use_calibrated_shape)
	{
		double shape_params[10]{
			-2.61435056, -1.16743336, -2.80988378,  0.12670897, -0.08323125, 2.28185672,
			-0.05833138, -2.95105206, -3.43976417,  0.30667237
		};
		for (int i = 0; i < 10; ++i)
			m_shape[i] = shape_params[i];
	}

	m_mano->init_rest_model(m_shape);
	cout << m_shape.transpose() << endl;

	m_pose_pca = Eigen::VectorXd::Zero(POSE_PCA_NUM);// (45);
	m_global_R = Eigen::Vector3d::Zero();// ::Zero();
	m_global_T = Eigen::Vector3d::Zero();// ::Zero();

	m_para = Eigen::VectorXd::Zero(POSE_PCA_NUM + 6);


	/**************************************************************/
	/*                    load re-mesh model                      */
	/**************************************************************/
	string remesh_data_path = "../../../data/mano/MANO_remesh";
	string vt_path = "../../../data/mano/640x480/vt.txt";
	// m_handmesh.load_remesh(remesh_data_path);
	m_handmesh.load_remesh_from_mano("../../../data/mano/mano_r.json");
	// m_handmesh.load_vt(vt_path);

	/*string obj_path = "../data/right_hand_model_texture_complete_front.obj";
	m_handmesh.loadobj(obj_path);
	m_handmesh.obtain_obj_attribute_arr();*/

	string texture_path = "../../../data/mano/640x480/hand_texture.png";

	// enumerate CUDA devices
	int num_devices;
	cudaGetDeviceCount(&num_devices);
	for (int k = 0; k < num_devices; ++k) {
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, k);
		printf("device %d: %s\n", k, device_prop.name);
	}

	// choose device 0 as parallel processor
	int selected_device = 0;
	cudaSetDevice(selected_device);
	printf("Device %d is used as parallel processor for model convert.\n", selected_device);

	m_frame_id = -1;
}

void ModelConvertor::convert(std::vector<float> keypoints, int frame_id, int max_iter)
{
	// google::InitGoogleLogging(argv[0]);

	if (frame_id <= m_frame_id)
		return;
	m_frame_id = frame_id;

	QueryPerformanceFrequency(&time_stmp);
	count_freq = (double)time_stmp.QuadPart;

	/////////////////////////////////////////////////////////////////////////////////
	//  give key-points, optimize pose of mano and render texture calibrated hand  //

	// Start solving
	LONGLONG count_start_interv, count_end_interv;
	double count_interv, time_inter;

	QueryPerformanceCounter(&time_stmp);
	count_start_interv = time_stmp.QuadPart;

	/***************************************************************/
	/*     give joints and obtain pose of mano by optimization     */
	/***************************************************************/
	{
		//float joints_give[63] = { 55.4324, -16.7496, 415.845, 57.441, 54.8646, 416.996, 24.7332, 68.0331, 388.97, 24.2243, 46.2253, 376.684, 39.4676, 45.7499, 432.202, 11.7958, 85.1057, 421.805, -7.39279, 98.3863, 405.825, 12.9856, 19.1501, 440.448, -11.1645, 46.5644, 453.794, -26.4307, 64.0352, 455.377, 26.0282, 32.1956, 439.175, -1.56517, 69.0767, 441.064, -20.7589, 86.8604, 428.434, 60.7091, -6.59063, 403.43, 65.1371, 27.6248, 374.313, 56.3961, 55.1627, 355.341, 35.6899, 37.3626, 382.486, -12.604, 90.1314, 389.134, -35.7673, 74.7216, 448.935, -26.4652, 87.6472, 412.935, 51.1067, 68.7826, 346.52 };

		Eigen::VectorXd given_joints = Eigen::VectorXd::Zero(63 + POSE_PCA_NUM);
		for (int i = 0; i < 63; i++)
		{
			given_joints(i) = keypoints[i];
		}

		//std::cout << "given_joints:" << given_joints.transpose() << std::endl;

		Eigen::MatrixX3d joints_give_(21, 3);

		for (int i = 0; i < 63; i++)
			joints_give_(i / 3, i % 3) = given_joints[i];

		//mano.ExportJoints("../data/given_joints.obj", joints_give_);

		//std::cout << "start" << endl;

		/////////////////////////////////////////////////////////////////
		for (int i = 0; i < max_iter; i++)
		{
			m_GN_solver.Solve(*(m_mano.get()), m_para, given_joints);
		}

		//std::cout << "solved para:" << para.transpose() << std::endl;

		for (int i = 0; i < POSE_PCA_NUM; i++)
			m_pose_pca(i) = m_para[i];

		m_global_R(0) = m_para[POSE_PCA_NUM + 0];
		m_global_R(1) = m_para[POSE_PCA_NUM + 1];
		m_global_R(2) = m_para[POSE_PCA_NUM + 2];

		m_global_T(0) = m_para[POSE_PCA_NUM + 3];
		m_global_T(1) = m_para[POSE_PCA_NUM + 4];
		m_global_T(2) = m_para[POSE_PCA_NUM + 5];

		Eigen::MatrixX3d model = m_mano->get_posed_model(m_pose_pca, m_global_R, m_global_T);
		//mano.ExportObj2("../data/posed_mano.obj", model, mano.get_faces());

		/*Eigen::MatrixX3d posed_joints = mano.get_posed_joints(pose_pca, global_R, global_T);
		mano.ExportJoints("../data/posed_joints.obj", posed_joints);*/

		m_model_f = model.cast<float>();
	}

	// output pose para
	ofstream ofs("../../../../result/mano_pose_param.txt", std::ios_base::app);
	ofs << frame_id;
	for (int i = 0; i < m_para.rows(); ++i)
	{
		ofs << ' ' << m_para[i];
	}
	ofs << endl;
	ofs.close();

	//while (!m_render.is_watch())
	{
		m_handmesh.re_mesh(m_model_f);
	}

	Eigen::MatrixX3d posed_joints = m_mano->get_posed_joints_solver(m_para);

	QueryPerformanceCounter(&time_stmp);
	count_end_interv = time_stmp.QuadPart;
	count_interv = (double)(count_end_interv - count_start_interv);
	time_inter = count_interv * 1000 / count_freq;
	std::cout << "Sphere mesh to mano time:" << time_inter << std::endl;
	//std::cout << "optimize end" << std::endl;

	
	/*stop cuda profiler*/
	cudaDeviceSynchronize();
	cudaProfilerStop();

	//cv::waitKey();
}

mesh* ModelConvertor::getMesh()
{
	return &(this->m_handmesh);
}