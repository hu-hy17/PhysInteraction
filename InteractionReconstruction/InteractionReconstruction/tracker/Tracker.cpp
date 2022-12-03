#include "Tracker.h"
#include "tracker/Data/DumpResult.h"

Tracker::Tracker(Worker*worker, RealSenseSR300& sr300, double FPS, std::string data_path, Json::Value& m_jsonRoot, bool real_color, bool is_benchmark) 
	: worker(worker), 
	sensor_sr300(sr300), 
	hand_segmentation(320, 240),
	record_right_camera_data(true)
{
	setSingleShot(false);
	setInterval((1.0 / FPS)*1000.0);
	this->data_path = data_path;
	this->real_color = real_color;
	tw_settings->tw_add_ro(current_fps, "FPS", "group=Tracker");
	tw_settings->tw_add(initialization_enabled, "Detect ON?", "group=Tracker");
	tw_settings->tw_add(tracking_enabled, "ArtICP ON?", "group=Tracker");
	tw_settings->tw_add_ro(tracking_failed, "Tracking Lost?", "group=Tracker");

	{
		is_nonrigid = m_jsonRoot["nonrigid"].asBool();
		file_left_camera = m_jsonRoot["left_camera_file_mask"].asString();
		file_right_camera = m_jsonRoot["right_camera_file_mask"].asString();

		start_frame = m_jsonRoot["start_frame"].asInt();
		set_recon_frame = m_jsonRoot["set_recon_frame"].asBool();
		reconstruction_frame = m_jsonRoot["recon_frame"].asInt();
		stop_recon_frame = m_jsonRoot["stop_recon_frame"].asInt();
		camera_use = m_jsonRoot["camera_using"].asInt();

		show_ADT = m_jsonRoot["show_ADT"].asBool();
		show_input = m_jsonRoot["show_input"].asBool();
		store_input = m_jsonRoot["store_input"].asBool();
		input_store_path = m_jsonRoot["input_store_path"].asString();
		store_solution_hand = m_jsonRoot["store_hand_solution"].asBool();
		hand_pose_store_path = m_jsonRoot["hand_pose_store_path"].asString();
		store_seg = m_jsonRoot["store_seg"].asBool();
		show_seg = m_jsonRoot["show_seg"].asBool();
		org_data_store_path = m_jsonRoot["org_data_store_path"].asString();
		store_org_data = m_jsonRoot["store_org_data"].asBool();
		Keypoint_2D_GT_path = m_jsonRoot["2D_Keypoint_path"].asString();
		is_using_2D_keypoint = m_jsonRoot["is_using_2D_keypoint"].asBool();
		nonrigid_start_frame = m_jsonRoot["nonrigid_start"].asInt();

		if (m_jsonRoot["use_tips_2D_point"] != Json::nullValue)
		{
			use_tip_2D_key_point = m_jsonRoot["use_tips_2D_point"].asBool();
			tip_2D_key_point_path = m_jsonRoot["tips_2D_point_path"].asString();
		}

		if (m_jsonRoot["track_hand_with_phys"] != Json::nullValue)
		{
			track_hand_with_phys = m_jsonRoot["track_hand_with_phys"].asBool();
		}

		if (m_jsonRoot["output_contact_info"] != Json::nullValue)
		{
			output_contact_info = m_jsonRoot["output_contact_info"].asBool();
		}

		if(m_jsonRoot["track_with_two_cameras"] != Json::nullValue)
		{
			track_with_two_cameras = m_jsonRoot["track_with_two_cameras"].asBool();
		}

		float* data = pose_camera1.data();
		for (const Json::Value& v : m_jsonRoot["pose_camera1"])
		{
			*data++ = v.asFloat();
		}
	}

	tip_3D_keypoint_pos.resize(5);

	if (!set_recon_frame)
		reconstruction_frame = INT_MAX;
	else
		worker->reconstruction_frame = reconstruction_frame;

	// If use exist object, do not need to recontstruct it.
	bool use_exist_obj = m_jsonRoot["use_exist_obj"].asBool();
	if (use_exist_obj) {
		stop_recon_frame = 0;
	}

	string exist_obj_path = m_jsonRoot["exist_obj_path"].asString();


	worker->camera_use = camera_use;

	current_frame = start_frame;

	real_color_map = cv::Mat(cv::Size(320, 240), CV_8UC3, cv::Scalar(0, 0, 0));

	Camera *camera_para = worker->camera;

	std::vector<int3> hand_blocks;
	std::vector<glm::ivec3> hand_block_org = worker->model->blocks;
	for (int i = 0; i < hand_block_org.size(); i++)
	{
		int3 temp;
		temp.x = hand_block_org[i][0];
		temp.y = hand_block_org[i][1];
		temp.z = hand_block_org[i][2];

		hand_blocks.push_back(temp);
	}

	hand_segmentation.init_HandBlock(hand_blocks);
	hand_segmentation.init_HandKeyPointsRadius(worker->model->radii);

	camera_depth_map_res.init(worker->camera_depth_texture->texid());

	// if use tips 2D GT as evaluation, read it from exist file and store the info.
	if (use_tip_2D_key_point)
	{
		tips_name_to_idx["thumb"] = 0;
		tips_name_to_idx["index"] = 1;
		tips_name_to_idx["middle"] = 2;
		tips_name_to_idx["ring"] = 3;
		tips_name_to_idx["pinky"] = 4;
	}


	// set camera parameters
	{
		////////////////////////////////////////////////////////
		//left camera parameters
		////////////////////////////////////////////////////////
		auto d2c_data = m_jsonRoot["depth_to_color"];
		if (d2c_data == Json::nullValue || d2c_data.size() != 16) {
			cout << "(Tracker) Error: depth to color matrix is not correctly configured!" << endl;
			exit(-1);
		}
		for (int r = 0; r < 4; ++r)
		{
			for (int c = 0; c < 4; ++c)
			{
				depth2color_cl(r, c) = d2c_data[4 * r + c].asFloat();
			}
		}

		auto dep_intr_data = m_jsonRoot["depth_intrinsic"];
		auto col_intr_data = m_jsonRoot["color_intrinsic"];
		if (dep_intr_data == Json::nullValue || dep_intr_data.size() != 4) {
			cout << "(Tracker) Error: depth intrinsic params are not correctly configured!" << endl;
			exit(-1);
		}
		if (col_intr_data == Json::nullValue || col_intr_data.size() != 4) {
			cout << "(Tracker) Error: color intrinsic params are not correctly configured!" << endl;
			exit(-1);
		}
		
		depth_camera_cl.fx = dep_intr_data[0].asFloat(); 
		depth_camera_cl.fy = dep_intr_data[1].asFloat();
		depth_camera_cl.cx = dep_intr_data[2].asFloat();
		depth_camera_cl.cy = dep_intr_data[3].asFloat();
		color_camera_cl.fx = col_intr_data[0].asFloat();
		color_camera_cl.fy = col_intr_data[1].asFloat();
		color_camera_cl.cx = col_intr_data[2].asFloat();
		color_camera_cl.cy = col_intr_data[3].asFloat();

		camera_intr depth_camera_left;
		depth_camera_left = depth_camera_cl;

		////////////////////////////////////////////////////////
		//right camera parameters 619204001397
		////////////////////////////////////////////////////////
		depth2color_cr(0, 0) = 0.999994f;  depth2color_cr(0, 1) = 0.003088f;  depth2color_cr(0, 2) = -0.001246f; depth2color_cr(0, 3) = 0.025700f;
		depth2color_cr(1, 0) = -0.003080f; depth2color_cr(1, 1) = 0.999976f;  depth2color_cr(1, 2) = 0.006157f;  depth2color_cr(1, 3) = -0.000733f;
		depth2color_cr(2, 0) = 0.001265f;  depth2color_cr(2, 1) = -0.006153f; depth2color_cr(2, 2) = 0.999980f;  depth2color_cr(2, 3) = 0.003947f;
		depth2color_cr(3, 0) = 0.0f;		 depth2color_cr(3, 1) = 0.0f;       depth2color_cr(3, 2) = 0.0f;       depth2color_cr(3, 3) = 1.0f;

		depth_camera_cr.fx = 474.984f; depth_camera_cr.fy = 474.984f; depth_camera_cr.cx = 310.504f; depth_camera_cr.cy = 245.546f;//by read from device
		color_camera_cr.fx = 617.068f; color_camera_cr.fy = 617.068f; color_camera_cr.cx = 306.625f; color_camera_cr.cy = 242.423f;

		camera_intr depth_camera_right;
		depth_camera_right = depth_camera_cr;

		dep_to_col = depth2color_cl;
		col_proj = color_camera_cl;

		image_bias left_device, right_device;
		left_device.u_bias = 0; left_device.v_bias = 0;
		right_device.u_bias = 15; right_device.v_bias = -6;

		//camera's orientation and position
		left_camera_RT = Eigen::Matrix4f::Identity();
		right_camera_RT = Eigen::Matrix4f::Identity();
		{
			right_camera_RT = pose_camera1;
		}

		left_data_parser.initial_parsar(0, left_device, color_camera_cl, depth_camera_cl, depth_camera_left, depth2color_cl, left_camera_RT/*, worker->handfinder*/);
		if (track_with_two_cameras || record_right_camera_data)
		{
			right_data_parser.initial_parsar(1, right_device, color_camera_cr, depth_camera_cr, depth_camera_right, depth2color_cr, right_camera_RT/*, worker->handfinder*/);
		}

		worker->handfinder->obtain_camera_direction(left_data_parser._camera_dir, right_data_parser._camera_dir);
		worker->is_using_2D_keypoint = is_using_2D_keypoint;
		worker->Keypoint_2D_GT_path = Keypoint_2D_GT_path;

		worker->start_frame_with_only_2D_keypoint = m_jsonRoot["start_frame_only_2D_keypoint"].asInt();

		ObjectRecon.initialization(320, 240, camera2device_intr(depth_camera_left), camera2device_intr(depth_camera_right), left_camera_RT, right_camera_RT);
		rigid_solver.initialize(320, 240, depth_camera_cl, depth_camera_cr, left_camera_RT, right_camera_RT, camera_use);

		ObjectRecon.camera_use = camera_use;

		object_pose = Eigen::Matrix4f::Identity();

		//initialize the renderer 
		CamerasParameters cameras_para_worker, cameras_para_render;
		{
			cameras_para_worker.c0 = depth_camera_cl;
			cameras_para_worker.c1 = depth_camera_cr;
			cameras_para_worker.camerapose_c0 = left_camera_RT;
			cameras_para_worker.camerapose_c1 = right_camera_RT;

			cameras_para_render.c0 = color_camera_cl;
			cameras_para_render.c1 = color_camera_cr;

			cameras_para_render.c1.cx += 15;
			cameras_para_render.c1.cy -= 18;

			auto depth2color_cl_mm = depth2color_cl;
			auto depth2color_cr_mm = depth2color_cr;
			for (int r = 0; r < 3; r++)
			{
				depth2color_cl_mm(r, 3) *= 1000;
				depth2color_cr_mm(r, 3) *= 1000;
			}
			cameras_para_render.camerapose_c0 = depth2color_cl_mm.inverse();
			cameras_para_render.camerapose_c1 = right_camera_RT * depth2color_cr_mm.inverse();
		}
		worker->initialWorkerCamMatrix(320, 240, cameras_para_worker);
		worker->initialRendererCamMatrix(320, 240, cameras_para_render);

		//synchronize the constraint mode
		worker->run_surface2surface = run_surface2surface;


		//map array buffers to cuda
		std::tie(ObjectRecon.m_can_vertices_buffer,
			ObjectRecon.m_can_normals_buffer,
			ObjectRecon.m_warp_vertices_buffer,
			ObjectRecon.m_warp_normals_buffer) = worker->map_vertex_attributes_to_CUDA();

		//initial the interaction radius
		if (!run_surface2surface)
		{
			interaction_datas.initial_finger_radius(worker->model->radii);
		}
		else
		{
			//initialize the hand radius
			interaction_datas.initial_SphereHand_radius(worker->model->radii);
			//initialize the hand block
			interaction_datas.initial_finger_blocks(worker->model->blocks);

			//initialize the phalange
			interaction_datas.initial_phalange(worker->model->get_phalange_centerId());
		}

		joint_radius = worker->model->radii;

		//set the curl attribution
		//to avoid access violation
		curl_global_init(CURL_GLOBAL_ALL);

		QueryPerformanceFrequency(&time_stmp);
		count_freq = (double)time_stmp.QuadPart;
	}

	// smooth 
	obj_smooth_ratio = 0.1;			// 0.1 0.1 0.3
	hand_smooth_ratio = 0.3;		// 0.3 0.4 0.4
	hand_smooth_ratio2 = 0.5;		// 0.5 0.5 0.6
	force_smooth_ratio = 0.7;

	//initialize the parameters used in friction
	rigid_solver.initialize_for_friction(worker->model->radii, interaction_datas.max_corresp_num);

}

Tracker::~Tracker()
{
	camera_depth_map_res.cleanup();

	char filename[512] = { 0 };

	if (store_org_data)
	{
		const std::string left_camera_id = sensor_sr300.get_left_camera_id();
		const std::string right_camera_id = sensor_sr300.get_right_camera_id();
		for (int i = 0; i < left_depth_org.size(); i++)
		{
			sprintf(filename, "%s_depth_frame%04d.png", left_camera_id, i);
			cv::imwrite(org_data_store_path + filename, left_depth_org[i]);

			sprintf(filename, "%s_color_frame%04d.png", left_camera_id, i);
			cv::imwrite(org_data_store_path + filename, left_color_org[i]);

			if (track_with_two_cameras || record_right_camera_data)
			{
				sprintf(filename, "%s_depth_frame%04d.png", right_camera_id, i);
				cv::imwrite(org_data_store_path + filename, right_depth_org[i]);

				sprintf(filename, "%s_color_frame%04d.png", right_camera_id, i);
				cv::imwrite(org_data_store_path + filename, right_color_org[i]);
			}
		}

		cout << "(Tracker) Info: Output all the image successful!" << endl;
	}

	ofstream time_performance("../../../time_performance.txt");
	time_performance << "frame id    " << "time of this frame   " << "mean time" << std::endl;

	for (int i = 0; i < time_sequence.size(); i++)
	{
		time_performance << frame_id_sequence[i] << "  " << time_sequence[i] << "  " << mean_time_sequence[i] << std::endl;
	}

	time_performance.close();

	cout << "Reconstruction frame:" << reconstruction_frame << std::endl;
	cout << "stop frame:" << stop_recon_frame << std::endl;

}

void Tracker::toggle_tracking(bool on) {
	if (on == false) return;
	mode = LIVE;
	//		if (sensor->spin_wait_for_data(5) == false) LOG(INFO) << "no sensor data";
	solutions->reserve(30 * 60 * 5); // fps * sec * min
	start();
}

void Tracker::toggle_benchmark(bool on) {
	if (on == false) return;

	setInterval((1.0 / 60)*1000.0);// 10
								   //		worker->settings->termination_max_iters = 8;//8

	mode = BENCHMARK;
	start();
}
	
void Tracker::timerEvent(QTimerEvent*) {
		process_track();
		//compute_initial_transformations();
}

void Tracker::get_online_data_from_sensors()
{
	int frame_num = 0;
	sensor_sr300.poll_frames(frame_num);

	left_data_parser.set_online_org_data(sensor_sr300.get_left_depth(), sensor_sr300.get_left_color());
	if (track_with_two_cameras)
	{
		right_data_parser.set_online_org_data(sensor_sr300.get_right_depth(), sensor_sr300.get_right_color());
	}

	if (store_org_data)
	{
		left_depth_org.push_back(sensor_sr300.get_left_depth());
		left_color_org.push_back(sensor_sr300.get_left_color());

		if (track_with_two_cameras || record_right_camera_data)
		{
			right_depth_org.push_back(sensor_sr300.get_right_depth());
			right_color_org.push_back(sensor_sr300.get_right_color());
		}
	}
}

void Tracker::load_multi_view_frames2(size_t current_frame)
{
	LONGLONG count_start_load_pre, count_end_load_pre;
	double count_interv, time_inter;
	/*QueryPerformanceCounter(&time_stmp);
	count_start_load_pre = time_stmp.QuadPart;*/

	left_data_parser.load_org_data(file_left_camera, current_frame);
	if (track_with_two_cameras)
	{
		right_data_parser.load_org_data(file_right_camera, current_frame);
	}

	//QueryPerformanceCounter(&time_stmp);
	//count_end_load_pre = time_stmp.QuadPart;
	//count_interv = (double)(count_end_load_pre - count_start_load_pre);
	//time_inter = count_interv * 1000 / count_freq;
	//std::cout << "data prepare" << "	" << time_inter << std::endl;
}

void Tracker::get_predicted_pose()
{
	pose_prediction_net.get_predicted_pose(current_hand_pose, predicted_hand_pose);
}

void Tracker::data_preprocessing()
{
	LONGLONG count_start_prepro, count_end_prepro;
	double count_interv, time_inter;
	QueryPerformanceCounter(&time_stmp);
	count_start_prepro = time_stmp.QuadPart;

	printf("\n\n%d frame\n", current_frame);

	if (mode == LIVE) {
		get_online_data_from_sensors();
	} else if (mode == BENCHMARK) {
		load_multi_view_frames2(current_frame);
	}

	std::thread get_predictedpose_thr(&Tracker::get_predicted_pose, this);

	left_data_parser.obtain_resized_data2();
	left_data_parser.obtain_aligned_data2();
	left_data_parser.obtain_hand_object_silhouette();
	if (track_with_two_cameras)
	{
		right_data_parser.obtain_resized_data2();
		right_data_parser.obtain_aligned_data2();
		right_data_parser.obtain_hand_object_silhouette();
	}

	get_predictedpose_thr.join();
	//get_predicted_pose();

	QueryPerformanceCounter(&time_stmp);
	count_end_prepro = time_stmp.QuadPart;
	count_interv = (double)(count_end_prepro - count_start_prepro);
	time_process = count_interv * 1000 / count_freq;
	//std::cout << "preprocess" << "	" << time_inter << std::endl;
}

void Tracker::data_segmentation_init()
{
	left_data_parser.HandObjectSegmentation_init();
	left_data_parser.ObtainJoints();
	left_data_parser.cal_ADT();
	left_data_parser.CalculateObjADT();

	if (track_with_two_cameras)
	{
		right_data_parser.HandObjectSegmentation_init();
		right_data_parser.ObtainJoints();
		right_data_parser.cal_ADT();
		right_data_parser.CalculateObjADT();
	}
}

void Tracker::data_segmentation()
{
	LONGLONG count_start_seg, count_end_seg;
	double count_interv, time_inter;
	QueryPerformanceCounter(&time_stmp);
	count_start_seg = time_stmp.QuadPart;

	left_data_parser.HandObjectSegmentation();
	left_data_parser.ObtainJoints();

	left_data_parser.cal_ADT();
	left_data_parser.CalculateObjADT();

	if (track_with_two_cameras)
	{
		right_data_parser.HandObjectSegmentation();
		right_data_parser.ObtainJoints();

		right_data_parser.cal_ADT();
		right_data_parser.CalculateObjADT();
	}

	QueryPerformanceCounter(&time_stmp);
	count_end_seg = time_stmp.QuadPart;
	count_interv = (double)(count_end_seg - count_start_seg);
	time_seg = count_interv * 1000 / count_freq;
	//std::cout << "seg+kpt" << "	" << time_inter << std::endl;

	/*mean_time_sum_seg += time_inter;
	sum_number_seg += 1;
	std::cout << "mean seg+kpt" << "	" << mean_time_sum_seg / sum_number_seg << std::endl;*/
}

void Tracker::result_assignment_pre2seg()
{
	left_data_parser._color_org_320_seg_in = left_data_parser._color_org_320_pre.clone();
	left_data_parser._depth_org_320_seg_in = left_data_parser._depth_org_320_pre.clone();
	left_data_parser._depth_mm_320_seg_in = left_data_parser._depth_mm_320_pre.clone();
	left_data_parser._aligned_color_320_seg_in = left_data_parser._aligned_color_320_pre.clone();
	left_data_parser._hand_object_silhouette_seg_in = left_data_parser._hand_object_silhouette_pre.clone();

	left_data_parser._color_org_320_seg_out = left_data_parser._color_org_320_seg_in.clone();
	left_data_parser._depth_org_320_seg_out = left_data_parser._depth_org_320_seg_in.clone();
	left_data_parser._depth_mm_320_seg_out = left_data_parser._depth_mm_320_seg_in.clone();
	left_data_parser._aligned_color_320_seg_out = left_data_parser._aligned_color_320_seg_in.clone();
	left_data_parser._hand_object_silhouette_seg_out = left_data_parser._hand_object_silhouette_seg_in.clone();

	if (track_with_two_cameras)
	{
		right_data_parser._color_org_320_seg_in = right_data_parser._color_org_320_pre.clone();
		right_data_parser._depth_org_320_seg_in = right_data_parser._depth_org_320_pre.clone();
		right_data_parser._depth_mm_320_seg_in = right_data_parser._depth_mm_320_pre.clone();
		right_data_parser._aligned_color_320_seg_in = right_data_parser._aligned_color_320_pre.clone();
		right_data_parser._hand_object_silhouette_seg_in = right_data_parser._hand_object_silhouette_pre.clone();

		right_data_parser._color_org_320_seg_out = right_data_parser._color_org_320_seg_in.clone();
		right_data_parser._depth_org_320_seg_out = right_data_parser._depth_org_320_seg_in.clone();
		right_data_parser._depth_mm_320_seg_out = right_data_parser._depth_mm_320_seg_in.clone();
		right_data_parser._aligned_color_320_seg_out = right_data_parser._aligned_color_320_seg_in.clone();
		right_data_parser._hand_object_silhouette_seg_out = right_data_parser._hand_object_silhouette_seg_in.clone();
	}
}

void Tracker::result_assignment_seg2main()
{
	left_data_parser._color_org_320 = left_data_parser._color_org_320_seg_out.clone();
	left_data_parser._depth_org_320 = left_data_parser._depth_org_320_seg_out.clone();
	left_data_parser._depth_mm_320 = left_data_parser._depth_mm_320_seg_out.clone();
	left_data_parser._aligned_color_320 = left_data_parser._aligned_color_320_seg_out.clone();
	left_data_parser._hand_object_silhouette = left_data_parser._hand_object_silhouette_seg_out.clone();
	left_data_parser._segmentation_org = left_data_parser._segmentation_org_seg.clone();
	left_data_parser._hand_silhouette = left_data_parser._hand_silhouette_seg.clone();
	left_data_parser._object_silhouette = left_data_parser._object_silhouette_seg.clone();
	left_data_parser._joints_pred_xyz = left_data_parser._joints_pred_xyz_seg;
	left_data_parser._joints_pred_uv = left_data_parser._joints_pred_uv_seg;
	left_data_parser.realADT = left_data_parser.realADT_seg;
	left_data_parser.realADT_obj = left_data_parser.realADT_obj_seg;

	if (track_with_two_cameras)
	{
		right_data_parser._color_org_320 = right_data_parser._color_org_320_seg_out.clone();
		right_data_parser._depth_org_320 = right_data_parser._depth_org_320_seg_out.clone();
		right_data_parser._depth_mm_320 = right_data_parser._depth_mm_320_seg_out.clone();
		right_data_parser._aligned_color_320 = right_data_parser._aligned_color_320_seg_out.clone();
		right_data_parser._hand_object_silhouette = right_data_parser._hand_object_silhouette_seg_out.clone();
		right_data_parser._segmentation_org = right_data_parser._segmentation_org_seg.clone();
		right_data_parser._hand_silhouette = right_data_parser._hand_silhouette_seg.clone();
		right_data_parser._object_silhouette = right_data_parser._object_silhouette_seg.clone();
		right_data_parser._joints_pred_xyz = right_data_parser._joints_pred_xyz_seg;
		right_data_parser._joints_pred_uv = right_data_parser._joints_pred_uv_seg;
		right_data_parser.realADT = right_data_parser.realADT_seg;
		right_data_parser.realADT_obj = right_data_parser.realADT_obj_seg;
	}
}

void Tracker::result_assignment_pre2seg_buffer()
{
	left_data_parser._color_org_320_seg_in = left_data_parser._color_org_320_pre.clone();
	left_data_parser._depth_org_320_seg_in = left_data_parser._depth_org_320_pre.clone();
	left_data_parser._depth_mm_320_seg_in = left_data_parser._depth_mm_320_pre.clone();
	left_data_parser._aligned_color_320_seg_in = left_data_parser._aligned_color_320_pre.clone();
	left_data_parser._hand_object_silhouette_seg_in = left_data_parser._hand_object_silhouette_pre.clone();

	if (track_with_two_cameras)
	{
		right_data_parser._color_org_320_seg_in = right_data_parser._color_org_320_pre.clone();
		right_data_parser._depth_org_320_seg_in = right_data_parser._depth_org_320_pre.clone();
		right_data_parser._depth_mm_320_seg_in = right_data_parser._depth_mm_320_pre.clone();
		right_data_parser._aligned_color_320_seg_in = right_data_parser._aligned_color_320_pre.clone();
		right_data_parser._hand_object_silhouette_seg_in = right_data_parser._hand_object_silhouette_pre.clone();
	}
}

void Tracker::result_assignment_seg2main_buffer()
{
	left_data_parser._color_org_320 = left_data_parser._color_org_320_seg_out.clone();
	left_data_parser._depth_org_320 = left_data_parser._depth_org_320_seg_out.clone();
	left_data_parser._depth_mm_320 = left_data_parser._depth_mm_320_seg_out.clone();
	left_data_parser._aligned_color_320 = left_data_parser._aligned_color_320_seg_out.clone();
	left_data_parser._hand_object_silhouette = left_data_parser._hand_object_silhouette_seg_out.clone();
	left_data_parser._segmentation_org = left_data_parser._segmentation_org_seg.clone();
	left_data_parser._hand_silhouette = left_data_parser._hand_silhouette_seg.clone();
	left_data_parser._object_silhouette = left_data_parser._object_silhouette_seg.clone();
	left_data_parser._joints_pred_xyz = left_data_parser._joints_pred_xyz_seg;
	left_data_parser._joints_pred_uv = left_data_parser._joints_pred_uv_seg;
	left_data_parser.realADT = left_data_parser.realADT_seg;
	left_data_parser.realADT_obj = left_data_parser.realADT_obj_seg;


	left_data_parser._color_org_320_seg_out = left_data_parser._color_org_320_seg_in.clone();
	left_data_parser._depth_org_320_seg_out = left_data_parser._depth_org_320_seg_in.clone();
	left_data_parser._depth_mm_320_seg_out = left_data_parser._depth_mm_320_seg_in.clone();
	left_data_parser._aligned_color_320_seg_out = left_data_parser._aligned_color_320_seg_in.clone();
	left_data_parser._hand_object_silhouette_seg_out = left_data_parser._hand_object_silhouette_seg_in.clone();

	if (track_with_two_cameras)
	{
		right_data_parser._color_org_320 = right_data_parser._color_org_320_seg_out.clone();
		right_data_parser._depth_org_320 = right_data_parser._depth_org_320_seg_out.clone();
		right_data_parser._depth_mm_320 = right_data_parser._depth_mm_320_seg_out.clone();
		right_data_parser._aligned_color_320 = right_data_parser._aligned_color_320_seg_out.clone();
		right_data_parser._hand_object_silhouette = right_data_parser._hand_object_silhouette_seg_out.clone();
		right_data_parser._segmentation_org = right_data_parser._segmentation_org_seg.clone();
		right_data_parser._hand_silhouette = right_data_parser._hand_silhouette_seg.clone();
		right_data_parser._object_silhouette = right_data_parser._object_silhouette_seg.clone();
		right_data_parser._joints_pred_xyz = right_data_parser._joints_pred_xyz_seg;
		right_data_parser._joints_pred_uv = right_data_parser._joints_pred_uv_seg;
		right_data_parser.realADT = right_data_parser.realADT_seg;
		right_data_parser.realADT_obj = right_data_parser.realADT_obj_seg;


		right_data_parser._color_org_320_seg_out = right_data_parser._color_org_320_seg_in.clone();
		right_data_parser._depth_org_320_seg_out = right_data_parser._depth_org_320_seg_in.clone();
		right_data_parser._depth_mm_320_seg_out = right_data_parser._depth_mm_320_seg_in.clone();
		right_data_parser._aligned_color_320_seg_out = right_data_parser._aligned_color_320_seg_in.clone();
		right_data_parser._hand_object_silhouette_seg_out = right_data_parser._hand_object_silhouette_seg_in.clone();
	}
}

void Tracker::hand_tracking(bool with_tips_pos)
{
	LONGLONG count_start_handtrack, count_end_handtrack;

	//QueryPerformanceCounter(&time_stmp);
	//count_start_handtrack = time_stmp.QuadPart;

	if (!with_tips_pos)
	{
		worker->handfinder->obtain_point_cloud(0, left_data_parser._hand_silhouette, left_data_parser.get_depth_mm_320(), 
			left_data_parser._camera_RT, left_data_parser._depth_para.fx / 2, left_data_parser._depth_para.fy / 2, 
			left_data_parser._depth_para.cx / 2, left_data_parser._depth_para.cy / 2);
		if (track_with_two_cameras)
		{
			worker->handfinder->obtain_point_cloud(1, right_data_parser._hand_silhouette, right_data_parser.get_depth_mm_320(),
				right_data_parser._camera_RT, right_data_parser._depth_para.fx / 2, right_data_parser._depth_para.fy / 2,
				right_data_parser._depth_para.cx / 2, right_data_parser._depth_para.cy / 2);
		}

		if (!track_with_two_cameras)
		{
			// track with single view camera
			tracking_failed = tracking_enabled ?
				worker->track_till_convergence(left_data_parser._hand_object_silhouette,
					left_data_parser.realADT,
					interaction_datas,
					object_pose,
					left_data_parser._joints_pred_xyz, left_data_parser._joints_pred_uv,
					store_time) : true;
		}
		else
		{
			// track with double cameras
			tracking_failed = tracking_enabled ?
				worker->track_till_convergence(left_data_parser._hand_object_silhouette, right_data_parser._hand_object_silhouette,
					left_data_parser.realADT, right_data_parser.realADT,
					interaction_datas,
					object_pose,
					left_data_parser._joints_pred_xyz, left_data_parser._joints_pred_uv,
					store_time) : true;
		}
	}
	else
	{
		worker->track_second_stage(tip_3D_keypoint_pos, interaction_datas);
	}

	/*QueryPerformanceCounter(&time_stmp);
	count_end_handtrack = time_stmp.QuadPart;
	count_interv = (double)(count_end_handtrack - count_start_handtrack);
	time_inter = count_interv * 1000 / count_freq;
	std::cout << "hand track" << "	" << time_inter << std::endl;*/

}

void Tracker::object_tracking_with_friction()
{
	/*LONGLONG count_start_objectTrack, count_end_objectTrack;

	QueryPerformanceCounter(&time_stmp);
	count_start_objectTrack = time_stmp.QuadPart;*/

	if (current_frame > reconstruction_frame)
	{
		/*                     solve the rigid motion                  */
		/***************************************************************/
		/*QueryPerformanceCounter(&time_stmp);
		count_start = time_stmp.QuadPart;*/

		{
			//calculate the depth map texture
			{
				left_data_parser.cal_depth_map_texture();

				if (show_mediate_result)
				{
					cv::Mat depth_vmap_c0(240, 320, CV_32FC4);
					cudaSafeCall(cudaMemcpyFromArray(depth_vmap_c0.data, left_data_parser._depth_processor.m_arrays.vmap_array, 0, 0, 4 * sizeof(float) * 240 * 320, cudaMemcpyDeviceToHost));
					cv::imshow("depth vmap c0", depth_vmap_c0);

					cv::Mat depth_nmap_c0(240, 320, CV_32FC4);
					cudaSafeCall(cudaMemcpyFromArray(depth_nmap_c0.data, left_data_parser._depth_processor.m_arrays.nmap_array, 0, 0, 4 * sizeof(float) * 240 * 320, cudaMemcpyDeviceToHost));
					cv::imshow("depth nmap c0", depth_nmap_c0);
					char depth_vmap_file[512];
					sprintf(depth_vmap_file, "depth_nmap_%04d.png", current_frame);
					cv::imwrite(input_store_path + depth_vmap_file, (depth_nmap_c0 + 1) * 255);
				}
			}

			worker->set_camera_object_motion(object_pose);
			//render the object map texture
			worker->render_texture_rigid();

			cudaTextureObject_t live_vmap_c0;
			cudaTextureObject_t live_nmap_c0;
			cudaTextureObject_t live_vmap_c1;
			cudaTextureObject_t live_nmap_c1;
			std::tie(live_vmap_c0, live_nmap_c0, live_vmap_c1, live_nmap_c1) = worker->map_rigid_icp_texobj();

			if (show_mediate_result)
			{
				{
					cv::Mat live_vmap_0(240, 320, CV_32FC4);
					cudaSafeCall(cudaMemcpyFromArray(live_vmap_0.data, worker->m_live_vmap_array_dv_c0, 0, 0, 4 * sizeof(float) * 240 * 320, cudaMemcpyDeviceToHost));
					cv::imshow("copy live vmap c0", live_vmap_0);

					int valid_number = 0;
					float weight_sum = 0;
					for (int r = 0; r<240; r++)
						for (int c = 0; c < 320; c++)
						{
							cv::Vec4f pixel_value = live_vmap_0.at<cv::Vec4f>(r, c);
							if (pixel_value[0] > 0 || pixel_value[1] > 0 || pixel_value[2] > 0)
							{
								valid_number++;
								weight_sum += pixel_value[3];
							}
						}

					printf("mean weight:%f\n", weight_sum / valid_number);


					cv::Mat live_nmap_0(240, 320, CV_32FC4);
					cudaSafeCall(cudaMemcpyFromArray(live_nmap_0.data, worker->m_live_nmap_array_dv_c0, 0, 0, 4 * sizeof(float) * 240 * 320, cudaMemcpyDeviceToHost));
					cv::imshow("copy live nmap c0", live_nmap_0);

					char live_vmap_file[512];
					sprintf(live_vmap_file, "live_nmap_%04d.png", current_frame);
					cv::imwrite(input_store_path + live_vmap_file, (live_nmap_0 + 1) * 255);
				}

				cv::waitKey(3);

			}

			//solve rigid motion of object

			rigid_solver.set_parameters2(left_data_parser._depth_processor.m_depth_tex_set.m_pDepthVMap, left_data_parser._depth_processor.m_depth_tex_set.m_pDepthNMap, live_vmap_c0, live_nmap_c0,
				object_pose, left_data_parser.realADT);

			cv::Mat hand_object_color;
			left_data_parser.get_aligned_color_320().copyTo(hand_object_color, left_data_parser._hand_object_silhouette);

			auto new_obj_pose = rigid_solver.run2_with_friction_huber(current_frame, hand_object_color, interaction_datas, run_surface2surface, 4, true, true/*, ObjectRecon.m_depth_data_pairs_stream*/);

			object_pose = new_obj_pose;

			worker->unmap_rigid_icp_texobj();

		}
		/*QueryPerformanceCounter(&time_stmp);
		count_end = time_stmp.QuadPart;
		count_interv = (double)(count_end - count_start);
		time_inter = count_interv * 1000 / count_freq;
		if (verbose)
		std::cout << "Rigid-Icp" << "	" << time_inter << std::endl;
		total_time += time_inter;*/

		/*if (store_time)
		nvprofiler::stop();*/
	}

	/*QueryPerformanceCounter(&time_stmp);
	count_end_objectTrack = time_stmp.QuadPart;
	count_interv = (double)(count_end_objectTrack - count_start_objectTrack);
	time_inter = count_interv * 1000 / count_freq;
	std::cout << "object track" << "	" << time_inter << std::endl;*/
}

void Tracker::object_tracking_without_friction()
{
	/*LONGLONG count_start_objectTrack, count_end_objectTrack;

	QueryPerformanceCounter(&time_stmp);
	count_start_objectTrack = time_stmp.QuadPart;*/

	if (current_frame > reconstruction_frame)
	{
		/*QueryPerformanceCounter(&time_stmp);
		count_start = time_stmp.QuadPart;*/
#if 1
		if (fuse_firstfra == 0)
		{
			cv::Mat Left_depth, Right_depth;
			left_data_parser.get_depth_mm_320().copyTo(Left_depth, left_data_parser._object_silhouette);

			ObjectRecon.fuse_FirstFrame(Left_depth, Right_depth);
			fuse_firstfra = 1;

			int weight_thr = current_frame - reconstruction_frame > 6 ? 5 : current_frame - reconstruction_frame - 1;

			worker->map_vertex_attributes();
			ObjectRecon.extract_NonRigid_SceneModel(weight_thr);
			ObjectRecon.warp_NonRigid_SceneModel();
			worker->vertex_number = ObjectRecon.m_valid_can_vertices.size();

			worker->unmap_vertex_attributes();

			ObjectRecon.construct_WarpField();
		}

		//calculate the distance of each node to the fingertips
		std::vector<float4> h_node_coordinate = ObjectRecon.get_node_coordinate_host();
		std::vector<DualQuaternion> h_node_motion = ObjectRecon.get_node_motion_host();

		HandObjDis.cal_Euler_distance(h_node_coordinate, h_node_motion, joint_position, worker->node_tip_idx, worker->variant_smooth, object_pose, current_frame);

		/*QueryPerformanceCounter(&time_stmp);
		count_end = time_stmp.QuadPart;
		count_interv = (double)(count_end - count_start);
		time_inter = count_interv * 1000 / count_freq;
		if (verbose)
		std::cout << "ConstructWarpField" << "	" << time_inter << std::endl;
		total_time = time_inter;*/

		/*                     solve the rigid motion                  */
		/***************************************************************/
		/*QueryPerformanceCounter(&time_stmp);
		count_start = time_stmp.QuadPart;*/

		{
			//calculate the depth map texture
			{
				left_data_parser.cal_depth_map_texture();

				if (show_mediate_result)
				{
					cv::Mat depth_vmap_c0(240, 320, CV_32FC4);
					cudaSafeCall(cudaMemcpyFromArray(depth_vmap_c0.data, left_data_parser._depth_processor.m_arrays.vmap_array, 0, 0, 4 * sizeof(float) * 240 * 320, cudaMemcpyDeviceToHost));
					cv::imshow("depth vmap c0", depth_vmap_c0);

					cv::Mat depth_nmap_c0(240, 320, CV_32FC4);
					cudaSafeCall(cudaMemcpyFromArray(depth_nmap_c0.data, left_data_parser._depth_processor.m_arrays.nmap_array, 0, 0, 4 * sizeof(float) * 240 * 320, cudaMemcpyDeviceToHost));
					cv::imshow("depth nmap c0", depth_nmap_c0);
					char depth_vmap_file[512];
					sprintf(depth_vmap_file, "depth_nmap_%04d.png", current_frame);
					cv::imwrite(input_store_path + depth_vmap_file, (depth_nmap_c0 + 1) * 255);
				}
			}

			worker->set_camera_object_motion(object_pose);
			//render the object map texture
			worker->render_texture_rigid();

			cudaTextureObject_t live_vmap_c0;
			cudaTextureObject_t live_nmap_c0;
			cudaTextureObject_t live_vmap_c1;
			cudaTextureObject_t live_nmap_c1;
			std::tie(live_vmap_c0, live_nmap_c0, live_vmap_c1, live_nmap_c1) = worker->map_rigid_icp_texobj();

			if (show_mediate_result)
			{
				{
					cv::Mat live_vmap_0(240, 320, CV_32FC4);
					cudaSafeCall(cudaMemcpyFromArray(live_vmap_0.data, worker->m_live_vmap_array_dv_c0, 0, 0, 4 * sizeof(float) * 240 * 320, cudaMemcpyDeviceToHost));
					cv::imshow("copy live vmap c0", live_vmap_0);

					int valid_number = 0;
					float weight_sum = 0;
					for (int r = 0; r<240; r++)
						for (int c = 0; c < 320; c++)
						{
							cv::Vec4f pixel_value = live_vmap_0.at<cv::Vec4f>(r, c);
							if (pixel_value[0] > 0 || pixel_value[1] > 0 || pixel_value[2] > 0)
							{
								valid_number++;
								weight_sum += pixel_value[3];
							}
						}

					printf("mean weight:%f\n", weight_sum / valid_number);

					cv::Mat live_nmap_0(240, 320, CV_32FC4);
					cudaSafeCall(cudaMemcpyFromArray(live_nmap_0.data, worker->m_live_nmap_array_dv_c0, 0, 0, 4 * sizeof(float) * 240 * 320, cudaMemcpyDeviceToHost));
					cv::imshow("copy live nmap c0", live_nmap_0);

					char live_vmap_file[512];
					sprintf(live_vmap_file, "live_nmap_%04d.png", current_frame);
					cv::imwrite(input_store_path + live_vmap_file, (live_nmap_0 + 1) * 255);
				}

				cv::waitKey(3);

			}

			//solve rigid motion of object
			rigid_solver.set_parameters2(left_data_parser._depth_processor.m_depth_tex_set.m_pDepthVMap, left_data_parser._depth_processor.m_depth_tex_set.m_pDepthNMap, live_vmap_c0, live_nmap_c0,
				object_pose, left_data_parser.realADT);

			cv::Mat hand_object_color;
			left_data_parser.get_aligned_color_320().copyTo(hand_object_color, left_data_parser._hand_object_silhouette);

			auto new_obj_pose = rigid_solver.run2_with_friction_huber(current_frame, hand_object_color, interaction_datas, run_surface2surface, 4, true, true/*, ObjectRecon.m_depth_data_pairs_stream*/);

			// solve new obj pose by interpolation
			if (current_frame > stop_recon_frame)
			{
				Eigen::Vector3f final_trans = (1 - obj_smooth_ratio) * new_obj_pose.block(0, 3, 3, 1) +
					obj_smooth_ratio * object_pose.block(0, 3, 3, 1);
				Eigen::Quaternionf new_obj_rot(Eigen::Matrix3f(new_obj_pose.block(0, 0, 3, 3)));
				Eigen::Quaternionf old_obj_rot(Eigen::Matrix3f(object_pose.block(0, 0, 3, 3)));
				Eigen::Quaternionf final_obj_rot = quatSlerp(old_obj_rot, new_obj_rot, 1 - obj_smooth_ratio);

				object_pose.block(0, 0, 3, 3) = final_obj_rot.toRotationMatrix();
				object_pose.block(0, 3, 3, 1) = final_trans;
			}
			else
			{
				object_pose = new_obj_pose;
			}

			worker->unmap_rigid_icp_texobj();

			/*std::cout << "icp" << std::endl;
			std::cout << object_pose << std::endl;*/

		}
#endif
		/*QueryPerformanceCounter(&time_stmp);
		count_end = time_stmp.QuadPart;
		count_interv = (double)(count_end - count_start);
		time_inter = count_interv * 1000 / count_freq;
		if (verbose)
		std::cout << "Rigid-Icp" << "	" << time_inter << std::endl;
		total_time += time_inter;*/

		/*                     solve the non-rigid motion                           */
		/****************************************************************************/
		/*QueryPerformanceCounter(&time_stmp);
		count_start = time_stmp.QuadPart;*/
		if (nonrigid_tracking)
		{
			//render the object map texture for nonrigid motion estimation
			worker->set_camera_object_motion(object_pose);
			worker->render_texture_nonrigid();

			cudaTextureObject_t can_vmap_c0;
			cudaTextureObject_t can_nmap_c0;
			cudaTextureObject_t can_vmap_c1;
			cudaTextureObject_t can_nmap_c1;
			std::tie(can_vmap_c0, can_nmap_c0, can_vmap_c1, can_nmap_c1) = worker->map_nonrigid_icp_texobj();
			/*if (!run_surface2surface)
			{
			ObjectRecon.nonrigid_MotionEstimation5(current_frame, data_parser._depth_processor.m_depth_tex_set.m_pDepthVMap, data_parser._depth_processor.m_depth_tex_set.m_pDepthNMap,
			can_vmap_c0, can_nmap_c0, object_pose, HandObjDis.NodeSmoothCoef_device, interaction_datas);
			}
			else*/
			{
				ObjectRecon.nonrigid_MotionEstimation_handblock(current_frame, left_data_parser._depth_processor.m_depth_tex_set.m_pDepthVMap, left_data_parser._depth_processor.m_depth_tex_set.m_pDepthNMap,
					can_vmap_c0, can_nmap_c0, object_pose, HandObjDis.NodeSmoothCoef_device, interaction_datas);
			}

			worker->unmap_nonrigid_icp_texobj();

			//std::cout << "nonrigid-icp!!!!!!" << std::endl;
			//std::cout << object_pose << std::endl;
		}
		/*QueryPerformanceCounter(&time_stmp);
		count_end = time_stmp.QuadPart;
		count_interv = (double)(count_end - count_start);
		time_inter = count_interv * 1000 / count_freq;
		if (verbose)
		std::cout << "Nonrigid-Icp" << "	" << time_inter << std::endl;
		total_time += time_inter;*/

		/*if (store_time)
		nvprofiler::stop();*/
	}

	/*QueryPerformanceCounter(&time_stmp);
	count_end_objectTrack = time_stmp.QuadPart;
	count_interv = (double)(count_end_objectTrack - count_start_objectTrack);
	time_inter = count_interv * 1000 / count_freq;
	std::cout << "object track" << "	" << time_inter << std::endl;*/
}

void Tracker::find_interaction_correspondence()
{

	/***************************************************************/
	/*               find interaction correspondence               */
	/***************************************************************/

	/*LONGLONG count_start_objectReco, count_end_objectReco;
	QueryPerformanceCounter(&time_stmp);
	count_start_objectReco = time_stmp.QuadPart;*/

	//update the joint positions after hand tracking
	joint_position.resize(worker->model->centers.size());
	for (int i = 0; i < worker->model->centers.size(); i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		joint_position[i] = make_float3(center_temp[0], center_temp[1], center_temp[2]);
	}

	//find interaction correspondences
	/*if (!run_surface2surface)
	{
	interaction_datas.set_finger_joint_position(joint_position);
	ObjectRecon.find_interaction_finger_joint_correspondence(Eigen2mat(object_pose), interaction_datas);
	interaction_datas.obtain_touch_status(Eigen2mat(object_pose));
	}
	else*/
	{
		interaction_datas.update_hand_joint_position(worker->model->centers);
		ObjectRecon.find_interaction_hand_correspondence_surface_contact(Eigen2mat(object_pose), interaction_datas);
	}

	/*QueryPerformanceCounter(&time_stmp);
	count_end = time_stmp.QuadPart;
	count_interv = (double)(count_end - count_start);
	time_inter = count_interv * 1000 / count_freq;
	if (verbose)
	std::cout << "Multi-Thread" << "	" << time_inter << std::endl;
	total_time += time_inter;*/
}

void Tracker::object_reconstruction()
{
	/****************************************************************************/
	/*                         object reconstruction                            */
	/****************************************************************************/
#if 1
	//assign data
	if (current_frame>reconstruction_frame)//&&!worker->is_only_track
	{
		//obtain the model-based segmentation of object and fuse data
		if (store_time)
			nvprofiler::start("Segment2andDataFusion", current_frame + 2);
		if ((current_frame > 500 && current_frame < 520)/*|| (current_frame>625 && current_frame<640)*/)
			;
		else
			;

		if (current_frame < stop_recon_frame)// && (current_frame<85 || current_frame>110) && (current_frame<210 || current_frame>216) && (current_frame<220 || current_frame>230) && (current_frame<270 || current_frame>280) && (current_frame<280 || current_frame>310)
		{
			/*QueryPerformanceCounter(&time_stmp);
			count_start = time_stmp.QuadPart;*/

			//calculate the depth map texture
			//if(camera_use==0||camera_use==1)
			//data_parser.cal_depth_map_texture();

			/*QueryPerformanceCounter(&time_stmp);
			count_start = time_stmp.QuadPart;*/

			ObjectRecon.nonrigid_fusion_C2(left_data_parser._depth_processor.m_depth_tex_set.m_pDepth, /*Right_DataParser._depth_processor.m_depth_tex_set.m_pDepth,*/ object_pose);

			/*QueryPerformanceCounter(&time_stmp);
			count_end = time_stmp.QuadPart;
			count_interv = (double)(count_end - count_start);
			time_inter = count_interv * 1000 / count_freq;
			if (verbose)
			std::cout << "DataFusion" << "	" << time_inter << std::endl;
			total_time += time_inter;*/
		}
		if (store_time)
			nvprofiler::stop();

		//assign data to worker variances to supply with OpenGL, this process can be optimized
		int weight_thr = current_frame - reconstruction_frame > 6 ? 6 : current_frame - reconstruction_frame - 1;//-6?5

																													/*if (current_frame > 480)
																													weight_thr = 1;*/

		worker->map_vertex_attributes();

		if (current_frame<stop_recon_frame + 1)
			ObjectRecon.extract_NonRigid_SceneModel(weight_thr);

		ObjectRecon.warp_NonRigid_SceneModel();
		worker->vertex_number = ObjectRecon.m_valid_can_vertices.size();

		worker->unmap_vertex_attributes();

		if (current_frame<stop_recon_frame + 1)
			ObjectRecon.construct_WarpField();

	}
#endif		
	/*QueryPerformanceCounter(&time_stmp);
	count_end_objectReco = time_stmp.QuadPart;
	count_interv = (double)(count_end_objectReco - count_start_objectReco);
	time_inter = count_interv * 1000 / count_freq;
	std::cout << "object recon" << "	" << time_inter << std::endl;*/
}

void Tracker::printf_hello()
{
	std::cout << "Hello World!" << std::endl;
}

void Tracker::show_store_seg(int store_frame_id)
{
	//cv::imshow("left segmentation", data_parser._segmentation);

	if (store_seg || show_seg)
	{
		if (show_seg)
		{
			// cv::imshow("hand-object slh", left_data_parser._hand_object_silhouette);
			// cv::imshow("left hand seg", left_data_parser._hand_silhouette);
			// cv::imshow("left object seg", left_data_parser._object_silhouette);
			cv::imshow("left segmentation", left_data_parser._segmentation);
			// cv::imshow("segment org", left_data_parser._segmentation_org);
			if (track_with_two_cameras)
			{
				cv::imshow("right segmentation", right_data_parser._segmentation);
			}
		}

		if (store_seg)
		{
			char mask_image_file[512] = { 0 };
			sprintf(mask_image_file, "left_hand_seg_%04d.png", store_frame_id);
			cv::imwrite(input_store_path + mask_image_file, left_data_parser._hand_silhouette);
			sprintf(mask_image_file, "left_object_seg_%04d.png", store_frame_id);
			cv::imwrite(input_store_path + mask_image_file, left_data_parser._object_silhouette);
			//sprintf(mask_image_file, "right_hand_seg_%04d.png", store_frame_id);
			//cv::imwrite(input_store_path + mask_image_file, Right_DataParser._hand_silhouette);
			//sprintf(mask_image_file, "right_object_seg_%04d.png", store_frame_id);
			//cv::imwrite(input_store_path + mask_image_file, Right_DataParser._object_silhouette);

			sprintf(mask_image_file, "left_segmentation_%04d.png", store_frame_id);
			cv::imwrite(input_store_path + mask_image_file, left_data_parser._segmentation);
			sprintf(mask_image_file, "left_segmentation_org_%04d.png", store_frame_id);
			cv::imwrite(input_store_path + mask_image_file, left_data_parser._segmentation_org);
		}

	}
}

void Tracker::showADT()
{
	if (show_ADT)
	{
		cv::imshow("left hand-object silh", left_data_parser._hand_object_silhouette);
		cv::imshow("left object silh", left_data_parser._object_silhouette);
		//cv::imshow("right hand-object silh", Right_DataParser._hand_object_silhouette);
		/*{
		std::string silh_store_path = "D:/Project/HandReconstruction/hmodel-master_vs2015_MultiCamera_s_TechnicalContribution/result/silhouette/";
		char mask_image_file[512] = { 0 };
		sprintf(mask_image_file, "left_silh_%04d.png", current_frame);
		cv::imwrite(silh_store_path + mask_image_file, data_parser._hand_object_silhouette);
		sprintf(mask_image_file, "right_silh_%04d.png", current_frame);
		cv::imwrite(silh_store_path + mask_image_file, Right_DataParser._hand_object_silhouette);
		}*/

		cv::Mat left_Silhouet(240, 320, CV_8U, cv::Scalar(0));
		for (int j = 0; j < left_Silhouet.rows; j++)
			for (int i = 0; i < left_Silhouet.cols; i++)
			{
				int idx_temp = left_data_parser.realADT[j * 320 + i];
				int row = idx_temp / left_Silhouet.cols;
				int col = idx_temp%left_Silhouet.cols;
				left_Silhouet.at<char>(row, col) = 255;
			}

		cv::Mat left_Obj_Silhouet(240, 320, CV_8U, cv::Scalar(0));
		for (int j = 0; j < left_Obj_Silhouet.rows; j++)
			for (int i = 0; i < left_Obj_Silhouet.cols; i++)
			{
				int idx_temp = left_data_parser.realADT_obj[j * 320 + i];
				int row = idx_temp / left_Obj_Silhouet.cols;
				int col = idx_temp%left_Obj_Silhouet.cols;
				left_Obj_Silhouet.at<char>(row, col) = 255;
			}

		/*cv::Mat right_Silhouet(240, 320, CV_8U, cv::Scalar(0));
		for (int j = 0; j < right_Silhouet.rows; j++)
		for (int i = 0; i < right_Silhouet.cols; i++)
		{
		int idx_temp = Right_DataParser.realADT[j * 320 + i];
		int row = idx_temp / right_Silhouet.cols;
		int col = idx_temp%right_Silhouet.cols;
		right_Silhouet.at<char>(row, col) = 255;
		}*/

		cv::imshow("left ADT", left_Silhouet);
		cv::imshow("left obj ADT", left_Obj_Silhouet);
		//cv::imshow("right ADT", right_Silhouet);
		cv::waitKey(3);
	}
}

void Tracker::show_store_input(int store_frame_id)
{
	/*cv::Mat left_hand_object_color;
	data_parser.get_aligned_color_320().copyTo(left_hand_object_color, data_parser._hand_object_silhouette);
	cv::imshow("left hand-object color", left_hand_object_color);*/

	if (show_input || store_input)
	{
		/*cv::imshow("left depth org", data_parser._depth_org);
		cv::imshow("left color org", data_parser._color_org);*/

		auto obtainColorInput = [&](DataParser& parser, cv::Mat& color_img, cv::Mat& depth_img, 
			camera_intr& dep_intr, camera_intr& color_intr, Eigen::Matrix4f& d2c) {
			cv::Mat tmp_hand_object_color, tmp_hand_obj_depth;
		    color_img = cv::Mat(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));
			parser.get_aligned_color_320().copyTo(tmp_hand_object_color, parser._hand_object_silhouette);
			parser.get_depth_mm_320().copyTo(tmp_hand_obj_depth, parser._hand_object_silhouette);

			int width = tmp_hand_obj_depth.cols;
			int height = tmp_hand_obj_depth.rows;

			for (int r = 0; r < height; ++r)
			{
				for (int c = 0; c < width; ++c)
				{
					float dep = (float)tmp_hand_obj_depth.at<ushort>(r, c) / 1000.0f;
					if (dep != 0)
					{
						float x = dep * (c - dep_intr.cx / 2) * 2.0f / dep_intr.fx;
						float y = dep * (r - dep_intr.cy / 2) * 2.0f / dep_intr.fy;
						Eigen::Vector3f p(x, y, dep);
						p = (d2c * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f)).head(3);
						Eigen::Vector3f cp(p.x() / p.z(), p.y() / p.z(), 1.0);
						int u = round(cp.x() * color_intr.fx / 2.0f + color_intr.cx / 2.0f);
						int v = round(cp.y() * color_intr.fy / 2.0f + color_intr.cy / 2.0f);
						if (u >= 0 && v >= 0 && v < 240 && u < 320)
						{
							color_img.at<Vec3b>(v, u) = parser.get_org_color_320().at<Vec3b>(v, u);
							for (int p = u - 1; p <= u + 1; ++p)
							{
								for (int q = v - 1; q <= v + 1; ++q)
								{
									if (p >= 0 && q >= 0 && q < 240 && p < 320)
										color_img.at<Vec3b>(q, p) = parser.get_org_color_320().at<Vec3b>(q, p);
								}
							}
						}
					}
				}
			}
			cv::Mat left_handobject_depth_8U(height, width, CV_8UC1, cv::Scalar(0));

			for (int j = 0; j < height; j++)
				for (int i = 0; i < width; i++)
					left_handobject_depth_8U.at<uchar>(j, i) = (uchar)((float)tmp_hand_obj_depth.at<ushort>(j, i) / 600.0f * 255);

			cv::applyColorMap(left_handobject_depth_8U, depth_img, COLORMAP_JET);
		};

		if (show_input || store_input)
		{
			cv::Mat left_ho_color, left_ho_depth_color;
			cv::Mat right_ho_color, right_ho_depth_color;
			obtainColorInput(left_data_parser, left_ho_color, left_ho_depth_color, depth_camera_cl, color_camera_cl, depth2color_cl);
			if (track_with_two_cameras)
			{
				obtainColorInput(right_data_parser, right_ho_color, right_ho_depth_color, depth_camera_cr, color_camera_cr, depth2color_cr);
			}

			if (show_input)
			{
				// cv::imshow("left hand-object color", left_hand_object_color);
				cv::imshow("left hand-object color", left_ho_color);
				cv::imshow("left hand-object depth", left_ho_depth_color);
				if (track_with_two_cameras)
				{
					cv::imshow("right hand-object color", right_ho_color);
					cv::imshow("right hand-object depth", right_ho_depth_color);
				}
			}

			if (store_input)
			{
				char mask_image_file[512] = { 0 };
				sprintf(mask_image_file, "left_color_%04d.png", store_frame_id);
				cv::imwrite(input_store_path + mask_image_file, left_ho_color);
				sprintf(mask_image_file, "left_depth_%04d.png", store_frame_id);
				cv::imwrite(input_store_path + mask_image_file, left_ho_depth_color);
				if (track_with_two_cameras)
				{
					sprintf(mask_image_file, "right_color_%04d.png", store_frame_id);
					cv::imwrite(input_store_path + mask_image_file, right_ho_color);
					sprintf(mask_image_file, "right_depth_%04d.png", store_frame_id);
					cv::imwrite(input_store_path + mask_image_file, right_ho_depth_color);
				}
			}
		}
	}
}

void Tracker::check_keypoints_energy(int store_frame_id)
{
	//output the keypoint
	std::vector<float3> key_points;
	key_points.clear();

	//thumb keypoints and palm root
	for (int i = 16; i <= 19; i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		key_points.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}
	key_points.push_back(make_float3(worker->model->centers[25][0], worker->model->centers[25][1], worker->model->centers[25][2]));

	//index keypoint
	for (int i = 12; i <= 15; i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		key_points.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}

	//middle keypoint
	for (int i = 8; i <= 11; i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		key_points.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}

	//ring keypoint
	for (int i = 4; i <= 7; i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		key_points.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}

	//pinky keypoint
	for (int i = 0; i <= 3; i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		key_points.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}

	std::vector<float3> local_3D_keypoints;
	std::vector<float2> keypoints2D_pos;
	//std::vector<uchar> left_2D_visible, right_2D_visible;

	std::vector<float2> keypoint2D_Pred = left_data_parser._joints_pred_uv;

	std::vector<float3> global_keypoints;
	global_keypoints.clear();

	for (int i = 0; i < worker->using_keypoint_2D.size(); i++)
	{
		global_keypoints.push_back(key_points[worker->using_keypoint_2D[i]]);

	}

	cv::Mat color_keypoints;
	std::vector<float2> TrackedHand_2DKeypoints;

	color_keypoints = left_data_parser.TrackedHand_3DKeypointsProjection2Dimage(global_keypoints, TrackedHand_2DKeypoints);

	for (int i = 0; i < keypoint2D_Pred.size(); i++)
	{
		for (int v = (int)(keypoint2D_Pred[i].y / 2) - 1; v <= (int)(keypoint2D_Pred[i].y / 2) + 1; v++)
			for (int u = (int)(keypoint2D_Pred[i].x / 2) - 1; u <= (int)(keypoint2D_Pred[i].x / 2) + 1; u++)
			{
				if (v >= 0 && v < color_keypoints.rows&&u >= 0 && u < color_keypoints.cols)
				{
					color_keypoints.at<cv::Vec3b>(v, u)[0] = 0;
					color_keypoints.at<cv::Vec3b>(v, u)[1] = 0;
					color_keypoints.at<cv::Vec3b>(v, u)[2] = 255;
				}
			}
	}

	{
		cv::Mat color_keypoints_640;
		cv::resize(color_keypoints, color_keypoints_640, cv::Size(640, 480));
		cv::imshow("projection of tracked hand with Pred", color_keypoints_640);
		char align_color_image_file[512] = { 0 };
		sprintf(align_color_image_file, "projection/Given_Tracked_Keypoints_%04d.png", store_frame_id);
		cv::imwrite(hand_pose_store_path + align_color_image_file, color_keypoints_640);
	}

	std::string solutions;
	solutions = hand_pose_store_path + "hmodel_solutions.txt";
	static ofstream solutions_file(solutions);
	if (solutions_file.is_open())
	{
		solutions_file << store_frame_id;
		for (int idx = 0; idx < current_hand_pose.size(); idx++)
			solutions_file << " " << current_hand_pose[idx];
		solutions_file << std::endl;
	}
}

void Tracker::show_visible_keypoints(int store_frame_id)
{
	//store the 3D keypoints and 2D projection
	//output the keypoint
	std::vector<float3> key_points;
	key_points.clear();

	//thumb keypoints and palm root
	for (int i = 16; i <= 19; i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		key_points.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}
	key_points.push_back(make_float3(worker->model->centers[25][0], worker->model->centers[25][1], worker->model->centers[25][2]));

	//index keypoint
	for (int i = 12; i <= 15; i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		key_points.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}

	//middle keypoint
	for (int i = 8; i <= 11; i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		key_points.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}

	//ring keypoint
	for (int i = 4; i <= 7; i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		key_points.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}

	//pinky keypoint
	for (int i = 0; i <= 3; i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		key_points.push_back(make_float3(center_temp[0], center_temp[1], center_temp[2]));
	}

	std::vector<float3> left_3D_keypoints;//, right_3D_keypoints;
	std::vector<float2> left_2D_keypoints;// , right_2D_keypoints;
	std::vector<uchar> left_2D_visible;// , right_2D_visible;

	cv::Mat color_keypoints_left = left_data_parser.TrackedHand_3DKeypointsProjection2Dimage(key_points, left_2D_keypoints);

	//get the use aligned color image and mm depth image 
	cv::Mat left_hand_object_color;
	left_data_parser.get_aligned_color_320().copyTo(left_hand_object_color, left_data_parser._hand_object_silhouette);

	cv::Mat left_handobject_depth;
	left_data_parser.get_depth_mm_320().copyTo(left_handobject_depth, left_data_parser._hand_object_silhouette);

	cv::Mat left_hand_mask = left_data_parser._hand_silhouette;// ExtractSkinbyEcclips(left_hand_object_color);


															/*cv::imshow("left hand mask", left_hand_mask);
															cv::imshow("right hand mask", right_hand_mask);*/

															//visibility_check(left_handobject_depth, left_hand_mask, left_3D_keypoints, left_2D_keypoints, left_2D_visible);

	for (int i = 0; i < left_2D_keypoints.size(); i++)
	{
		if (left_2D_visible[i]>128)
		{
			for (int v = (int)(left_2D_keypoints[i].y / 2) - 2; v <= (int)(left_2D_keypoints[i].y / 2) + 2; v++)
				for (int u = (int)(left_2D_keypoints[i].x / 2) - 2; u <= (int)(left_2D_keypoints[i].x / 2) + 2; u++)
				{
					if (v >= 0 && v < left_hand_object_color.rows&&u >= 0 && u < left_hand_object_color.cols)
					{
						left_hand_object_color.at<cv::Vec3b>(v, u)[0] = 255;
						left_hand_object_color.at<cv::Vec3b>(v, u)[1] = 0;
						left_hand_object_color.at<cv::Vec3b>(v, u)[2] = 0;
					}
				}
		}
	}

	cv::imshow("left visibility 2D keypoint", left_hand_object_color);
	char align_color_image_file[512] = { 0 };
	sprintf(align_color_image_file, "projection/left_visibility_2D_keypoints%04d.png", store_frame_id);
	cv::imwrite(hand_pose_store_path + align_color_image_file, left_hand_object_color);

	std::string left_3D_keypoints_file, left_2D_keypoints_file, left_2D_keypoints_visible_file, left_visibility;

	left_3D_keypoints_file = hand_pose_store_path + "left_3D_keypoints.txt";
	left_2D_keypoints_file = hand_pose_store_path + "left_2D_keypoints.txt";
	left_2D_keypoints_visible_file = hand_pose_store_path + "left_2D_keypoints_visible.txt";
	left_visibility = hand_pose_store_path + "left_visibility.txt";

	//left 3D keypoints
	static ofstream left_3D_solutions_file(left_3D_keypoints_file);
	if (left_3D_solutions_file.is_open())
	{
		left_3D_solutions_file << store_frame_id;
		for (int idx = 0; idx < left_3D_keypoints.size(); idx++)
			left_3D_solutions_file << " " << left_3D_keypoints[idx].x << " " << left_3D_keypoints[idx].y << " " << left_3D_keypoints[idx].z;
		left_3D_solutions_file << std::endl;
	}
	//left 2D keypoints -- all
	static ofstream left_2D_solutions_file(left_2D_keypoints_file);
	if (left_2D_solutions_file.is_open())
	{
		left_2D_solutions_file << store_frame_id;
		for (int idx = 0; idx < left_2D_keypoints.size(); idx++)
		{
			/*if (left_2D_visible[idx] > 128)*/
			left_2D_solutions_file << " " << left_2D_keypoints[idx].x << " " << left_2D_keypoints[idx].y;
			/*else
			left_2D_solutions_file << " " << -1.00 << " " << -1.00;*/
		}

		left_2D_solutions_file << std::endl;
	}
	//left 2D keypoints -- visible
	static ofstream left_2D_solutions_visible_file(left_2D_keypoints_visible_file);
	if (left_2D_solutions_visible_file.is_open())
	{
		left_2D_solutions_visible_file << store_frame_id;
		for (int idx = 0; idx < left_2D_keypoints.size(); idx++)
		{
			if (left_2D_visible[idx] > 128)
				left_2D_solutions_visible_file << " " << left_2D_keypoints[idx].x << " " << left_2D_keypoints[idx].y;
			else
				left_2D_solutions_visible_file << " " << -1.00 << " " << -1.00;
		}

		left_2D_solutions_visible_file << std::endl;
	}

	//store left visibility
	static ofstream left_2D_visibility_file(left_visibility);
	if (left_2D_visibility_file.is_open())
	{
		left_2D_visibility_file << store_frame_id;
		for (int idx = 0; idx < left_2D_visible.size(); idx++)
		{
			left_2D_visibility_file << " " << (left_2D_visible[idx]>128);
			//						std::cout << " " << (left_2D_visible[idx]>128);
		}

		left_2D_visibility_file << std::endl;
	}
}

void Tracker::display_color_and_depth_input() {
	cv::Mat normalized_depth = worker->current_frame.depth.clone();
	cv::inRange(normalized_depth, worker->camera->zNear(), worker->camera->zFar(), normalized_depth);
	cv::normalize(normalized_depth, normalized_depth, 127, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::resize(normalized_depth, normalized_depth, cv::Size(2 * normalized_depth.cols, 2 * normalized_depth.rows), cv::INTER_CUBIC);//resize image
	cv::moveWindow("DEPTH", 592, 855); cv::imshow("DEPTH", normalized_depth);

	cv::namedWindow("RGB");	cv::moveWindow("RGB", 592, 375); cv::imshow("RGB", worker->model->real_color);
}

void Tracker::perform_smooth()
{
	static bool is_first_frame = true;
	if (is_first_frame)
	{
		is_first_frame = false;
		old_hand_theta = worker->model->get_theta();
		return;
	}
	else
	{
		// smooth hand motion
		std::set<int> tip_theta{ 12, 16, 20, 24, 28 };
		std::vector<float> new_theta = worker->model->get_theta();
		for (int i = 0; i < 3; i++)
		{
			new_theta[i] = obj_smooth_ratio * (old_hand_theta[i] - new_theta[i]);
		}
		for (int i = 3; i < 7; i++)
		{
			new_theta[i] = 0;
		}
		for (int i = 7; i < new_theta.size(); i++)
		{
			if (tip_theta.find(i) == tip_theta.end())
			{
				new_theta[i] = hand_smooth_ratio * (old_hand_theta[i] - new_theta[i]);
			}
			else
			{
				new_theta[i] = hand_smooth_ratio2 * (old_hand_theta[i] - new_theta[i]);
			}
		}
		new_theta = worker->model->get_updated_parameters(worker->model->get_theta(), new_theta);
		worker->model->move(new_theta);
		worker->model->update_centers();
		worker->model->compute_outline();
		old_hand_theta = worker->model->get_theta();

		// smooth object motion
		Eigen::Vector3f final_trans = (1 - obj_smooth_ratio) * object_pose.block(0, 3, 3, 1) +
			obj_smooth_ratio * old_obj_motion.block(0, 3, 3, 1);
		Eigen::Quaternionf new_obj_rot(Eigen::Matrix3f(object_pose.block(0, 0, 3, 3)));
		Eigen::Quaternionf old_obj_rot(Eigen::Matrix3f(old_obj_motion.block(0, 0, 3, 3)));
		Eigen::Quaternionf final_obj_rot = quatSlerp(old_obj_rot, new_obj_rot, 1 - obj_smooth_ratio);

		object_pose.block(0, 0, 3, 3) = final_obj_rot.toRotationMatrix();
		object_pose.block(0, 3, 3, 1) = final_trans;

		old_obj_motion = object_pose;
	}
}

void Tracker::perform_force_smooth()
{
	if (worker->force_history.size() == 0)
	{
		worker->force_history.resize(5, Eigen::Vector3f(0, 0, 0));
		return;
	}

	int contact_num = worker->contact_forces.size();
	vector<bool> is_update(5, false);
	for (int i = 0; i < contact_num; i++)
	{
		int tip_idx = JOINT_IDX_TO_TIP[worker->contact_corr[i]];
		Eigen::Vector3f hforce = worker->force_history[tip_idx];
		Eigen::Vector3f cforce = worker->contact_forces[i];
		if (hforce.norm() <= 1e-3 || cforce.norm() <= 1e-3)
		{
			worker->force_history[tip_idx] = cforce;
		}
		else
		{
			worker->contact_forces[i] = worker->force_history[tip_idx] =
				force_smooth_ratio * hforce + (1.0f - force_smooth_ratio) * cforce;
		}
		is_update[tip_idx] = true;
	}
	for (int i = 0; i < 5; i++)
	{
		if (!is_update[i])
			worker->force_history[i] = Eigen::Vector3f(0, 0, 0);
	}
}

void Tracker::track_hand_with_physics()
{
	/*
	Add by HuHaoyu
	2021/11/11
	Calculate physical constraints based on contact and perform optimization on hand tips
	*/

	if (current_frame == stop_recon_frame + 1)
	{
		// canonical object vertices and normals
		std::vector<float4> host_obj_can_vertices = std::vector<float4>();
		std::vector<float4> host_obj_can_normals = std::vector<float4>();
		ObjectRecon.m_valid_can_vertices.download(host_obj_can_vertices);
		ObjectRecon.m_valid_can_normals.download(host_obj_can_normals);

		std::vector<Eigen::Vector3f> obj_can_vertices_eigen(host_obj_can_vertices.size());
		std::vector<Eigen::Vector3f> obj_can_normals_eigen(host_obj_can_normals.size());

		for (int i = 0; i < host_obj_can_vertices.size(); i++)
		{
			obj_can_vertices_eigen[i].x() = host_obj_can_vertices[i].x;
			obj_can_vertices_eigen[i].y() = host_obj_can_vertices[i].y;
			obj_can_vertices_eigen[i].z() = host_obj_can_vertices[i].z;

			obj_can_normals_eigen[i].x() = host_obj_can_normals[i].x;
			obj_can_normals_eigen[i].y() = host_obj_can_normals[i].y;
			obj_can_normals_eigen[i].z() = host_obj_can_normals[i].z;
		}

		phys_hand_solver.initForceSolver(nonrigid_tracking);
		phys_hand_solver.loadObjectMesh(obj_can_vertices_eigen, obj_can_normals_eigen);
	}

	if (!phys_hand_solver.isReadyToSolve())
	{
		return;
	}

	// collect data
	// contact points and normals
	auto contact_points = interaction_datas.get_valid_interaction_corrs_warped_vertex();
	auto contact_norm = interaction_datas.get_valid_interaction_corrs_warped_normal();

	std::vector<Eigen::Vector3f> contact_points_eigen(contact_points.size());
	std::vector<Eigen::Vector3f> contact_normals_eigen(contact_norm.size());
	std::vector<Eigen::Vector3f> joint_pos_eigen(worker->model->centers.size());
	std::vector<float> joint_radii_arr(worker->model->radii.size());

	for (int i = 0; i < contact_points.size(); i++)
	{
		contact_points_eigen[i].x() = contact_points[i].x;
		contact_points_eigen[i].y() = contact_points[i].y;
		contact_points_eigen[i].z() = contact_points[i].z;

		contact_normals_eigen[i].x() = contact_norm[i].x;
		contact_normals_eigen[i].y() = contact_norm[i].y;
		contact_normals_eigen[i].z() = contact_norm[i].z;
	}

	// joint position
	for (int i = 0; i < TOTAL_JOINT_NUM; i++)
	{
		const std::string name = JOINT_NAME[i];
		auto joint_id = worker->model->centers_name_to_id_map[name];
		glm::vec3 center_temp = worker->model->centers[joint_id];
		float radius = worker->model->radii[joint_id];

		joint_pos_eigen[i].x() = center_temp.x;
		joint_pos_eigen[i].y() = center_temp.y;
		joint_pos_eigen[i].z() = center_temp.z;
		joint_radii_arr[i] = radius;
	}

	LARGE_INTEGER freq;
	LARGE_INTEGER start_t, stop_t;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start_t);
	{
		tip_3D_keypoint_pos = phys_hand_solver.solve(contact_points_eigen, contact_normals_eigen, joint_pos_eigen,
			joint_radii_arr, worker->tips_rel_conf, object_pose);
		if (output_contact_info && current_frame > stop_recon_frame)
		{
			// output kinematic result
			outputContactInfo("../../../../result/contact_data_kin/obj_motion_and_contact_info_");
			outputHandObjMotion("../../../../result/motion_seq_kin.json");
		}
	}
	QueryPerformanceCounter(&stop_t);
	printf("Solve phys hand costs %f ms\n",
		1e3 * (stop_t.QuadPart - start_t.QuadPart) / freq.QuadPart);

	QueryPerformanceCounter(&start_t);
	{
		hand_tracking(true);
	}
	QueryPerformanceCounter(&stop_t);
	printf("Second time hand tracking costs %f ms\n",
		1e3 * (stop_t.QuadPart - start_t.QuadPart) / freq.QuadPart);

	// send final tips pos back to physhand solver
	std::vector<Eigen::Vector3f> tips_final_pos(5);
	for (int i = 0; i < 5; i++)
	{
		const std::string name = JOINT_NAME[TIPS_JOINT_IDX[i]];
		auto joint_id = worker->model->centers_name_to_id_map[name];
		glm::vec3 center_temp = worker->model->centers[joint_id];

		tips_final_pos[i].x() = center_temp.x;
		tips_final_pos[i].y() = center_temp.y;
		tips_final_pos[i].z() = center_temp.z;
	}
	phys_hand_solver.setTipsFinalPos(tips_final_pos);

	// output tips joint difference between result and ref
	if (0)
	{
		ofstream ofs("../../../../result/tips_diff.txt", ios_base::app);
		ofstream ofs2("../../../../result/tips_diff_org.txt", ios_base::app);
		float max_diff = 0;
		float max_diff2 = 0;
		for (int i = 0; i < 5; i++)
		{
			max_diff = std::max(max_diff, (tips_final_pos[i] - tip_3D_keypoint_pos[i]).norm());
			max_diff2 = std::max(max_diff2, (joint_pos_eigen[TIPS_JOINT_IDX[i]] - tip_3D_keypoint_pos[i]).norm());
		}
		// ofs << max_diff << '\t' << max_diff2 << endl;
		ofs << max_diff << endl;
		ofs2 << max_diff2 << endl;
		ofs.close();
		ofs2.close();
	}

	// get physics info from solver
	phys_hand_solver.getPhysInfo(worker->contact_points, worker->contact_forces, worker->contact_corr,
		worker->target_force, worker->target_moment, worker->object_center,
		worker->object_vel, worker->object_ang_vel, worker->object_rot);

	perform_force_smooth();
}

void Tracker::eval_tips_err(int frame_id)
{
	stringstream ss;
	ss << std::setw(4) << std::setfill('0') << frame_id;
	string mark_file = tip_2D_key_point_path + ss.str() + ".json";

	Json::Reader reader;
	Json::Value m_jsonRoot;
	std::ifstream fin(mark_file);

	if (!fin.is_open())
	{
		return;
	}

	if (!reader.parse(fin, m_jsonRoot))
	{
		std::cout << "can't parse json file\n";
		exit(-1);
	}

	//get GT
	std::vector<pair<float, float>> GT_pos(5);
	for (int i = 0; i < 5; i++)
	{
		GT_pos[i] = make_pair(0, 0);
	}
	for (auto label : m_jsonRoot["shapes"])
	{
		int tip_idx = tips_name_to_idx[label["label"].asString()];
		GT_pos[tip_idx] = make_pair(label["points"][0][0].asFloat(), label["points"][0][1].asFloat());
	}

	//get tips keypoint
	std::vector<Eigen::Vector3f> key_points;
	key_points.clear();

	glm::vec3 center_temp;

	center_temp = worker->model->centers[16];//thumb
	key_points.push_back(Eigen::Vector3f(center_temp[0], -center_temp[1], center_temp[2]));
	center_temp = worker->model->centers[12];//index
	key_points.push_back(Eigen::Vector3f(center_temp[0], -center_temp[1], center_temp[2]));
	center_temp = worker->model->centers[8];//middle
	key_points.push_back(Eigen::Vector3f(center_temp[0], -center_temp[1], center_temp[2]));
	center_temp = worker->model->centers[4];//ring
	key_points.push_back(Eigen::Vector3f(center_temp[0], -center_temp[1], center_temp[2]));
	center_temp = worker->model->centers[0];//pinky
	key_points.push_back(Eigen::Vector3f(center_temp[0], -center_temp[1], center_temp[2]));

	std::vector<pair<float, float>> keypoint_pixel_left;

	for (auto kp : key_points)
	{
		Eigen::Vector3f col_kp = dep_to_col.block(0, 0, 3, 3) * kp + 1e3 * dep_to_col.block(0, 3, 3, 1);
		float u = col_proj.fx * col_kp.x() / col_kp.z() + col_proj.cx;
		float v = col_proj.fy * col_kp.y() / col_kp.z() + col_proj.cy;
		keypoint_pixel_left.push_back(make_pair(u, v));
	}

	// calculate L2 distance
	int valid_num = 0;
	float avg_dist = 0;
	for (int idx = 0; idx < keypoint_pixel_left.size(); idx++)
	{
		float GT_posX = GT_pos[idx].first;
		float GT_posY = GT_pos[idx].second;
		float recon_posX = keypoint_pixel_left[idx].first;
		float recon_posY = keypoint_pixel_left[idx].second;
		if (GT_posX == 0 && GT_posY == 0)
		{
			// no GT infomation of this tip, ignore it
			continue;
		}
		float keypoint_dist = sqrtf((GT_posX - recon_posX) * (GT_posX - recon_posX) + (GT_posY - recon_posY) * (GT_posY - recon_posY));
		avg_dist += keypoint_dist;
		valid_num += 1;
	}

	avg_dist /= valid_num;

	// output result;
	ofstream ofs("../../../../result/pixel_diff.txt", ios_base::app);
	ofs << frame_id << ' ' << avg_dist << endl;
	ofs.close();
}

void Tracker::outputContactInfo(std::string file_prefix)
{
	// output contact points and object pose
	auto contact_points = interaction_datas.get_valid_interaction_corrs_warped_vertex();
	auto contact_norm = interaction_datas.get_valid_interaction_corrs_warped_normal();

	Json::Value frame_data;
	Json::Value obj_motion_data;
	Json::Value contact_data;
	Json::Value contact_point_data;
	Json::Value contact_norm_data;

	Json::Value keypoint_data;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
			obj_motion_data.append(object_pose(i, j));
	}

	for (int i = 0; i < contact_points.size(); i++)
	{
		contact_point_data.append(contact_points[i].x);
		contact_point_data.append(contact_points[i].y);
		contact_point_data.append(contact_points[i].z);

		contact_norm_data.append(contact_norm[i].x);
		contact_norm_data.append(contact_norm[i].y);
		contact_norm_data.append(contact_norm[i].z);
	}
	contact_data["point"] = contact_point_data;
	contact_data["norm"] = contact_norm_data;

	for (std::string& name : joint_names_output_brief)
	{
		auto joint_id = worker->model->centers_name_to_id_map[name];
		glm::vec3 center_temp = worker->model->centers[joint_id];
		float radius = worker->model->radii[joint_id];
		Json::Value joint_data;
		joint_data["joint_name"] = name;
		joint_data["joint_id"] = joint_id;
		joint_data["center"].append(center_temp[0]);
		joint_data["center"].append(center_temp[1]);
		joint_data["center"].append(center_temp[2]);
		joint_data["radius"] = radius;
		keypoint_data.append(joint_data);
	}

	frame_data["frame_id"] = current_frame;
	frame_data["obj_motion"] = obj_motion_data;
	frame_data["contacts"] = contact_data;
	frame_data["keypoints"] = keypoint_data;

	Json::StyledWriter sw;
	ofstream os;
	os.open(file_prefix + std::to_string(current_frame) + ".json");
	if (!os.is_open())
	{
		std::cerr << "(Tracker) Error: Can not open file for obj motion and contact info record!" << std::endl;
		return;
	}
	os << sw.write(frame_data);
	os.close();
}

void Tracker::outputHandObjMotion(std::string filename)
{
	static std::set<std::string> file_set;

	Json::StyledWriter sw;

	// convert motion info into json format
	Json::Value hand_motion;
	Json::Value obj_motion;
	Json::Value tips_target_pos;
	Json::Value tips_conf;
	Json::Value tips_org_conf;
	Json::Value motion_info;
	Json::Value j_radius;

	for (std::string& name : joint_names_output_full)
	{
		auto joint_id = worker->model->centers_name_to_id_map[name];
		glm::vec3 center_temp = worker->model->centers[joint_id];

		hand_motion.append(center_temp[0]);
		hand_motion.append(center_temp[1]);
		hand_motion.append(center_temp[2]);
		j_radius.append(worker->model->radii[joint_id]);
	}
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
			obj_motion.append(object_pose(i, j));
	}
	for (auto p : tip_3D_keypoint_pos)
	{
		tips_target_pos.append(p.x());
		tips_target_pos.append(p.y());
		tips_target_pos.append(p.z());
	}
	for (auto c : worker->tips_rel_conf)
	{
		tips_conf.append(c);
	}
	for (auto c : worker->tips_org_conf)
	{
		tips_org_conf.append(c);
	}

	motion_info["frame_id"] = current_frame;
	motion_info["hand_motion"] = hand_motion;
	motion_info["joint_radius"] = j_radius;
	motion_info["tips_target_pos"] = tips_target_pos;
	motion_info["obj_motion"] = obj_motion;
	motion_info["tips_conf"] = tips_conf;
	motion_info["tips_org_conf"] = tips_org_conf;

	if (file_set.find(filename) == file_set.end())
	{
		ofstream os;
		os.open(filename);

		Json::Value json_root;
		Json::Value hand_info;
		Json::Value joint_name;
		Json::Value joint_id;

		for (std::string& name : joint_names_output_full)
		{
			auto center_id = worker->model->centers_name_to_id_map[name];

			joint_name.append(name);
			joint_id.append(center_id);
		}
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
				obj_motion.append(object_pose(i, j));
		}

		hand_info["joint_name"] = joint_name;
		hand_info["joint_id"] = joint_id;
		json_root["hand_info"] = hand_info;
		json_root["motion_info"].append(motion_info);

		os << sw.write(json_root);
		os.close();

		file_set.insert(filename);
	}
	else
	{
		std::fstream os;
		os.open(filename);

		os.seekp(-11, std::ios::end);
		os << ",\n" << sw.write(motion_info) << "   ]\n}\n";

		os.close();
	}
}

void Tracker::process_track() {
	nonrigid_tracking = false;
	if (current_frame > nonrigid_start_frame && is_nonrigid)
	{
		nonrigid_tracking = true;
	}

	// In online tracking, when to start object reconstruction is controlled by the user
	if (worker->is_reconstruction && is_set_recon_frame == false)
	{
		reconstruction_frame = current_frame;
		worker->reconstruction_frame = current_frame;
		is_set_recon_frame = true;
	}

	// Object non-rigid tracking usually starts when it is fully reconstructed
	if (worker->is_only_track && is_set_nonrigid_frame == false)
	{
		nonrigid_start_frame = current_frame;
		stop_recon_frame = current_frame;
		is_set_nonrigid_frame = true;
	}

	// For time efficiency evaluation
	float total_time = 0;
	LONGLONG count_start_total, count_end_total;
	LONGLONG count_start_mainpipe, count_end_mainpipe;
	LONGLONG count_start_tracking, count_end_tracking;
	LONGLONG count_start_fusion, count_end_fusion;
	LONGLONG count_start_render, count_end_render;
	LONGLONG count_start_multi_thrd, count_end_multi_thrd;

	QueryPerformanceCounter(&time_stmp);
	count_start_total = time_stmp.QuadPart;

	/*******************************************************************************/
	/*        data preprocess (h-o segmentation and keypoints prediction)          */
	/*******************************************************************************/

	if (current_frame == start_frame)
	{
		data_preprocessing();
		result_assignment_pre2seg();
		data_segmentation_init();
		result_assignment_seg2main();

		++current_frame;
	}

	std::thread data_preprocess_thread(&Tracker::data_preprocessing, this);

	int store_frame_id = current_frame - 3;

	worker->frame_id = store_frame_id;
	worker->current_frame.id = store_frame_id;

	QueryPerformanceCounter(&time_stmp);
	count_start_mainpipe = time_stmp.QuadPart;

	std::thread data_segmentation_thread(&Tracker::data_segmentation, this);

	QueryPerformanceCounter(&time_stmp);
	count_start_multi_thrd = time_stmp.QuadPart;

	show_store_seg(store_frame_id);
	showADT();
	show_store_input(store_frame_id);

	// Keypoints Prediction Debug
	bool _debug_keypoints_prediction = false;
	if (_debug_keypoints_prediction)
	{
		cv::Mat color_3Dkeypoints = left_data_parser.draw_3Dkeypoint2image(left_data_parser._joints_pred_xyz, cv::Vec3b(0, 0, 255));
		cv::imshow("projection of 3Dkeypoints", color_3Dkeypoints);

		cv::Mat color_2Dkeypoints = left_data_parser.draw_2Dkeypoint2image(left_data_parser._joints_pred_uv, cv::Vec3b(255, 0, 0));
		cv::imshow("projection of 2Dkeypoints", color_2Dkeypoints);
	}

	/*******************************************************************************/
	/*                   hand tracking and object reconstruction                   */
	/*******************************************************************************/

	//--------------stage1: kinematic hand tracking-----------------------//
	if (store_time)
	{
		nvprofiler::start("MultiThread", current_frame + 1);
	}
	if (store_time)
	{
		nvprofiler::stop();
	}
	QueryPerformanceCounter(&time_stmp);
	count_start_tracking = time_stmp.QuadPart;

	interaction_datas.store_hand_joint_position_before(worker->model->centers);
	interaction_datas.set_phalange_transformation_local2global_before(worker->model->get_phalange_global_mat());

	// Track hand and object motion seperately
	std::vector<Eigen::Vector3f> empty_tips_pos;
	std::thread hand_track_thr(&Tracker::hand_tracking, this, false);
	object_tracking_without_friction();
	hand_track_thr.join();

	// Update h-o interaction data
	joint_position.resize(worker->model->centers.size());
	for (int i = 0; i < worker->model->centers.size(); i++)
	{
		glm::vec3 center_temp = worker->model->centers[i];
		joint_position[i] = make_float3(center_temp[0], center_temp[1], center_temp[2]);
	}

	if (!run_surface2surface)
	{
		interaction_datas.set_finger_joint_position(joint_position);
	}
	else
	{
		interaction_datas.update_hand_joint_position(worker->model->centers);
	}

	QueryPerformanceCounter(&time_stmp);
	count_end_tracking = time_stmp.QuadPart;
	count_interv = (double)(count_end_tracking - count_start_tracking);
	time_inter_tracking_hand = count_interv * 1000 / count_freq;

	QueryPerformanceCounter(&time_stmp);
	count_start_tracking = time_stmp.QuadPart;

	interaction_datas.set_phalange_transformation_local2global(worker->model->get_phalange_global_mat());

	// Track object motion again with tangential movement term
	object_tracking_with_friction();

	//--------------stage2: Physical Interaction Refine-----------------------//
	if (track_hand_with_phys)
	{
		track_hand_with_physics();
	}
	if (current_frame > reconstruction_frame)
	{
		perform_smooth();
	}

	QueryPerformanceCounter(&time_stmp);
	count_end_tracking = time_stmp.QuadPart;
	count_interv = (double)(count_end_tracking - count_start_tracking);
	time_inter_tracking_obj = count_interv * 1000 / count_freq;
	QueryPerformanceCounter(&time_stmp);
	count_start_fusion = time_stmp.QuadPart;

	//--------------stage3: Reconstruction object shape-----------------------//
	object_reconstruction();

	// Update h-o interaction data
	interaction_datas.set_phalange_transformation_global2local(worker->model->get_phalange_global_mat());
	find_interaction_correspondence();

	//assign segmentation
	worker->segmentation = left_data_parser.get_segmentation_greyC3().clone();
	worker->keypt_pred_img = left_data_parser.get_keypoint_pred_color();
	worker->Keypoint_3D_pred = left_data_parser._joints_pred_xyz;

	// Handle tracking failure
	if (initialization_enabled && tracking_failed) {
		static QianDetection detection(worker);
		if (detection.can_reinitialize()) {
			detection.reinitialize();
		}
	}

	worker->set_camera_object_motion(object_pose);
	worker->updateGL();

	while (worker->is_watch)
	{
		worker->updateGL();
		cv::waitKey(30);
	}

	if (store_input)
	{
		cv::imshow("seg", worker->keypt_pred_img);
		char mask_image_file[512] = { 0 };
		sprintf(mask_image_file, "handkeypoint_prediction_%04d.png", store_frame_id);
		cv::imwrite(input_store_path + mask_image_file, worker->keypt_pred_img.clone());
	}

	if (store_solution_hand)
	{
		std::string solutions;
		solutions = hand_pose_store_path + "hmodel_solutions.txt";
		static ofstream solutions_file(solutions);
		if (solutions_file.is_open())
		{
			solutions_file << store_frame_id;
			for (int idx = 0; idx < current_hand_pose.size(); idx++)
				solutions_file << " " << current_hand_pose[idx];
			solutions_file << std::endl;
		}
	}

	//--------------Output some results if necessary-----------------------//
	bool _output_cano_object = false;
	bool _output_object_motion = false;
	// output cano object 
	if (_output_cano_object && current_frame == stop_recon_frame + 1)
	{
		std::vector<float4> host_obj_can_vertices = std::vector<float4>();
		std::vector<float4> host_obj_can_normals = std::vector<float4>();
		ObjectRecon.m_valid_can_vertices.download(host_obj_can_vertices);
		ObjectRecon.m_valid_can_normals.download(host_obj_can_normals);
		ofstream ofs_v("../../../../result/obj_v.txt");
		ofstream ofs_n("../../../../result/obj_n.txt");
		if (!ofs_v.is_open() || !ofs_n.is_open())
		{
			std::cout << "Can not open file for object mesh record!" << std::endl;
		}
		for (float4& p : host_obj_can_vertices)
		{
			ofs_v << p.x << ' ' << p.y << ' ' << p.z << std::endl;
		}
		for (float4& n : host_obj_can_normals)
		{
			ofs_n << n.x << ' ' << n.y << ' ' << n.z << std::endl;
		}
		ofs_v.close();
		ofs_n.close();
	}

	// output object motion
	if (_output_object_motion && current_frame > stop_recon_frame)
	{
		ofstream ofs("../../../../result/motion_seq.txt", ios::app);
		ofs << current_frame;
		for (int r = 0; r < 4; ++r)
		{
			for (int c = 0; c < 4; ++c)
				ofs << " " << object_pose(r, c);
		}
		ofs << endl;
	}

	if (output_contact_info && current_frame > stop_recon_frame)
	{
		outputContactInfo("../../../../result/contact_data/obj_motion_and_contact_info_");
		outputHandObjMotion("../../../../result/motion_seq.json");
	}

	if (use_tip_2D_key_point)
	{
		eval_tips_err(store_frame_id);
	}

	QueryPerformanceCounter(&time_stmp);
	count_end_fusion = time_stmp.QuadPart;
	count_interv = (double)(count_end_fusion - count_start_fusion);
	time_inter_fusion = count_interv * 1000 / count_freq;

	if (store_time)
		nvprofiler::stop();

	QueryPerformanceCounter(&time_stmp);
	count_end_multi_thrd = time_stmp.QuadPart;
	count_interv = (double)(count_end_multi_thrd - count_start_multi_thrd);
	time_inter = count_interv * 1000 / count_freq;

	data_segmentation_thread.join();

	QueryPerformanceCounter(&time_stmp);
	count_end_mainpipe = time_stmp.QuadPart;
	count_interv = (double)(count_end_mainpipe - count_start_mainpipe);
	time_inter_mainpipe = count_interv * 1000 / count_freq;

	data_preprocess_thread.join();

	result_assignment_seg2main_buffer();
	result_assignment_pre2seg_buffer();

	current_hand_pose = worker->model->get_theta();
	worker->predicted_pose = predicted_hand_pose;

	QueryPerformanceCounter(&time_stmp);
	count_end_total = time_stmp.QuadPart;
	count_interv = (double)(count_end_total - count_start_total);
	time_inter_total = count_interv * 1000 / count_freq;

	// Output detailed time performance data
	if (verbose)
	{
		std::cout << "preprocess" << "	" << time_process << std::endl;
		std::cout << "seg+kpt" << "	" << time_seg << std::endl;
		std::cout << "hand tracking" << "	" << time_inter_tracking_hand << std::endl;
		std::cout << "object tracking" << "	" << time_inter_tracking_obj << std::endl;
		std::cout << "object fusion + find itr" << "	" << time_inter_fusion << std::endl;
		std::cout << "main thread" << "	" << time_inter << std::endl;
		std::cout << "mainpipeline" << "	" << time_inter_mainpipe << std::endl;

		std::cout << "total" << "	" << time_inter_total << std::endl;
		std::cout << "valid contact number:" << interaction_datas.valid_interaction_corres_num << std::endl;
	}

	// Output brief time performance data
	// if (current_frame > reconstruction_frame + 5 || true)
	{
		if (current_frame == nonrigid_start_frame)
		{
			mean_main_thread = 0;
			mean_time_mainpipeline = 0;
			mean_time_total = 0;
			sum_number = 0;
		}

		mean_main_thread += time_inter;
		mean_time_mainpipeline += time_inter_mainpipe;
		mean_time_total += time_inter_total;

		sum_number += 1;

		if (nonrigid_tracking)
		{
			std::cout << "mean time fusion nonrigid:" << "	" << mean_main_thread / sum_number << "ms" << std::endl;
			std::cout << "mean time main pipeline nonrigid" << "	" << mean_time_mainpipeline / sum_number << std::endl;
			std::cout << "mean time total nonrigid" << "	" << mean_time_total / sum_number << std::endl;
		}
		else
		{
			std::cout << "mean time fusion rigid:" << "	" << mean_main_thread / sum_number << "ms" << std::endl;
			std::cout << "mean time main pipeline rigid" << "	" << mean_time_mainpipeline / sum_number << std::endl;
			std::cout << "mean time total rigid" << "	" << mean_time_total / sum_number << std::endl;
		}

		time_sequence.push_back(time_inter_total);
		mean_time_sequence.push_back(mean_time_mainpipeline / sum_number);
		frame_id_sequence.push_back(current_frame);
	}

	current_frame += speedup;
}
