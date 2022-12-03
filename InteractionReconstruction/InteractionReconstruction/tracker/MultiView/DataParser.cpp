#include "DataParser.h"

Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));

Eigen::Vector3f depth_to_point(cv::Mat& depth, int col, int row)
{
	float fx = 237.0f;
	float fy = 237.0f;
	float cx = 160.0f;
	float cy = 120.0f;

	unsigned short depth_temp = depth.at<unsigned short>(row, col);
	Eigen::Vector3f point_temp;

	point_temp[2] = depth_temp;
	point_temp[0] = point_temp[2] * (col - cx) / fx;
	point_temp[1] = -point_temp[2] * (row - cy) / fy;

	return point_temp;
}

void DataParser::initial_parsar(int sequence_num, image_bias image_bias_value, camera_intr color_intrinsic, camera_intr depth_intrinsic, camera_intr camera_para, Eigen::Matrix4f depth2color_extrinsic, Eigen::Matrix4f camera_pose/*, HandFinder* handfinder*/)
{
	_sequence_id = sequence_num;
	_color_intr = color_intrinsic;
	_depth_intr = depth_intrinsic;
	_depth2color_extr = depth2color_extrinsic;
	_u_bias = image_bias_value.u_bias;
	_v_bias = image_bias_value.v_bias;
	_depth_para = camera_para;

	_camera_RT = camera_pose;

//	hand_finder = handfinder;

	Eigen::Vector4f camera_dir_temp = _camera_RT*Eigen::Vector4f(0, 0, 1, 0);
	_camera_dir = Eigen::Vector3f(camera_dir_temp(0), camera_dir_temp(1), camera_dir_temp(2));

	_depth_processor.initialize(320, 240);

	realDT.resize(width*height);
	DTTps.resize(width*height);
	ADTTps.resize(width*height);
	realADT_seg.resize(width*height);
	v.resize(width*height);
	z.resize(width*height);

	realDT_obj.resize(width*height);
	DTTps_obj.resize(width*height);
	ADTTps_obj.resize(width*height);
	realADT_obj_seg.resize(width*height);
	v_obj.resize(width*height);
	z_obj.resize(width*height);

	chunk1.memory = (char*)malloc(100000);  /* will be grown as needed by realloc above */

	process_pipeline= TimeContinuityProcess(3);

	QueryPerformanceFrequency(&time_stmp);
	count_freq = (double)time_stmp.QuadPart;

	if (_sequence_id == 0)
		camera_id = "1";
	if (_sequence_id == 1)
		camera_id = "2";
}

void DataParser::set_online_org_data(cv::Mat& depth_img, cv::Mat& color_img)
{
	_depth_org_pre = depth_img.clone();
	_color_org_pre = color_img.clone();
}

void DataParser::load_org_data(std::string data_path, int current_frame)
{
	std::ostringstream stringstream;
	stringstream << std::setw(4) << std::setfill('0') << current_frame;

	_color_org_pre = cv::imread(data_path + "_color_frame" + stringstream.str() + ".png");
	_depth_org_pre = cv::imread(data_path + "_depth_frame" + stringstream.str() + ".png", cv::IMREAD_UNCHANGED);

	/*std::ostringstream stringstream;
	stringstream << std::setw(7) << std::setfill('0') << current_frame;

	_color_org = cv::imread(data_path + "color-" + stringstream.str() + ".png");
	_depth_org = cv::imread(data_path + "depth-" + stringstream.str() + ".png", cv::IMREAD_UNCHANGED); */
}

void DataParser::obtain_resized_data2()
{
	Mat resized_color;
	Mat resized_depth_mm = cv::Mat(240, 320, CV_16U, cv::Scalar(0));
	Mat resized_depth_org = cv::Mat(240, 320, CV_16U, cv::Scalar(0));

	cv::Size d_size(320, 240);

	cv::resize(_color_org_pre, resized_color, d_size);
//	cv::resize(_depth_org, resized_depth_org, d_size, CV_INTER_NN);

	for(int row=1;row<resized_depth_mm.rows;row++)
		for (int col = 1; col < resized_depth_mm.cols; col++)
		{
			std::vector<int> neighbors = {
				_depth_org_pre.at<unsigned short>(2 * row - 1, 2 * col - 1),//D_width - (x - 1)- 1
				_depth_org_pre.at<unsigned short>(2 * row + 0, 2 * col - 1),
				_depth_org_pre.at<unsigned short>(2 * row + 1, 2 * col - 1),
				_depth_org_pre.at<unsigned short>(2 * row - 1, 2 * col + 0),//D_width - (x + 0) - 1
				_depth_org_pre.at<unsigned short>(2 * row + 0, 2 * col + 0),
				_depth_org_pre.at<unsigned short>(2 * row + 1, 2 * col + 0),
				_depth_org_pre.at<unsigned short>(2 * row - 1, 2 * col + 1),//D_width - (x + 1) - 1
				_depth_org_pre.at<unsigned short>(2 * row + 0, 2 * col + 1),
				_depth_org_pre.at<unsigned short>(2 * row + 1, 2 * col + 1),
			};

			std::sort(neighbors.begin(), neighbors.end());

			if (neighbors[4] > 100)
			{
				resized_depth_org.at<ushort>(row, col) = neighbors[4];
				resized_depth_mm.at<ushort>(row, col) = neighbors[4]/ _depth_to_mm_scale;
			}
		}

//	cv::Mat _depth_org_320_buffer_temp;// = cv::Mat(240, 320, CV_16U, cv::Scalar(0));

	//for (int row = 0; row < 240; row++)
	//	for (int col = 0; col < 320; col++)
	//		_depth_org_320_buffer_temp.at<unsigned short>(row, col) = _depth_org.at<unsigned short>(2 * row, 2 * col);

//	cv::resize(_depth_org, _depth_org_320_buffer_temp, cv::Size(320, 240), INTER_NEAREST);

	_color_org_320_pre = resized_color.clone();
	_depth_org_320_pre = resized_depth_org.clone();
	_depth_mm_320_pre = resized_depth_mm.clone();
}

void DataParser::obtain_aligned_data2()
{
	int height = _color_org_320_pre.rows;
	int width = _color_org_320_pre.cols;

	Mat dst_color_map(_color_org_320_pre.size(), _color_org_320_pre.type());

	for (int ridx = 0; ridx < height; ++ridx)
	{
		for (int cidx = 0; cidx < width; ++cidx)
		{
			unsigned short depth_temp = _depth_mm_320_pre.at<unsigned short>(ridx, cidx);
			Vec3b pxl_u_v;
			pxl_u_v[0] = 0; pxl_u_v[1] = 0; pxl_u_v[2] = 0;
			dst_color_map.at<Vec3b>(ridx, cidx) = pxl_u_v;

			if (depth_temp > 200)
			{
				Point3D depth_point;
				depth_point.z = (float)depth_temp / 1000.0f;
				depth_point.x = ((float)cidx - _depth_intr.cx / 2) / (_depth_intr.fx / 2) * depth_point.z;
				depth_point.y = ((float)ridx - _depth_intr.cy / 2) / (_depth_intr.fy / 2) * depth_point.z;

				Eigen::Vector4f homo_depth_point(depth_point.x, depth_point.y, depth_point.z, 1.0f);

				Eigen::Vector4f homo_color_point = _depth2color_extr*homo_depth_point;//depth2color.inverse()*

				float c_u = _color_intr.fx/2*homo_color_point[0] / homo_color_point[2] + _color_intr.cx/2 + _u_bias/2;
				float c_v = _color_intr.fy/2*homo_color_point[1] / homo_color_point[2] + _color_intr.cy/2 + _v_bias/2;

				int u0 = (int)c_u, v0 = (int)c_v;

				/*if (u0 < 0)u0 = 0;
				if (u0 > depth_width - 1)u0 = depth_width - 1;
				if (v0 < 0)v0 = 0;
				if (v0 > depth_height - 1)v0 = depth_height - 1;*/

				if (u0<0 || u0>(width - 2) || v0<0 || v0>(height - 2))
					continue;

				int u1 = u0 + 1, v1 = v0 + 1;

				Vec3b pxl_u0_v0 = _color_org_320_pre.at<Vec3b>(v0, u0);
				Vec3b pxl_u1_v0 = _color_org_320_pre.at<Vec3b>(v0, u1);
				Vec3b pxl_u0_v1 = _color_org_320_pre.at<Vec3b>(v1, u0);
				Vec3b pxl_u1_v1 = _color_org_320_pre.at<Vec3b>(v1, u1);

				for (int i = 0; i < 3; i++)
				{
					pxl_u_v[i] = (uchar)((u1 - c_u)*(v1 - c_v)*pxl_u0_v0[i] + (c_u - u0)*(v1 - c_v)*pxl_u1_v0[i] + (u1 - c_u)*(c_v - v0)*pxl_u0_v1[i] + (c_u - u0)*(c_v - v0)*pxl_u1_v1[i]);
				}
				dst_color_map.at<Vec3b>(ridx, cidx) = pxl_u_v;
			}
		}
	}

	_aligned_color_320_pre = dst_color_map.clone();

	Mat aligned_color_bgr;
	cv::cvtColor(_aligned_color_320_pre, aligned_color_bgr, COLOR_RGB2BGR);
	_aligned_color_bgr_320 = aligned_color_bgr.clone();
}

void DataParser::show_original_data()
{
	char image_name[256];
	sprintf(image_name, "%d original color", _sequence_id);
	imshow(image_name, _color_org_pre);
	sprintf(image_name, "%d original depth", _sequence_id);
	imshow(image_name, _depth_gray);
}

void DataParser::show_result_data()
{
	char image_name[256];
	sprintf(image_name, "%d color_320_bgr", _sequence_id);
	imshow(image_name, _aligned_color_bgr);
	sprintf(image_name, "%d depth_mm_320 ", _sequence_id);
	imshow(image_name, _depth_mm_320);
}

cv::Mat DataParser::get_color_bgr()
{
	return _aligned_color_bgr;
}

cv::Mat DataParser::get_color_bgr_320()
{
	return _aligned_color_bgr_320;//_aligned_color_bgr_320_buffer
}

cv::Mat DataParser::get_depth_mm_320()
{
	return _depth_mm_320;
}

cv::Mat DataParser::get_org_color_320()
{
	return _color_org_320;
}

cv::Mat DataParser::get_aligned_color_320()
{
	return _aligned_color_320;
}

cv::Mat DataParser::draw_3Dkeypoint2image(std::vector<float3> &local_key_points, cv::Vec3b color_value)
{
	//project key point to image
	cv::Mat color_map = _aligned_color_320.clone();
	int window = 2;
	camera_intr came_para;
	came_para.fx = _depth_para.fx / 2; came_para.fy = _depth_para.fy / 2; came_para.cx = _depth_para.cx / 2; came_para.cy = _depth_para.cy / 2;

	for (int i = 0; i < local_key_points.size(); i++)
	{
		int img_u = came_para.fx*local_key_points[i].x / local_key_points[i].z + came_para.cx;
		int img_v = came_para.fy*local_key_points[i].y / local_key_points[i].z + came_para.cy;
		//			img_v = came_para.image_height - img_v;

		for (int v = img_v - window; v < img_v + window; v++)
			for (int u = img_u - window; u < img_u + window; u++)
			{
				if (v >= 0 && v < color_map.rows&&u >= 0 && u < color_map.cols)
				{
					color_map.at<cv::Vec3b>(v, u) = color_value;
				}
			}
	}

	return color_map;
}

cv::Mat DataParser::draw_2Dkeypoint2image(std::vector<float2> &local_key_points, cv::Vec3b color_value)
{
	//project key point to image
	cv::Mat color_map = _aligned_color_320.clone();
	int window = 2;
	
	for (int i = 0; i < local_key_points.size(); i++)
	{
		int img_u = local_key_points[i].x/2;
		int img_v = local_key_points[i].y/2;
		//			img_v = came_para.image_height - img_v;

		for (int v = img_v - window; v < img_v + window; v++)
			for (int u = img_u - window; u < img_u + window; u++)
			{
				if (v >= 0 && v < color_map.rows&&u >= 0 && u < color_map.cols)
				{
					color_map.at<cv::Vec3b>(v, u) = color_value;
				}
			}
	}

	return color_map;
}

cv::Mat DataParser::TrackedHand_3DKeypointsProjection2Dimage(const std::vector<float3> global_key_points, std::vector<float2> &image_2D_keypoints)
{
	//obtain local key_points
	std::vector<float3> local_key_points;
	local_key_points.clear();

	for (int i=0; i < global_key_points.size(); i++)
	{
		Eigen::Vector4f key_point_temp(global_key_points[i].x, -global_key_points[i].y, global_key_points[i].z, 1.0f);

		local_key_points.push_back(make_float3(key_point_temp[0], key_point_temp[1], key_point_temp[2]));
	}

	//project key point to image
	cv::Mat color_map = _aligned_color_320.clone();
	int window = 2;
	camera_intr came_para;
	came_para.fx = _depth_para.fx / 2; came_para.fy = _depth_para.fy / 2; came_para.cx = _depth_para.cx / 2; came_para.cy = _depth_para.cy / 2;

	image_2D_keypoints.clear();
	for (int i = 0; i < local_key_points.size(); i++)
	{
		float img_u = came_para.fx*local_key_points[i].x / local_key_points[i].z + came_para.cx;
		float img_v = came_para.fy*local_key_points[i].y / local_key_points[i].z + came_para.cy;
		//			img_v = came_para.image_height - img_v;

		image_2D_keypoints.push_back(make_float2(img_u * 2, img_v * 2));

		for (int v = (int)img_v - window; v <= (int)img_v + window; v++)
			for (int u = (int)img_u - window; u <= (int)img_u + window; u++)
			{
				if (v >= 0 && v < color_map.rows&&u >= 0 && u < color_map.cols)
				{
					color_map.at<cv::Vec3b>(v, u)[0] = 255;
					color_map.at<cv::Vec3b>(v, u)[1] = 255;
					color_map.at<cv::Vec3b>(v, u)[2] = 0;
				}
			}
	}

	return color_map;
}

cv::Mat DataParser::project_keypoint2image640(std::vector<float3> &key_points, std::vector<float2> &key_points_pixel)
{
	//obtain local key_points
	key_points_pixel.clear();
	std::vector<float3> local_key_points;
	local_key_points.clear();

	for (int i = 0; i < key_points.size(); i++)
	{
		Eigen::Vector4f key_point_temp(key_points[i].x, -key_points[i].y, key_points[i].z, 1.0f);

		Eigen::Vector4f local_point = _camera_RT.inverse()*key_point_temp;

		local_key_points.push_back(make_float3(local_point[0], local_point[1], local_point[2]));
	}

	//project key point to image
	cv::Mat color_map_320;
	_aligned_color_320.copyTo(color_map_320, _hand_object_silhouette);
	cv::Mat color_map;
	cv::resize(color_map_320, color_map, cv::Size(640, 480));

	int window = 2;
	camera_intr came_para;
	came_para.fx = _depth_para.fx; came_para.fy = _depth_para.fy; came_para.cx = _depth_para.cx; came_para.cy = _depth_para.cy;

	float img_u;
	float img_v;
	int finger_idx;
	{
		//thumb
		finger_idx = 0;
		img_u = came_para.fx*local_key_points[finger_idx].x / local_key_points[finger_idx].z + came_para.cx;
		img_v = came_para.fy*local_key_points[finger_idx].y / local_key_points[finger_idx].z + came_para.cy;
		key_points_pixel.push_back(make_float2(img_u, img_v));
		for (int v = img_v - window; v < img_v + window; v++)
			for (int u = img_u - window; u < img_u + window; u++)
			{
				if (v >= 0 && v < color_map.rows&&u >= 0 && u < color_map.cols)
				{
					color_map.at<cv::Vec3b>(v, u)[0] = 0;
					color_map.at<cv::Vec3b>(v, u)[1] = 0;
					color_map.at<cv::Vec3b>(v, u)[2] = 255;
				}
			}

		//index
		finger_idx = 1;
		img_u = came_para.fx*local_key_points[finger_idx].x / local_key_points[finger_idx].z + came_para.cx;
		img_v = came_para.fy*local_key_points[finger_idx].y / local_key_points[finger_idx].z + came_para.cy;
		key_points_pixel.push_back(make_float2(img_u, img_v));
		for (int v = img_v - window; v < img_v + window; v++)
			for (int u = img_u - window; u < img_u + window; u++)
			{
				if (v >= 0 && v < color_map.rows&&u >= 0 && u < color_map.cols)
				{
					color_map.at<cv::Vec3b>(v, u)[0] = 0;
					color_map.at<cv::Vec3b>(v, u)[1] = 255;
					color_map.at<cv::Vec3b>(v, u)[2] = 255;
				}
			}

		//middle
		finger_idx = 2;
		img_u = came_para.fx*local_key_points[finger_idx].x / local_key_points[finger_idx].z + came_para.cx;
		img_v = came_para.fy*local_key_points[finger_idx].y / local_key_points[finger_idx].z + came_para.cy;
		key_points_pixel.push_back(make_float2(img_u, img_v));
		for (int v = img_v - window; v < img_v + window; v++)
			for (int u = img_u - window; u < img_u + window; u++)
			{
				if (v >= 0 && v < color_map.rows&&u >= 0 && u < color_map.cols)
				{
					color_map.at<cv::Vec3b>(v, u)[0] = 0;
					color_map.at<cv::Vec3b>(v, u)[1] = 255;
					color_map.at<cv::Vec3b>(v, u)[2] = 0;
				}
			}

		//ring
		finger_idx = 3;
		img_u = came_para.fx*local_key_points[finger_idx].x / local_key_points[finger_idx].z + came_para.cx;
		img_v = came_para.fy*local_key_points[finger_idx].y / local_key_points[finger_idx].z + came_para.cy;
		key_points_pixel.push_back(make_float2(img_u, img_v));
		for (int v = img_v - window; v < img_v + window; v++)
			for (int u = img_u - window; u < img_u + window; u++)
			{
				if (v >= 0 && v < color_map.rows&&u >= 0 && u < color_map.cols)
				{
					color_map.at<cv::Vec3b>(v, u)[0] = 255;
					color_map.at<cv::Vec3b>(v, u)[1] = 0;
					color_map.at<cv::Vec3b>(v, u)[2] = 0;
				}
			}

		//pinky
		finger_idx = 4;
		img_u = came_para.fx*local_key_points[finger_idx].x / local_key_points[finger_idx].z + came_para.cx;
		img_v = came_para.fy*local_key_points[finger_idx].y / local_key_points[finger_idx].z + came_para.cy;
		key_points_pixel.push_back(make_float2(img_u, img_v));
		for (int v = img_v - window; v < img_v + window; v++)
			for (int u = img_u - window; u < img_u + window; u++)
			{
				if (v >= 0 && v < color_map.rows&&u >= 0 && u < color_map.cols)
				{
					color_map.at<cv::Vec3b>(v, u)[0] = 255;
					color_map.at<cv::Vec3b>(v, u)[1] = 0;
					color_map.at<cv::Vec3b>(v, u)[2] = 255;
				}
			}
	}

	return color_map;
}

mat34 DataParser::get_camera_pose_mat34()
{
	mat34 camera_RT34;
	camera_RT34.rot.m00() = _camera_RT(0, 0); camera_RT34.rot.m01() = _camera_RT(0, 1); camera_RT34.rot.m02() = _camera_RT(0, 2); camera_RT34.trans.x = _camera_RT(0, 3)/1000.0;
	camera_RT34.rot.m10() = _camera_RT(1, 0); camera_RT34.rot.m11() = _camera_RT(1, 1); camera_RT34.rot.m12() = _camera_RT(1, 2); camera_RT34.trans.y = _camera_RT(1, 3)/1000.0;
	camera_RT34.rot.m20() = _camera_RT(2, 0); camera_RT34.rot.m21() = _camera_RT(2, 1); camera_RT34.rot.m22() = _camera_RT(2, 2); camera_RT34.trans.z = _camera_RT(2, 3)/1000.0;

	return camera_RT34;
}

void DataParser::cal_depth_map_texture()
{
	cv::Mat depth_object;
	_depth_mm_320.copyTo(depth_object, _object_silhouette);
//	_depth_processor.get_map_texture(depth_object, _depth_para.fx/2, _depth_para.fy/2, _depth_para.cx/2, _depth_para.cy/2);

	mat34 camera_RT34;
	camera_RT34.rot.m00() = _camera_RT(0, 0); camera_RT34.rot.m01() = _camera_RT(0, 1); camera_RT34.rot.m02() = _camera_RT(0, 2); camera_RT34.trans.x = _camera_RT(0, 3);
	camera_RT34.rot.m10() = _camera_RT(1, 0); camera_RT34.rot.m11() = _camera_RT(1, 1); camera_RT34.rot.m12() = _camera_RT(1, 2); camera_RT34.trans.y = _camera_RT(1, 3);
	camera_RT34.rot.m20() = _camera_RT(2, 0); camera_RT34.rot.m21() = _camera_RT(2, 1); camera_RT34.rot.m22() = _camera_RT(2, 2); camera_RT34.trans.z = _camera_RT(2, 3);

	_depth_processor.get_map_texture2(depth_object, _depth_para.fx / 2, _depth_para.fy / 2, _depth_para.cx / 2, _depth_para.cy / 2, camera_RT34);
}

void DataParser::TransformDistance(unsigned char *label_image, int mask_th, int width, int height, std::vector<float> &realDT, std::vector<float> &DTTps, std::vector<int> &ADTTps, std::vector<int> &realADT,
	std::vector<float> &v, std::vector<float> &z)
{
	//        #pragma omp parallel
	{
		//            #pragma omp for
		for (int i = 0; i < width*height; ++i)
		{
			if (label_image[i] < mask_th)
				realDT[i] = FLT_MAX;
			else
				realDT[i] = 0.0f;
		}

		/////////////////////////////////////////////////////////////////
		/// DT and ADT
		/////////////////////////////////////////////////////////////////

		//First PASS (rows)
		//            #pragma omp for
		for (int row = 0; row < height; ++row)
		{
			unsigned int k = 0;
			unsigned int indexpt1 = row*width;
			v[indexpt1] = 0;
			z[indexpt1] = FLT_MIN;
			z[indexpt1 + 1] = FLT_MAX;
			for (int q = 1; q < width; ++q)
			{
				float sp1 = float(realDT[(indexpt1 + q)] + (q*q));
				unsigned int index2 = indexpt1 + k;
				unsigned int vk = v[index2];
				float s = (sp1 - float(realDT[(indexpt1 + vk)] + (vk*vk))) / float((q - vk) << 1);
				while (s <= z[index2] && k > 0)
				{
					k--;
					index2 = indexpt1 + k;
					vk = v[index2];
					s = (sp1 - float(realDT[(indexpt1 + vk)] + (vk*vk))) / float((q - vk) << 1);
				}
				k++;
				index2 = indexpt1 + k;
				v[index2] = q;
				z[index2] = s;
				z[index2 + 1] = FLT_MAX;
			}
			k = 0;
			for (int q = 0; q < width; ++q)
			{
				while (z[indexpt1 + k + 1] < q)
					k++;
				unsigned int index2 = indexpt1 + k;
				unsigned int vk = v[index2];
				float tp1 = float(q) - float(vk);
				DTTps[indexpt1 + q] = tp1*tp1 + float(realDT[(indexpt1 + vk)]);
				ADTTps[indexpt1 + q] = indexpt1 + vk;
			}
		}

		//--- Second PASS (columns)
		//            #pragma omp for
		for (int col = 0; col < width; ++col)
		{
			unsigned int k = 0;
			unsigned int indexpt1 = col*height;
			v[indexpt1] = 0;
			z[indexpt1] = FLT_MIN;
			z[indexpt1 + 1] = FLT_MAX;
			for (int row = 1; row < height; ++row)
			{
				float sp1 = float(DTTps[col + row*width] + (row*row));
				unsigned int index2 = indexpt1 + k;
				unsigned int vk = v[index2];
				float s = (sp1 - float(DTTps[col + vk*width] + (vk*vk))) / float((row - vk) << 1);
				while (s <= z[index2] && k > 0)
				{
					k--;
					index2 = indexpt1 + k;
					vk = v[index2];
					s = (sp1 - float(DTTps[col + vk*width] + (vk*vk))) / float((row - vk) << 1);
				}
				k++;
				index2 = indexpt1 + k;
				v[index2] = row;
				z[index2] = s;
				z[index2 + 1] = FLT_MAX;
			}
			k = 0;
			for (int row = 0; row < height; ++row)
			{
				while (z[indexpt1 + k + 1] < row)
					k++;
				unsigned int index2 = indexpt1 + k;
				unsigned int vk = v[index2];

				realADT[col + row*width] = ADTTps[col + vk*width];
			}
		}
	} ///< OPENMP
};

cv::Mat DataParser::extract_hand_object_by_marker3(cv::Mat& depth_org, cv::Mat& color_org/*, cv::Mat& hand_object_silhouette*//*, cv::Mat &wband_cut,*/ /*bool& is_wristband_found, *//*bool& has_useful_data,*//* Eigen::Vector3f& wband_center,*/ /*Eigen::Vector3f& wband_dir,*/ /*cv::Mat& mask_wristband_temp*/)
{
	bool is_wristband_found = false;
	bool has_useful_data = true;
	cv::Mat mask_wristband_temp;
	cv::Mat hand_object_silhouette;

	cv::Mat depth = depth_org.clone();
	cv::Mat color = color_org.clone();

	cv::Scalar hsv_min = cv::Scalar(90, 100, 0);// cv::Scalar(94, 111, 20)
	cv::Scalar hsv_max = cv::Scalar(120, 255, 100);// cv::Scalar(120, 255, 80/60)

	cv::Scalar rgb_min = cv::Scalar(76, 73, 97);
	cv::Scalar rgb_max = cv::Scalar(130, 130, 170);//(129, 132, 167)
	float wband_size = 30;
	float depth_range = 150;		// 200;

	///--- We look for wristband up to here...
	float depth_farplane = 700;
	float depth_nearplane = 100;

	float crop_radius = 150;		// 200;

	///--- Allocated once
	cv::Mat color_hsv;
	cv::Mat in_z_range;

	{
		cv::cvtColor(color, color_hsv, CV_RGB2HSV);
		cv::inRange(color_hsv, hsv_min, hsv_max, /*=*/ mask_wristband_temp);
		cv::inRange(depth, depth_nearplane, depth_farplane /*mm*/, /*=*/ in_z_range);
		cv::bitwise_and(mask_wristband_temp, in_z_range, mask_wristband_temp);
	}

	{
		cv::Mat labels, stats, centroids;
		int num_components = cv::connectedComponentsWithStats(mask_wristband_temp, labels, stats, centroids, 4 /*connectivity={4,8}*/);

		///--- Generate array to sort
		std::vector< int > to_sort(num_components);
		std::iota(to_sort.begin(), to_sort.end(), 0 /*start from*/);

		///--- Sort accoding to area
		auto lambda = [stats](int i1, int i2) {
			int area1 = stats.at<int>(i1, cv::CC_STAT_AREA);
			int area2 = stats.at<int>(i2, cv::CC_STAT_AREA);
			return area1 > area2;
		};
		std::sort(to_sort.begin(), to_sort.end(), lambda);

		if (num_components < 2 /*not found anything beyond background*/) {
			has_useful_data = false;
		}
		else
		{
			if (has_useful_data == false) {
				//std::cout << "NEW useful data => reinit" << std::endl;
				//trivial_detector->exec(frame, sensor_silhouette);
			}
			has_useful_data = true;

			///--- Select 2nd biggest component
			mask_wristband_temp = (labels == to_sort[1]);
			is_wristband_found = true;
		}
	}

	//if (_settings.show_wband || 1) {
		//cv::imshow("show_wband", mask_wristband_temp);
	//	cv::waitKey(1);
	//}
	//else
	//	cv::destroyWindow("show_wband");

	{
		///--- Extract wristband average depth
		std::pair<float, int> avg;
		for (int row = 0; row < mask_wristband_temp.rows; ++row) {
			for (int col = 0; col < mask_wristband_temp.cols; ++col) {
				float depth_wrist = depth.at<ushort>(row, col);
				if (mask_wristband_temp.at<uchar>(row, col) == 255) {
					if (depth_wrist>depth_nearplane&&depth_wrist<depth_farplane) {
						avg.first += depth_wrist;
						avg.second++;
					}
				}
			}
		}
		ushort depth_wrist = (avg.second == 0) ? depth_nearplane : avg.first / avg.second;

		///--- First just extract pixels at the depth range of the wrist
		cv::inRange(depth, depth_wrist - depth_range, /*mm*/
			depth_wrist + depth_range, /*mm*/
			hand_object_silhouette /*=*/);
	}

	Eigen::Vector3f wband_center = Eigen::Vector3f(0, 0, 0);
	Eigen::Vector3f wband_dir = Eigen::Vector3f(0, 0, -1);
	// TIMED_BLOCK(timer,"Worker_classify::(PCA)")
	{
		///--- Compute MEAN
		int counter = 0;
		for (int row = 0; row < mask_wristband_temp.rows; ++row) {
			for (int col = 0; col < mask_wristband_temp.cols; ++col) {
				if (mask_wristband_temp.at<uchar>(row, col) != 255) continue;
				wband_center += depth_to_point(depth, col, row);
				counter++;
			}
		}
		wband_center /= counter;
		std::vector<Eigen::Vector3f> pts; pts.push_back(wband_center);

		///--- Compute Covariance
		std::vector<Eigen::Vector3f> points_pca;
		points_pca.reserve(100000);
		points_pca.clear();
		for (int row = 0; row < hand_object_silhouette.rows; ++row) {//sensor_silhouette
			for (int col = 0; col < hand_object_silhouette.cols; ++col) {
				if (hand_object_silhouette.at<uchar>(row, col) != 255) continue;
				Eigen::Vector3f p_pixel = depth_to_point(depth, col, row);
				if ((p_pixel - wband_center).norm() < 100) {//100
															// sensor_silhouette.at<uchar>(row,col) = 255;
					points_pca.push_back(p_pixel);
				}
				else {
					// sensor_silhouette.at<uchar>(row,col) = 0;
				}
			}
		}
		if (points_pca.size() == 0) return cv::Mat(240,320,CV_8U, cv::Scalar(0));
		///--- Compute PCA
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> points_mat(points_pca[0].data(), 3, points_pca.size());
		for (int i : {0, 1, 2})
			points_mat.row(i).array() -= wband_center(i);
		Eigen::Matrix3f cov = points_mat*points_mat.adjoint();
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
		wband_dir = eig.eigenvectors().col(2);

		///--- Allow wrist to point downward
		if (wband_dir.y() < 0)
			wband_dir = -wband_dir;

			/*if (_sequence_id == 0)
				wband_dir = -wband_dir;*/

		/*Eigen::Vector3f wband_dir_temp = wband_dir;

		int suit_number = 0;
		if (wband_dir_array.size() < 10)
		wband_dir_array.push_back(wband_dir);
		else
		{
		for (int i = 0; i < wband_dir_array.size(); i++)
		{
		if (wband_dir_array[i].dot(wband_dir) > 0.5)
		suit_number++;
		}

		if (suit_number < 5)
		wband_dir = -wband_dir;

		wband_dir_array.pop_front();
		wband_dir_array.push_back(wband_dir_temp);
		}*/

	}

	{
		wband_size = 10;//50
		float crop_radius_sq = crop_radius*crop_radius;
		//float displce_dist = 280;
		Eigen::Vector3f crop_center = wband_center + wband_dir * (crop_radius - wband_size);// wband_dir * 270;

		for (int row = 0; row < hand_object_silhouette.rows; ++row) {
			for (int col = 0; col < hand_object_silhouette.cols; ++col) {
				if (hand_object_silhouette.at<uchar>(row, col) != 255) continue;

				Eigen::Vector3f p_pixel = depth_to_point(depth, col, row);
				if (hand_object_silhouette.at<uchar>(row, col) == 255)
				{
					if ((p_pixel - crop_center).squaredNorm() < crop_radius_sq)
						hand_object_silhouette.at<uchar>(row, col) = 255;
					else
						hand_object_silhouette.at<uchar>(row, col) = 0;
				}
			}
		}
	}

	return hand_object_silhouette;
}

void DataParser::obtain_hand_object_silhouette()
{
	/*cv::Mat wband = cv::Mat(_depth_mm_320_buffer.size(), CV_8U, cv::Scalar(0));
	bool is_wristband_found;
	bool has_useful_data;
	Eigen::Vector3f wband_center;
	Eigen::Vector3f wband_dir;
	cv::Mat mask_wristband_temp;*/

	//extract_hand_object_by_marker(_depth_mm_320_buffer, _aligned_color_bgr_320_buffer, _hand_object_silhouette_buffer, wband, is_wristband_found, has_useful_data, wband_center, wband_dir, mask_wristband_temp);

	_hand_object_silhouette_pre = extract_hand_object_by_marker3(_depth_mm_320_pre, _aligned_color_bgr_320);
}

void DataParser::cal_ADT()
{
	TransformDistance(_hand_object_silhouette_seg_out.data, 125, width, height, realDT, DTTps, ADTTps, realADT_seg, v, z);
}

void unify(cv::Mat &a) {
	//	cv::Mat b = a.clone();
	for (cv::Mat_<uchar>::iterator it = a.begin<uchar>(); it != a.end<uchar>(); it++) {
		if (*it < 20)
			*it = 0;
		else if (*it > 180)
			*it = 2;
		else
			*it = 1;
	}
	//	return b;
}

using namespace boost::archive::iterators;

bool DataParser::Base64Decode(const string & input, string * output)
{
	typedef transform_width<binary_from_base64<string::const_iterator>, 8, 6> Base64DecodeIterator;
	stringstream result;
	try {
		copy(Base64DecodeIterator(input.begin()), Base64DecodeIterator(input.end()), ostream_iterator<char>(result));
	}
	catch (...) {
		return false;
	}
	*output = result.str();
	return output->empty() == false;
}

cv::Mat DataParser::hand_object_segment_320_id_init_pure()
{
	//cv::Mat hand_object_depth_silhouette;
	//_depth_org_320.copyTo(hand_object_depth_silhouette, _hand_object_silhouette);


	std::string frame_flag = "1";

	std::vector<unsigned char> hand_object_depth_silhouette_compressed;

	cv::Mat _depth_for_pred;
	_depth_org_320_seg_in.copyTo(_depth_for_pred, _hand_object_silhouette_seg_in);

	//	if (_sequence_id == 0)
	cv::imencode(".png", _depth_for_pred, hand_object_depth_silhouette_compressed);
	//	else
	//		cv::imencode(".png", _depth_org_320_right, hand_object_depth_silhouette_compressed);

	size_t size = hand_object_depth_silhouette_compressed.size();

	/*********************************************************/
	/* transfer the pakage to the server to get segmentation */

	chunk1.size = 0;

	CURL *curl;

	struct curl_httppost *formpost = NULL;
	struct curl_httppost *lastptr = NULL;

	//curl_formadd(&formpost,
	//	&lastptr,
	//	CURLFORM_COPYNAME, "depth",
	//	CURLFORM_BUFFER, "depth.png",
	//	CURLFORM_BUFFERPTR, hand_object_depth_silhouette_compressed.data(),
	//	CURLFORM_BUFFERLENGTH, size,
	//	CURLFORM_END);

	curl = curl_easy_init();

	if (curl) {

		form = curl_mime_init(curl);

		field = curl_mime_addpart(form);
		curl_mime_name(field, "depth");
		curl_mime_filename(field, "depth.png");
		curl_mime_data(field, (char *)hand_object_depth_silhouette_compressed.data(), size);

		field = curl_mime_addpart(form);
		curl_mime_name(field, "id");
		curl_mime_data(field, camera_id.c_str(), CURL_ZERO_TERMINATED);

		field = curl_mime_addpart(form);
		curl_mime_name(field, "begin");
		curl_mime_data(field, frame_flag.c_str(), CURL_ZERO_TERMINATED);

		struct curl_slist *list = NULL;
		list = curl_slist_append(list, "Expect: ");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

		curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8080");//http://166.111.81.29:8080   http://166.111.81.15:8080
																	 //		curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
		curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

		/* send all data to this function  */
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback2);

		/* we pass our 'chunk' struct to the callback function */
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk1);

		CURLcode ret = curl_easy_perform(curl);
	}

	reader.parse(chunk1.memory, chunk1.memory + chunk1.size, json_value);

	/*std::vector<char> data(chunk1.memory, chunk1.memory + chunk1.size);
	cv::Mat segimage = cv::imdecode(data, CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat segimage320 = cv::Mat(cv::Size(320, 240), CV_8U, cv::Scalar(0));

	for (int r = 0; r<segimage320.rows; r++)
	for (int c = 0; c < segimage320.cols; c++)
	{
	segimage320.at<unsigned char>(r, c) = segimage.at<unsigned char>(r , c );
	}*/

	//if (_sequence_id == 0)
	//	cv::imshow("left result", segimage);
	//else
	//	cv::imshow("right result", segimage);

	//cv::waitKey(1);

	/* always cleanup */
	curl_easy_cleanup(curl);

	/******************************************************************/
	/*                             *                                  */
	/******************************************************************/
	/*cv::Mat hand_object_depth_silhouette;
	_depth_org_320.copyTo(hand_object_depth_silhouette, _hand_object_silhouette);*/

	//_segmentation = segimage320.clone();

	/*return segimage320;*/

	std::string joints_string = json_value["joints"].asString();
	std::string segment_string = json_value["segment"].asString();

	std::string joints_byte_decode;
	Base64Decode(joints_string, &joints_byte_decode);
	std::vector<float> joints_array;
	joints_array.resize(joints_byte_decode.size() / 4);
	memcpy((void *)joints_array.data(), (void *)joints_byte_decode.c_str(), joints_byte_decode.size());
	//resize uv_320 to uv_640
	_joints_uvz.resize(joints_array.size());
	for (int i = 0; i < joints_array.size() / 3; i++)
	{
		_joints_uvz[3 * i] = 2 * joints_array[3 * i];
		_joints_uvz[3 * i + 1] = 2 * joints_array[3 * i + 1];
		_joints_uvz[3 * i + 2] = joints_array[3 * i + 2];
	}

	string segment_string_decode;
	Base64Decode(segment_string, &segment_string_decode);
	std::vector<uchar> seg_uchar = std::vector<uchar>(segment_string_decode.begin(), segment_string_decode.end());

	cv::Mat segimage = cv::imdecode(seg_uchar, CV_LOAD_IMAGE_UNCHANGED);

	/*cv::imshow("seg org", segimage);
	cv::waitKey(3);*/

	//cv::Mat segimage320;

	//segimage.copyTo(segimage320, _hand_object_silhouette);

	///*unify(segimage320);
	//bool is_valid;
	//cv::Mat segimage_out = process_pipeline.process(segimage320, is_valid, true, 0.6);*/

	//_segmentation = segimage320.clone();

	return segimage;
}

cv::Mat DataParser::hand_object_segment_320_id_pure()
{
	//cv::Mat hand_object_depth_silhouette;
	//_depth_org_320.copyTo(hand_object_depth_silhouette, _hand_object_silhouette);


	std::string frame_flag = "0";

	std::vector<unsigned char> hand_object_depth_silhouette_compressed;

	cv::Mat _depth_for_pred;
	_depth_org_320_seg_in.copyTo(_depth_for_pred, _hand_object_silhouette_seg_in);

	cv::imencode(".png", _depth_for_pred, hand_object_depth_silhouette_compressed);

	size_t size = hand_object_depth_silhouette_compressed.size();

	/*********************************************************/
	/* transfer the pakage to the server to get segmentation */

	chunk1.size = 0;

	CURL *curl;

	struct curl_httppost *formpost = NULL;
	struct curl_httppost *lastptr = NULL;

	//curl_formadd(&formpost,
	//	&lastptr,
	//	CURLFORM_COPYNAME, "depth",
	//	CURLFORM_BUFFER, "depth.png",
	//	CURLFORM_BUFFERPTR, hand_object_depth_silhouette_compressed.data(),
	//	CURLFORM_BUFFERLENGTH, size,
	//	CURLFORM_END);

	curl = curl_easy_init();

	if (curl) {

		form = curl_mime_init(curl);

		field = curl_mime_addpart(form);
		curl_mime_name(field, "depth");
		curl_mime_filename(field, "depth.png");
		curl_mime_data(field, (char *)hand_object_depth_silhouette_compressed.data(), size);

		field = curl_mime_addpart(form);
		curl_mime_name(field, "id");
		curl_mime_data(field, camera_id.c_str(), CURL_ZERO_TERMINATED);

		field = curl_mime_addpart(form);
		curl_mime_name(field, "begin");
		curl_mime_data(field, frame_flag.c_str(), CURL_ZERO_TERMINATED);

		struct curl_slist *list = NULL;
		list = curl_slist_append(list, "Expect: ");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

		curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8080");//http://166.111.81.29:8080   http://166.111.81.15:8080
																		 //		curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
		curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

		/* send all data to this function  */
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback2);

		/* we pass our 'chunk' struct to the callback function */
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk1);

		CURLcode ret = curl_easy_perform(curl);
	}

	reader.parse(chunk1.memory, chunk1.memory + chunk1.size, json_value);

	/*std::vector<char> data(chunk1.memory, chunk1.memory + chunk1.size);
	cv::Mat segimage = cv::imdecode(data, CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat segimage320 = cv::Mat(cv::Size(320, 240), CV_8U, cv::Scalar(0));

	for (int r = 0; r<segimage320.rows; r++)
		for (int c = 0; c < segimage320.cols; c++)
		{
			segimage320.at<unsigned char>(r, c) = segimage.at<unsigned char>(r , c );
		}*/

	//cv::waitKey(1);

	/* always cleanup */
	curl_easy_cleanup(curl);

	/******************************************************************/
	/*                             *                                  */
	/******************************************************************/
	/*cv::Mat hand_object_depth_silhouette;
	_depth_org_320.copyTo(hand_object_depth_silhouette, _hand_object_silhouette);*/

	//_segmentation = segimage320.clone();

	/*return segimage320;*/

	std::string joints_string = json_value["joints"].asString();
	std::string segment_string = json_value["segment"].asString();

	std::string joints_byte_decode;
	Base64Decode(joints_string, &joints_byte_decode);
	std::vector<float> joints_array;
	joints_array.resize(joints_byte_decode.size() / 4);
	memcpy((void *)joints_array.data(), (void *)joints_byte_decode.c_str(), joints_byte_decode.size());
	//resize uv_320 to uv_640
	_joints_uvz.resize(joints_array.size());
	for (int i = 0; i < joints_array.size() / 3; i++)
	{
		_joints_uvz[3 * i] = 2 * joints_array[3 * i];
		_joints_uvz[3 * i + 1] = 2 * joints_array[3 * i + 1];
		_joints_uvz[3 * i + 2] = joints_array[3 * i + 2];
	}

	string segment_string_decode;
	Base64Decode(segment_string, &segment_string_decode);
	std::vector<uchar> seg_uchar = std::vector<uchar>(segment_string_decode.begin(), segment_string_decode.end());

	cv::Mat segimage = cv::imdecode(seg_uchar, CV_LOAD_IMAGE_UNCHANGED);

	/*cv::imshow("seg org", segimage);
	cv::waitKey(3);*/

	//cv::Mat segimage320;

	//segimage.copyTo(segimage320, _hand_object_silhouette);

	///*unify(segimage320);
	//bool is_valid;
	//cv::Mat segimage_out = process_pipeline.process(segimage320, is_valid, true, 0.6);*/

	//_segmentation = segimage320.clone();

	return segimage;
}

void DataParser::HandObjectSil_CalADT()
{
	/*LONGLONG count_start, count_end;

	QueryPerformanceCounter(&time_stmp);
	count_start = time_stmp.QuadPart;*/

	obtain_hand_object_silhouette();
	cal_ADT();

	/*QueryPerformanceCounter(&time_stmp);
	count_end = time_stmp.QuadPart;
	count_interv = (double)(count_end - count_start);
	time_inter = count_interv * 1000 / count_freq;

	if (_sequence_id == 0)
	std::cout << "left hand-object extraction" << "	" << time_inter << std::endl;
	else
	std::cout << "right hand-object extraction" << "	" << time_inter << std::endl;*/
}

void DataParser::HandObjectSegmentation_init()
{
	cv::Mat segment_temp = hand_object_segment_320_id_init_pure();

	_segmentation_org_seg = segment_temp.clone();
	//cv::imshow("segment org", _segmentation_org);

	//handobjectADT_thr.join();

	/*cv::Mat segimage320;
	segment_temp.copyTo(segimage320, _hand_object_silhouette_buffer);*/

	/*unify(segimage320);
	bool is_valid;
	cv::Mat segimage_out = process_pipeline.process(segimage320, is_valid, true, 0.6);
	_segmentation = segimage_out.clone();

	if (is_valid)
	invalid_num = 0;
	else
	invalid_num++;

	if (invalid_num == 25)
	process_pipeline.reset();*/
	//cv::imshow("segment temp", segment_temp);

	//segment_temp.copyTo(_segmentation, _hand_object_silhouette);
	cv::bitwise_and(_segmentation_org_seg, _hand_object_silhouette_seg_out, _segmentation);
	/*cv::imshow("segment", _segmentation);
	cv::imshow("hand object", _hand_object_silhouette);*/

	cv::inRange(_segmentation, 200, 255, _hand_silhouette_seg);
	cv::inRange(_segmentation, 100, 180, _object_silhouette_seg);
}

void DataParser::HandObjectSegmentation()
{
	cv::Mat segment_temp = hand_object_segment_320_id_pure();

	_segmentation_org_seg = segment_temp.clone();

	//handobjectADT_thr.join();

	/*cv::Mat segimage320;
	segment_temp.copyTo(segimage320, _hand_object_silhouette_buffer);*/

	/*unify(segimage320);
	bool is_valid;
	cv::Mat segimage_out = process_pipeline.process(segimage320, is_valid, true, 0.6);
	_segmentation = segimage_out.clone();

	if (is_valid)
	invalid_num = 0;
	else
	invalid_num++;

	if (invalid_num == 25)
	process_pipeline.reset();*/
	//cv::imshow("segment temp", segment_temp);

	//segment_temp.copyTo(_segmentation, _hand_object_silhouette);
	cv::bitwise_and(_segmentation_org_seg, _hand_object_silhouette_seg_out, _segmentation);
	/*cv::imshow("segment", _segmentation);
	cv::imshow("hand object", _hand_object_silhouette);*/

	cv::inRange(_segmentation, 200, 255, _hand_silhouette_seg);	// 200 
	cv::inRange(_segmentation, 100, 180, _object_silhouette_seg);

	//add to segment object mask
	{
		cv::Mat labels, stats, centroids;
		int num_components = cv::connectedComponentsWithStats(_object_silhouette_seg.clone(), labels, stats, centroids, 4 /*connectivity={4,8}*/);

		///--- Generate array to sort
		std::vector< int > to_sort(num_components);
		std::iota(to_sort.begin(), to_sort.end(), 0 /*start from*/);

		///--- Sort accoding to area
		auto lambda = [stats](int i1, int i2) {
			int area1 = stats.at<int>(i1, cv::CC_STAT_AREA);
			int area2 = stats.at<int>(i2, cv::CC_STAT_AREA);
			return area1 > area2;
		};
		std::sort(to_sort.begin(), to_sort.end(), lambda);

		_object_silhouette_seg = (labels == to_sort[1]);
	}
}

void DataParser::CalculateObjADT()
{
	cv::Mat object_mask = _object_silhouette_seg.clone();

	TransformDistance(object_mask.data, 125, width, height, realDT_obj, DTTps_obj, ADTTps_obj, realADT_obj_seg, v_obj, z_obj);
}

void DataParser::ObtainJoints()
{
	float fx = _depth_intr.fx;
	float fy = _depth_intr.fy;
	float cx = _depth_intr.cx;
	float cy = _depth_intr.cy;

	_joints_pred_xyz_seg.resize(_joints_uvz.size() / 3);
	_joints_pred_uv_seg.resize(_joints_uvz.size() / 3);
	for (int i = 0; i < _joints_pred_xyz_seg.size(); i++)
	{
		float u = _joints_uvz[3 * i + 1];
		float v = _joints_uvz[3 * i + 0];
		float z = _joints_uvz[3 * i + 2];

		float3 joint_xyz;
		joint_xyz.x = z * (u - cx) / fx;
		joint_xyz.y = z * (v - cy) / fy;
		joint_xyz.z = z;

		_joints_pred_xyz_seg[i] = joint_xyz;

		float2 joint_uv;
		joint_uv.x = u;
		joint_uv.y = v;

		_joints_pred_uv_seg[i] = joint_uv;
	}
}

cv::Mat DataParser::get_segmentation_color()
{
	cv::Mat seg_color;
	cv::applyColorMap(_segmentation_org.clone(), seg_color, cv::COLORMAP_JET);
	return seg_color;
}

cv::Mat DataParser::get_segmentation_greyC3()
{
	cv::Mat seg_greyC3;
	cv::cvtColor(_segmentation_org.clone(), seg_greyC3, cv::COLOR_GRAY2BGR);

	return seg_greyC3;
}

cv::Mat DataParser::get_segmentation_org()
{
	return _segmentation_org.clone();
}

cv::Mat DataParser::get_keypoint_pred_color()
{
	cv::Mat base_img = cv::Mat(240, 320, CV_8UC3, cv::Scalar(255, 255, 255));

	//cv::Mat depth_8U = _depth_mm_320 / 4;

	//cv::cvtColor(depth_8U.clone(), base_img, cv::COLOR_GRAY2BGR);

	/*cv::Mat HO_mask = _hand_object_silhouette.clone();
	cv::Mat depth_mm = _depth_mm_320.clone();
	for (int r = 0; r < HO_mask.rows; r++)
		for (int c = 0; c < HO_mask.cols; c++)
		{
			if (HO_mask.at<uchar>(r, c) > 0)
			{
				int depth_temp = depth_mm.at<unsigned short>(r, c);
				depth_temp = (depth_temp - 200) / 500.0 * 255;
				base_img.at<cv::Vec3b>(r, c) = cv::Vec3b(depth_temp, depth_temp, depth_temp);
			}
		}*/

	std::vector<float2> pred_uv = _joints_pred_uv;

	const std::vector<int> skeleton_id = { 5,6, 6,7, 7,8, 8,4,  9,10, 10,11, 11,12, 12,4,  13,14, 14,15, 15,16, 16,4,  17,18, 18,19, 19,20, 20,4, 0,1, 1,2, 2,3, 3,4 };

	//initialize color
	std::vector<float3> hand_keypoints_color;

	hand_keypoints_color.resize(21);

	float3 base_color = make_float3(0.5, 0.5, 0.5);

	for (int i = 0; i < hand_keypoints_color.size(); i++)
		hand_keypoints_color[i] = base_color;

	hand_keypoints_color[4] = base_color;

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

	//draw line
	for (int i = 0; i < skeleton_id.size(); i = i + 2)
	{
		float2 p1 = pred_uv[skeleton_id[i]];
		float2 p2 = pred_uv[skeleton_id[i + 1]];
		float3 color_temp = hand_keypoints_color[skeleton_id[i]];

		line(base_img, cv::Point((int)p1.x/2, (int)p1.y/2), cv::Point((int)p2.x/2, (int)p2.y/2), cv::Scalar(color_temp.x * 255, color_temp.y * 255, color_temp.z * 255), 2);
	}

	return base_img;
}

void de_unify(cv::Mat &a) {
	for (cv::Mat_<uchar>::iterator it = a.begin<uchar>(); it != a.end<uchar>(); it++) {
		if (*it == 1)
			*it = 127;
		else if (*it == 2)
			*it = 255;
	}
}

float iou(const cv::Mat &a, const cv::Mat &b, std::vector<int> labels = { 1, 2 }) {
	cv::Mat inter, uni;
	float iou_val = 0;
	for (int label : labels) {
		inter = (a == label & b == label) / 255;
		uni = (a == label | b == label) / 255;
		iou_val += cv::sum(inter)[0] / cv::sum(uni)[0];
	}
	return iou_val / labels.size();
}

float iou2(const cv::Mat &a, const cv::Mat &b, std::vector<int> labels = { 1, 2 }) {
	/*cv::Mat inter, uni;
	float iou_val = 0;
	for (int label : labels) {
		inter = (a == label & b == label) / 255;
		uni = (a == label | b == label) / 255;
		iou_val += cv::sum(inter)[0] / cv::sum(uni)[0];
	}
	return iou_val / labels.size();*/

	cv::Mat inter1, uni1;
	float iou_val1;
	cv::Mat inter2, uni2;
	float iou_val2;

	inter1 = (a == labels[0] & b == labels[0]) / 255;
	uni1 = (a == labels[0] | b == labels[0]) / 255;
	iou_val1 += cv::sum(inter1)[0] / cv::sum(uni1)[0];

	inter2 = (a == labels[1] & b == labels[1]) / 255;
	uni2 = (a == labels[1] | b == labels[1]) / 255;
	iou_val2 += cv::sum(inter2)[0] / cv::sum(uni2)[0];

	float iou_val = iou_val1;
	if (iou_val > iou_val2)
		iou_val = iou_val2;

	return iou_val;
}

cv::Mat TimeContinuityProcess::process(const cv::Mat frame, bool& is_valid, bool de_unified, float iou_threshold ) {
	// Denoise
	cv::Mat new_frame = frame.clone();
	//		*new_frame = *frame;
	//		*new_frame = denoise_filter(*frame);

	int last_valid_id = -1;
	ious.clear();
	for (int i = 0; i < q.size(); i++) {
		if (q[i].second) {
			ious.push_back(iou(new_frame, q[i].first));
			last_valid_id = i;
		}
	}

	//		bool is_valid = contour_valid(*new_frame);
	is_valid = true;
	if (is_valid && ious.size() > 0) {
		float iou_sum = 0;
		for (auto iu : ious)
			iou_sum += iu;
		float mean_frame_iou = iou_sum / ious.size();
		if (mean_frame_iou < iou_threshold)
			is_valid = false;
	}

	cv::Mat output;// = new cv::Mat;
	if (!is_valid && last_valid_id > -1)
		output = q[last_valid_id].first.clone();
	else
		output = (new_frame).clone();
	if (de_unified)
		de_unify(output);

	if (q.size() >= buffer_size)
		q.pop_front();
	if (is_valid)
		q.push_back({ new_frame, is_valid });

	return output;
}

cv::Mat TimeContinuityProcess::process2(const cv::Mat frame, bool& is_valid, bool de_unified, float iou_threshold) {
	// Denoise
	cv::Mat new_frame = frame.clone();
	//		*new_frame = *frame;
	//		*new_frame = denoise_filter(*frame);

	int last_valid_id = -1;
	ious.clear();
	for (int i = 0; i < q.size(); i++) {
		if (q[i].second) {
			ious.push_back(iou2(new_frame, q[i].first));
			last_valid_id = i;
		}
	}

	//		bool is_valid = contour_valid(*new_frame);
	is_valid = true;
	if (is_valid && ious.size() > 0) {
		float iou_sum = 0;
		for (auto iu : ious)
			iou_sum += iu;
		float mean_frame_iou = iou_sum / ious.size();
		if (mean_frame_iou < iou_threshold)
			is_valid = false;
	}

	cv::Mat output;// = new cv::Mat;
	if (!is_valid && last_valid_id > -1)
		output = q[last_valid_id].first.clone();
	else
		output = (new_frame).clone();
	if (de_unified)
		de_unify(output);

	if (q.size() >= buffer_size)
		q.pop_front();
	if (is_valid)
		q.push_back({ new_frame, is_valid });

	return output;
}

void TimeContinuityProcess::reset() {
	q.clear();
	ious.clear();
}

bool TimeContinuityProcess::contour_valid(const cv::Mat a, int num_mismatched_point_threshold) {
	std::vector<cv::Mat> masks;
	cv::Mat compar;
//	static std::vector<uchar> compar_data2;
	compar = (a == 2);
//	compar_data2 = std::vector<uchar>(compar.datastart, compar.dataend);
	masks.push_back(compar);// / 255);
//	cv::imshow("2", compar);

//	static std::vector<uchar> compar_data1;
	compar = (a == 1);
//	compar_data1 = std::vector<uchar>(compar.datastart, compar.dataend);
	masks.push_back(compar);// / 255);
//	cv::imshow("1", compar);
//	cv::waitKey(3);

	static std::vector<uchar> contour_uchar;
	for (auto mask : masks) {
		cv::Mat mask_temp = mask.clone();
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(mask_temp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		cv::Mat contours_mask(mask_temp.rows, mask_temp.cols, mask_temp.type(), cv::Scalar(0));
		cv::drawContours(contours_mask, contours, -1, cv::Scalar(255), CV_FILLED);
		float num_mismatched_point = cv::sum((mask != contours_mask) / 255)[0];// 255
		if (num_mismatched_point > num_mismatched_point_threshold)
			return false;
	}
	return true;
}

cv::Mat TimeContinuityProcess::denoise_filter(const cv::Mat a) {
	cv::Mat hand_mask = a == 2;
	cv::Mat object_mask = a == 1;

	cv::Mat big_k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat small_k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));

	cv::Mat new_hand_mask, new_object_mask;
	cv::morphologyEx(hand_mask, new_hand_mask, cv::MORPH_CLOSE, big_k);
	cv::morphologyEx(new_hand_mask, new_hand_mask, cv::MORPH_OPEN, small_k);
	cv::morphologyEx(object_mask, new_object_mask, cv::MORPH_CLOSE, small_k);
	cv::morphologyEx(new_object_mask, new_object_mask, cv::MORPH_OPEN, big_k);

	cv::Mat new_mat = 2 * (new_hand_mask > 0) / 255 + (new_object_mask > 0) / 255;
	return new_mat;
}