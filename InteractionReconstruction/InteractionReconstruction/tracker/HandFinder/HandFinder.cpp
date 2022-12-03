#include "HandFinder.h"

#include <numeric> ///< std::iota
#include <fstream> ///< ifstream
#include "util/mylogger.h"
#include "util/opencv_wrapper.h"
#include "util/qfile_helper.h"
#include "tracker/Worker.h"
#include "tracker/Data/DataFrame.h"
#include "tracker/Data/DataStream.h"
#include "tracker/Detection/TrivialDetector.h"
//#include "tracker/Legacy/util/Util.h"
#include "./connectedComponents.h" ///< only declared in opencv3

#include "tracker/TwSettings.h"

HandFinder::HandFinder(Camera *camera) : camera(camera){
    CHECK_NOTNULL(camera);
	sensor_indicator = new int[upper_bound_num_sensor_points];

	//point_index.resize(2 * upper_bound_num_sensor_points);
	//point_cloud.resize(2 * upper_bound_num_sensor_points);

    tw_settings->tw_add(settings->show_hand, "show_hand", "group=HandFinder");
    tw_settings->tw_add(settings->show_wband, "show_wband", "group=HandFinder");
    tw_settings->tw_add(settings->wband_size, "wband_size", "group=HandFinder");
    tw_settings->tw_add(settings->depth_range, "depth_range", "group=HandFinder");

#ifdef TODO_TWEAK_WRISTBAND_COLOR
     // TwDefine(" Settings/classifier_hsv_min colormode=hls ");
     TwAddVarRW(tw_settings->anttweakbar(), "rgb_min", TW_TYPE_COLOR3F,  &_settings.hsv_min.data, "group=HandFinder");
     TwAddVarRW(tw_settings->anttweakbar(), "rgb_max", TW_TYPE_COLOR3F,  &_settings.hsv_max.data, "group=HandFinder");
#endif

    std::string path = local_file_path("wristband.txt",false/*exit*/);
    if(!path.empty()){
        std::cout << "Reading Wristband Colors from: " << path << std::endl;
        ifstream myfile(path);
        std::string dump;
        myfile >> dump; ///< "hsv_min:"
        myfile >> settings->hsv_min[0];
        myfile >> settings->hsv_min[1];
        myfile >> settings->hsv_min[2];
        myfile >> dump; ///< "hsv_max:"
        myfile >> settings->hsv_max[0];
        myfile >> settings->hsv_max[1];
        myfile >> settings->hsv_max[2];
        //std::cout << "  hsv_min: " << settings->hsv_min << std::endl;
        //std::cout << "  hsv_max: " << settings->hsv_max << std::endl;
	}
	else 
	{
		std::cerr << "(HandFinder) Error: can not load wristband file!" << std::endl;
		exit(-1);
	}
}

Vector3 point_at_depth_pixel(const cv::Mat& depth, int x, int y, Camera* camera) {
	Integer z = depth.at<unsigned short>(y, x);
	return camera->depth_to_world(x, y, z);
}

void pca_transform_inRange(cv::Mat &color, cv::Mat &depth, cv::Mat transform_matrix, cv::Scalar pca_min, cv::Scalar pca_max, cv::Mat &mask_img)
{
	if (mask_img.empty())
		mask_img = cv::Mat(color.rows, color.cols, CV_8UC1, cv::Scalar(0));
	//else
	//	mask_img = cv::Mat::zeros(color.rows, color.cols, CV_8UC1);

	for (int v = 0; v < color.rows; v++)
		for (int u = 0; u < color.cols; u++)
		{
			if (depth.at<short>(v, u) > 10)
			{
				cv::Mat rgb_temp = cv::Mat::ones(3, 1, CV_32FC1);
				rgb_temp.at<float>(0, 0) = color.at<cv::Vec3b>(v, u)[0];
				rgb_temp.at<float>(1, 0) = color.at<cv::Vec3b>(v, u)[1];
				rgb_temp.at<float>(2, 0) = color.at<cv::Vec3b>(v, u)[1];

				cv::Mat rgb_pca = transform_matrix*rgb_temp;
				//				printf("org: %f %f %f\n", rgb_temp.at<float>(0,0), rgb_temp.at<float>(1,0), rgb_temp.at<float>(2,0));
				//				printf("dst: %f %f %f\n\n", rgb_pca.at<float>(0, 0), rgb_pca.at<float>(1, 0), rgb_pca.at<float>(2, 0));
				if (rgb_pca.at<float>(0, 0) > pca_min(0) && rgb_pca.at<float>(1, 0) > pca_min(1) && rgb_pca.at<float>(2, 0) > pca_min(2)
					&& rgb_pca.at<float>(0, 0) < pca_max(0) && rgb_pca.at<float>(1, 0) < pca_max(1) && rgb_pca.at<float>(2, 0) < pca_max(2))
				{
					mask_img.at<uchar>(v, u) = 255;
				}
				//else
				//	mask_img.at<uchar>(v, u) = 0;
			}
			//else
			//	mask_img.at<uchar>(v, u) = 0;
		}
}

void HandFinder::binary_classification(cv::Mat& depth, cv::Mat& color, cv::Mat& real_color) {
    _wristband_found = false;

    TIMED_SCOPE(timer, "Worker::binary_classification");

    ///--- Fetch from settings
    cv::Scalar hsv_min = settings->hsv_min;
    cv::Scalar hsv_max = settings->hsv_max;
	cv::Scalar rgb_min = cv::Scalar(76, 73, 97);
	cv::Scalar rgb_max = cv::Scalar(130, 130, 170);//(129, 132, 167)
    Scalar wband_size = _settings.wband_size;
    Scalar depth_range= _settings.depth_range;

    ///--- We look for wristband up to here...
    Scalar depth_farplane = camera->zFar();

    Scalar crop_radius = 150;

    ///--- Allocated once
    static cv::Mat color_hsv;
    static cv::Mat in_z_range;

    // TIMED_BLOCK(timer,"Worker_classify::(convert to HSV)")
    {
        cv::cvtColor(color, color_hsv, CV_RGB2HSV);
        cv::inRange(color_hsv, hsv_min, hsv_max, /*=*/ mask_wristband);
        cv::inRange(depth, camera->zNear(), depth_farplane /*mm*/, /*=*/ in_z_range);
        cv::bitwise_and(mask_wristband, in_z_range, mask_wristband);
		//cv::imshow("mask_wristband (pre)", mask_wristband); cv::waitKey(1);
    }

    // TIMED_BLOCK(timer,"Worker_classify::(robust wrist)")
    {
        cv::Mat labels, stats, centroids;
        int num_components = cv::connectedComponentsWithStats(mask_wristband, labels, stats, centroids, 4 /*connectivity={4,8}*/);       

        ///--- Generate array to sort
        std::vector< int > to_sort(num_components);
        std::iota(to_sort.begin(), to_sort.end(), 0 /*start from*/);       

        ///--- Sort accoding to area
        auto lambda = [stats](int i1, int i2){
            int area1 = stats.at<int>(i1,cv::CC_STAT_AREA);
            int area2 = stats.at<int>(i2,cv::CC_STAT_AREA);
            return area1>area2;
        };
        std::sort(to_sort.begin(), to_sort.end(), lambda);

        if(num_components<2 /*not found anything beyond background*/){            		
            _has_useful_data = false;
        }
        else
        {
            if(_has_useful_data==false){
                //std::cout << "NEW useful data => reinit" << std::endl;
                //trivial_detector->exec(frame, sensor_silhouette);
            }
            _has_useful_data = true;
            
            ///--- Select 2nd biggest component
            mask_wristband = (labels==to_sort[1]);
            _wristband_found = true;
        }
    }

	if (_settings.show_wband||0) {
		cv::imshow("show_wband", mask_wristband);
		cv::waitKey(1);
	}		
    else
        cv::destroyWindow("show_wband");

    // TIMED_BLOCK(timer,"Worker_classify::(crop at wrist depth)")
    {
        ///--- Extract wristband average depth
        std::pair<float, int> avg;
        for (int row = 0; row < mask_wristband.rows; ++row) {
            for (int col = 0; col < mask_wristband.cols; ++col) {
                float depth_wrist = depth.at<ushort>(row,col);
                if(mask_wristband.at<uchar>(row,col)==255){
                     if(camera->is_valid(depth_wrist)){
                         avg.first += depth_wrist;
                         avg.second++;
                     }
                 }
            }
        }
        ushort depth_wrist = (avg.second==0) ? camera->zNear() : avg.first / avg.second; 

        ///--- First just extract pixels at the depth range of the wrist
        cv::inRange(depth, depth_wrist-depth_range, /*mm*/
                           depth_wrist+depth_range, /*mm*/
                           sensor_silhouette /*=*/);

		vector< vector<cv::Point> > contours;   // 轮廓     
		vector< vector<cv::Point> > filterContours; // 筛选后的轮廓  
		vector< cv::Vec4i > hierarchy;    // 轮廓的结构信息   
		contours.clear();
		hierarchy.clear();
		filterContours.clear();

		findContours(sensor_silhouette, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		// 去除伪轮廓   
		for (size_t i = 0; i < contours.size(); i++)
		{
			if (fabs(contourArea(cv::Mat(contours[i]))) > 10/*&&fabs(arcLength(Mat(contours[i]),true))<2000*/)
				filterContours.push_back(contours[i]);
		}

		sensor_silhouette.setTo(0);
		drawContours(sensor_silhouette, filterContours, -1, cv::Scalar(255, 0, 0), CV_FILLED); //8, hierarchy);

		sensor_silhouette_HO = sensor_silhouette.clone();
    }

	if(1)
	{
		cv::Mat hand_mask_color;
		cv::Scalar rgb_pca_min = cv::Scalar(100, 5, -14);// cv::Scalar(80, 5, -14);
		cv::Scalar rgb_pca_max = cv::Scalar(320, 60, 13);// cv::Scalar(330, 60, 13);

		cv::Mat transform_m = cv::Mat::ones(3, 3, CV_32FC1);
		transform_m.at<float>(0, 0) = 0.6287; transform_m.at<float>(0, 1) = 0.5691; transform_m.at<float>(0, 2) = 0.5300;
		transform_m.at<float>(1, 0) = -0.5679; transform_m.at<float>(1, 1) = -0.1296; transform_m.at<float>(1, 2) = 0.8128;
		transform_m.at<float>(2, 0) = -0.5313; transform_m.at<float>(2, 1) = 0.8120; transform_m.at<float>(2, 2) = -0.2417;

		pca_transform_inRange(real_color, depth, transform_m, rgb_pca_min, rgb_pca_max, hand_mask_color);
//		cv::imshow("hand mask color", hand_mask_color);

//		cv::inRange(real_color, rgb_min, rgb_max, hand_mask_color);
		cv::imshow("hand mask color", hand_mask_color);
//		cv::imshow("sensor_silhouette (before)", sensor_silhouette);
		cv::bitwise_and(sensor_silhouette, hand_mask_color, sensor_silhouette);
//		cv::imshow("sensor_silhouette (after)", sensor_silhouette);
		cv::bitwise_or(sensor_silhouette, mask_wristband, sensor_silhouette);
	}

    _wband_center = Vector3(0,0,0);
    _wband_dir = Vector3(0,0,-1);
    // TIMED_BLOCK(timer,"Worker_classify::(PCA)")
    {
        ///--- Compute MEAN
        int counter = 0;
        for (int row = 0; row < mask_wristband.rows; ++row){
            for (int col = 0; col < mask_wristband.cols; ++col){
                if(mask_wristband.at<uchar>(row,col)!=255) continue;
				_wband_center += point_at_depth_pixel(depth, col, row, camera);
                counter ++;
            }
        }
        _wband_center /= counter;
        std::vector<Vector3> pts; pts.push_back(_wband_center);

        ///--- Compute Covariance
        static std::vector<Vector3> points_pca;
        points_pca.reserve(100000);
        points_pca.clear();		
        for (int row = 0; row < sensor_silhouette_HO.rows; ++row){//sensor_silhouette
            for (int col = 0; col < sensor_silhouette_HO.cols; ++col){
                if(sensor_silhouette_HO.at<uchar>(row,col)!=255) continue;
				Vector3 p_pixel = point_at_depth_pixel(depth, col, row, camera);
                if((p_pixel-_wband_center).norm()<100){
                    // sensor_silhouette.at<uchar>(row,col) = 255;
                    points_pca.push_back(p_pixel);
                } else {
                    // sensor_silhouette.at<uchar>(row,col) = 0;
                }
            }
        }
        if (points_pca.size() == 0) return;
        ///--- Compute PCA
        Eigen::Map<Matrix_3xN> points_mat(points_pca[0].data(), 3, points_pca.size() );       
        for(int i : {0,1,2})
            points_mat.row(i).array() -= _wband_center(i);
        Matrix3 cov = points_mat*points_mat.adjoint();
        Eigen::SelfAdjointEigenSolver<Matrix3> eig(cov);
        _wband_dir = eig.eigenvectors().col(2);

        ///--- Allow wrist to point downward
        if(_wband_dir.y()<0)
            _wband_dir = -_wband_dir;
    }
    // TIMED_BLOCK(timer,"Worker_classify::(in sphere)")

    {
		wband_size = 10;
        Scalar crop_radius_sq = crop_radius*crop_radius;
        Vector3 crop_center = _wband_center + _wband_dir*( crop_radius - wband_size /*mm*/);
		//Vector3 crop_center = _wband_center + _wband_dir*( crop_radius + wband_size /*mm*/);

        for (int row = 0; row < sensor_silhouette.rows; ++row){
            for (int col = 0; col < sensor_silhouette.cols; ++col){
                if(sensor_silhouette.at<uchar>(row,col)!=255&& sensor_silhouette_HO.at<uchar>(row, col) != 255) continue;

				Vector3 p_pixel = point_at_depth_pixel(depth, col, row, camera);
				if (sensor_silhouette.at<uchar>(row, col) == 255)
				{
					if ((p_pixel - crop_center).squaredNorm() < crop_radius_sq)
						sensor_silhouette.at<uchar>(row, col) = 255;
					else
						sensor_silhouette.at<uchar>(row, col) = 0;
				}

				if (sensor_silhouette_HO.at<uchar>(row, col) == 255)
				{
					if ((p_pixel - crop_center).squaredNorm() < crop_radius_sq)
						sensor_silhouette_HO.at<uchar>(row, col) = 255;
					else
						sensor_silhouette_HO.at<uchar>(row, col) = 0;
				}
            }
        }
    }
	// cv::imshow("hand zone dilated before", sensor_silhouette);
	//use dilate to fill some holes
//	cv::Mat elemt = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
//	cv::dilate(sensor_silhouette, sensor_silhouette, elemt);

	sensor_silhouette = sensor_silhouette_HO.clone();

    if(_settings.show_hand||1){
        cv::imshow("show_hand", sensor_silhouette);
		cv::imshow("show hand object", sensor_silhouette_HO);
    } else {
        cv::destroyWindow("show_hand");
    }
}

void HandFinder::obtain_point_cloud(const int camera_id, const cv::Mat &silhouette, const cv::Mat &depth_map, const Eigen::Matrix4f &camera_pose, const float fx, const float fy, const float cx, const float cy)
{
	int step = 1;
	if (camera_id == 0)
	{
		point_cloud.clear(); point_index.clear();
		for (int row = silhouette.rows - 1; row >= 0; row = row - step)//--row
		{
			for (int col = 0; col < silhouette.cols; col = col + step)//++col
			{
				if (silhouette.at<uchar>(row, col) != 255) continue;

				ushort depth_temp = depth_map.at<ushort>(row, col);
				Eigen::Vector3f point_temp;// = camera->depth_to_world(col, row, depth_temp);//be careful here
				point_temp[2] = (float)depth_temp;
				point_temp[0] = point_temp[2] * (col - cx) / fx;
				point_temp[1] = point_temp[2] * (row - cy) / fy;

				Eigen::Vector4f point_eigen = camera_pose*Eigen::Vector4f(point_temp(0), point_temp(1), point_temp(2), 1);
				point_cloud.push_back(make_float3(point_eigen[0], -point_eigen[1], point_eigen[2]));
				point_index.push_back(point_cloud.size() - 1);
			}
		}
	}
	if (camera_id == 1)
	{
		point_cloud2.clear(); point_index2.clear();
		for (int row = silhouette.rows - 1; row >= 0; row = row - step)
		{
			for (int col = 0; col < silhouette.cols; col = col + step)
			{
				if (silhouette.at<uchar>(row, col) != 255) continue;

				ushort depth_temp = depth_map.at<ushort>(row, col);
				Eigen::Vector3f point_temp;// = camera->depth_to_world(col, row, depth_temp);//be careful here
				point_temp[2] = (float)depth_temp;
				point_temp[0] = point_temp[2] * (col - cx) / fx;
				point_temp[1] = point_temp[2] * (row - cy) / fy;

				Eigen::Vector4f point_eigen = camera_pose*Eigen::Vector4f(point_temp(0), point_temp(1), point_temp(2), 1);
				point_cloud2.push_back(make_float3(point_eigen[0], -point_eigen[1], point_eigen[2]));
				point_index2.push_back(point_cloud2.size() - 1);
			}
		}
	}
}

void HandFinder::obtain_camera_direction(const Eigen::Vector3f left_camera_dir, const Eigen::Vector3f right_camera_dir)
{
	camera_dir1 = make_float3(left_camera_dir(0), left_camera_dir(1), left_camera_dir(2));
	camera_dir2 = make_float3(right_camera_dir(0), right_camera_dir(1), right_camera_dir(2));
}