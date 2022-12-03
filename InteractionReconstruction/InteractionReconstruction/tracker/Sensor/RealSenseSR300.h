#pragma once
#include <librealsense2/rs.hpp>     // Include RealSense Cross Platform API
//#include "example.hpp"              // Include short list of convenience functions for rendering

#include <string>
#include <map>
#include <algorithm>
#include <mutex>                    // std::mutex, std::lock_guard
#include <cmath>                    // std::ceil
#include <vector>

//ZH added
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

const std::string no_camera_message = "No camera connected, please connect 1 or more";
const std::string platform_camera_name = "Platform Camera";
// const std::string left_ID = "617204007612";
// const std::string right_ID = "619204001397";

class RealSenseSR300
{
	// Helper struct per pipeline
	struct view_port
	{
		std::map<int, rs2::frame> frames_per_stream;
		rs2::colorizer colorize_frame;
		rs2::pipeline pipe;
		rs2::pipeline_profile profile;
		rs2::align align_ctrl;//ZH
	};

	struct device_bias
	{
		int u_bias;
		int v_bias;
	};

	struct data_mat
	{
		cv::Mat infrare = cv::Mat(480, 640, CV_8U, cv::Scalar(0));
		cv::Mat depth = cv::Mat(480, 640, CV_16U, cv::Scalar(0));
		cv::Mat color = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));

		bool infrare_valid = 0;
		bool depth_valid = 0;
		bool color_valid = 0;

		float infrare_time = 0.0f;
		float depth_time = 0.0f;
		float color_time = 0.0f;
		/*cv::Mat infrare;
		cv::Mat depth;
		cv::Mat color;*/
	};

private:
	std::mutex _mutex;
	std::map<std::string, view_port> _devices;
	std::map<std::string, device_bias> _device_bias;//ZH add
	std::map<std::string, data_mat> _device_data;

	std::string m_left_cam_ID;
	std::string m_right_cam_ID;

public:

	RealSenseSR300(const std::string& left_cam_ID, const std::string& right_cam_ID = "")
	{
		m_left_cam_ID = left_cam_ID;
		m_right_cam_ID = right_cam_ID;
	}

	const std::string get_left_camera_id() { return m_left_cam_ID; }

	const std::string get_right_camera_id() { return m_right_cam_ID; }

	float get_depth_scale(rs2::device dev)
	{
		// Go over the device's sensors
		for (rs2::sensor& sensor : dev.query_sensors())
		{
			// Check if the sensor if a depth sensor
			if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
			{
				return dpt.get_depth_scale();
			}
		}
		throw std::runtime_error("Device does not have a depth sensor");
	}

	rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
	{
		//Given a vector of streams, we try to find a depth stream and another stream to align depth with.
		//We prioritize color streams to make the view look better.
		//If color is not available, we take another stream that (other than depth)
		rs2_stream align_to = RS2_STREAM_ANY;
		bool depth_stream_found = false;
		bool color_stream_found = false;
		for (rs2::stream_profile sp : streams)
		{
			rs2_stream profile_stream = sp.stream_type();
			if (profile_stream != RS2_STREAM_DEPTH)
			{
				if (!color_stream_found)         //Prefer color
					align_to = profile_stream;

				if (profile_stream == RS2_STREAM_COLOR)
				{
					color_stream_found = true;
				}
			}
			else
			{
				depth_stream_found = true;
			}
		}

		if (!depth_stream_found)
			throw std::runtime_error("No Depth stream available");

		if (align_to == RS2_STREAM_ANY)
			throw std::runtime_error("No stream found to align with Depth");

		return align_to;
	}

	void remove_background(cv::Mat& other_frame, const cv::Mat& depth_frame, float depth_scale, float clipping_dist)
	{
		int  height = other_frame.rows;
		int  width = other_frame.cols;

#pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				// Get the depth value of the current pixel
				auto pixels_distance = depth_scale * depth_frame.at<unsigned short>(y, x);

				// Check if the depth value is invalid (<=0) or greater than the threashold
				if (pixels_distance <= 0.f || pixels_distance > clipping_dist)
				{
					other_frame.at<cv::Vec3b>(y, x)[0] = 0;
					other_frame.at<cv::Vec3b>(y, x)[1] = 0;
					other_frame.at<cv::Vec3b>(y, x)[2] = 0;
				}
			}
		}
	}

	cv::Mat rectify_depth(const cv::Mat& depth_frame, int u_bias, int v_bias)
	{
		int  height = depth_frame.rows;
		int  width = depth_frame.cols;

		cv::Mat rectified_depth(height, width, CV_16U, cv::Scalar(0));

#pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int x_bias = x + u_bias;
				int y_bias = y + v_bias;

				if (x_bias >= 0 && x_bias < width&&y_bias >= 0 && y_bias < height)
					rectified_depth.at<unsigned short>(y_bias, x_bias) = depth_frame.at<unsigned short>(y, x);
			}
		}

		return rectified_depth;
	}

	void print_camera_parameters(const std::vector<rs2::stream_profile>& streams)
	{
		rs2::stream_profile depth_stream_prof;
		rs2::stream_profile color_stream_prof;

		for (rs2::stream_profile sp : streams)
		{
			rs2_stream profile_stream = sp.stream_type();

			rs2::video_stream_profile video_prof_stream = (rs2::video_stream_profile)sp;

			rs2_intrinsics intr = video_prof_stream.get_intrinsics();

			if (profile_stream == RS2_STREAM_DEPTH || profile_stream == RS2_STREAM_INFRARED)
			{
				printf("intrinsic of depth camera:\n");
				printf("width:%d  height:%d\n", intr.width, intr.height);
				printf("fx:%f  fy:%f  cx:%f  cy:%f\n", intr.fx, intr.fy, intr.ppx, intr.ppy);
				depth_stream_prof = sp;
			}
			if (profile_stream == RS2_STREAM_COLOR)
			{
				printf("intrinsic of color camera:\n");
				printf("width:%d  height:%d\n", intr.width, intr.height);
				printf("fx:%f  fy:%f  cx:%f  cy:%f\n", intr.fx, intr.fy, intr.ppx, intr.ppy);
				color_stream_prof = sp;
			}
		}
		rs2_extrinsics extr = depth_stream_prof.get_extrinsics_to(color_stream_prof);
		printf("extrinsic from depth camera to color camera:\n");
		printf("%f  %f  %f  %f\n", extr.rotation[0], extr.rotation[3], extr.rotation[6], extr.translation[0]);
		printf("%f  %f  %f  %f\n", extr.rotation[1], extr.rotation[4], extr.rotation[7], extr.translation[1]);
		printf("%f  %f  %f  %f\n", extr.rotation[2], extr.rotation[5], extr.rotation[8], extr.translation[2]);
	}

	void enable_device(rs2::device dev)
	{
		std::string serial_number(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
		std::lock_guard<std::mutex> lock(_mutex);

		printf("%s\n", serial_number);
		if (_devices.find(serial_number) != _devices.end())
		{
			return; //already in
		}

		// Ignoring platform cameras (webcams, etc..)
		if (platform_camera_name == dev.get_info(RS2_CAMERA_INFO_NAME))
		{
			return;
		}
		// Create a pipeline from the given device
		rs2::pipeline p;
		rs2::config c;
		c.enable_device(serial_number);
		c.enable_stream(RS2_STREAM_COLOR, 0, 640, 480, RS2_FORMAT_RGB8, 30);
		c.enable_stream(RS2_STREAM_DEPTH, 0, 640, 480, RS2_FORMAT_Z16, 30);

		//		if (serial_number != "617203000734")
		//		c.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 60);
		//		if (serial_number != "617203000734")

		// Start the pipeline with the configuration
		rs2::pipeline_profile profile = p.start(c);

		//get depth scale of each sensor
		float depth_scale = get_depth_scale(profile.get_device());
		printf("depth scale:%f\n", depth_scale);

		//ZH add
		//print the intrinsic parameters
		//		print_camera_parameters(profile.get_streams());

		// ZCW add
		std::vector<rs2::sensor> sensors = dev.query_sensors();
		std::cout << "Device consists of " << sensors.size() << " sensors:\n" << std::endl;
		int index = 0;
		// We can now iterate the sensors and print their names
		for (rs2::sensor sensor : sensors)
		{
			std::cout << "  " << index++ << " : " << sensor.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
		}

		rs2::sensor sen = sensors[1];
		sen.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
		//sen.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 1);
		//sen.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
		//sen.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 0);
		//sen.set_option(RS2_OPTION_EXPOSURE, 750);
		//sen.set_option(RS2_OPTION_WHITE_BALANCE, 5000);
		//sen.set_option(RS2_OPTION_GAIN, 80);
		//sen.set_option(RS2_OPTION_GAMMA, 300);
		// The following loop shows how to iterate over all available options
		// Starting from 0 until RS2_OPTION_COUNT (exclusive)
		//for (int i = 0; i < static_cast<int>(RS2_OPTION_COUNT); i++)
		//{
		//    rs2_option option_type = static_cast<rs2_option>(i);
		//    //SDK enum types can be streamed to get a string that represents them
		//    std::cout << "  " << i << ": " << option_type;
		//
		//    // To control an option, use the following api:
		//    // First, verify that the sensor actually supports this option
		//    if (sen.supports(option_type))
		//    {
		//        std::cout << std::endl;
		//
		//        // Get a human readable description of the option
		//        const char* description = sen.get_option_description(option_type);
		//        std::cout << "       Description   : " << description << std::endl;
		//
		//        // Get the current value of the option
		//        float current_value = sen.get_option(option_type);
		//        std::cout << "       Current Value : " << current_value << std::endl;
		//
		//        //To change the value of an option, please follow the change_sensor_option() function
		//    }
		//    else
		//    {
		//        std::cout << " is not supported" << std::endl;
		//    }
		//}

		rs2_stream align_to = RS2_STREAM_COLOR;// find_stream_to_align(profile.get_streams());
											   //		printf("align_to:%d\n", align_to);

											   // Hold it internally
		_devices.emplace(serial_number, view_port{ {},{}, p, profile,align_to });
		//		_align.emplace(serial_number, align_to);
		device_bias d_bias;
		d_bias.u_bias = 0;
		d_bias.v_bias = 0;

		if (serial_number == "617203000734")
		{
			d_bias.u_bias = 15;
			d_bias.v_bias = -21;
		}

		_device_bias.emplace(serial_number, d_bias);

		data_mat img_mat;
		img_mat.color_valid = 1;
		if (serial_number != "617203000734")
			img_mat.depth_valid = 1;

		_device_data.emplace(serial_number, img_mat);
	}

	void remove_devices(const rs2::event_information& info)
	{
		std::lock_guard<std::mutex> lock(_mutex);
		// Go over the list of devices and check if it was disconnected
		auto itr = _devices.begin();
		while (itr != _devices.end())
		{
			if (info.was_removed(itr->second.profile.get_device()))
			{
				itr = _devices.erase(itr);
			}
			else
			{
				++itr;
			}
		}
	}

	size_t device_count()
	{
		std::lock_guard<std::mutex> lock(_mutex);
		return _devices.size();
	}

	int stream_count()
	{
		std::lock_guard<std::mutex> lock(_mutex);
		int count = 0;
		for (auto&& sn_to_dev : _devices)
		{
			for (auto&& stream : sn_to_dev.second.frames_per_stream)
			{
				if (stream.second)
				{
					count++;
				}
			}
		}
		return count;
	}

	void poll_frames(int& frame_number)
	{
		std::lock_guard<std::mutex> lock(_mutex);
		int map_id = 0;
		char image_name[128] = {};
		char write_image_name[512] = {};
		bool added = 0;

		//		printf("\n\nstream_id:\n");
		// Go over all device
		for (auto&& view : _devices)
		{
			// Ask each pipeline if there are new frames available
			//			rs2::frameset frameset = view.second.pipe.wait_for_frames();
			rs2::frameset frameset;
			if (view.second.pipe.poll_for_frames(&frameset))//
			{
				if (view.first == m_left_cam_ID && added == 0)//
				{
					frame_number++;
					added = 1;
				}

//				auto processed = view.second.align_ctrl.process(frameset);
				for (int i = 0; i < frameset.size(); i++)
				{
					rs2::frame new_frame = frameset[i];
					int stream_id = new_frame.get_profile().unique_id();
//					printf("%d ", stream_id);
					view.second.frames_per_stream[stream_id] = view.second.colorize_frame.colorize(new_frame); //update view port with the new stream

					rs2::video_frame video_fra = new_frame;
					int img_height = video_fra.get_height();
					int img_width = video_fra.get_width();

					cv::Mat infrare;
					cv::Mat depth;
					cv::Mat depth_8U;
					cv::Mat color;
					cv::Mat right_colored;

					switch (new_frame.get_profile().stream_type())
					{
					case RS2_STREAM_INFRARED:
						infrare = cv::Mat(img_height, img_width, CV_8U, (unsigned char*)video_fra.get_data());
						_device_data[view.first].infrare = infrare.clone();
						_device_data[view.first].infrare_time = video_fra.get_timestamp();
//						printf("infrared clock name %d", video_fra.get_frame_timestamp_domain());
						break;
					case RS2_STREAM_DEPTH:
						depth = cv::Mat(img_height, img_width, CV_16U, (unsigned char *)video_fra.get_data());//depth_fra.get_data()
						_device_data[view.first].depth = depth.clone();
						_device_data[view.first].depth_time = video_fra.get_timestamp();
//						printf("depth clock name %d", video_fra.get_frame_timestamp_domain());
						/*depth_8U = cv::Mat(img_height, img_width, CV_8U);
						for (int r = 0; r<img_height; r++)
							for (int c = 0; c < img_width; c++)
							{
								depth_8U.at<uchar>(r, c) = depth.at<unsigned short>(r, c) / 10000.0 * 200;
							}
						sprintf(image_name, "depth %d", map_id++);
						cv::imshow(image_name, depth_8U);
						cv::waitKey(3);*/
						break;
					case RS2_STREAM_COLOR:
						color = cv::Mat(img_height, img_width, CV_8UC3, (unsigned char *)video_fra.get_data());//depth_fra.get_data()
						cv::cvtColor(color, right_colored, cv::COLOR_RGB2BGR);
						_device_data[view.first].color = right_colored.clone();
						_device_data[view.first].color_time = video_fra.get_timestamp();
//						printf("color clock name %d", video_fra.get_frame_timestamp_domain());
						/*sprintf(image_name, "color %d", map_id++);
						cv::imshow(image_name, right_colored);
						cv::waitKey(3);*/
						break;
					default:
						break;
					}
				}
			}
		}
	}

public:
	cv::Mat get_left_depth()
	{
		return _device_data[m_left_cam_ID].depth;
	}
	cv::Mat get_left_color()
	{
		return _device_data[m_left_cam_ID].color;
	}

	cv::Mat get_right_depth()
	{
		return _device_data[m_right_cam_ID].depth;
	}
	cv::Mat get_right_color()
	{
		return _device_data[m_right_cam_ID].color;
	}
};