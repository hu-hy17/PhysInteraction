#pragma once
#include <Eigen/Eigen>

struct  camera_intr
{
	float fx;
	float fy;
	float cx;
	float cy;
};

struct CamerasParameters
{
	camera_intr c0;
	camera_intr c1;

	Eigen::Matrix4f camerapose_c0;
	Eigen::Matrix4f camerapose_c1;
};