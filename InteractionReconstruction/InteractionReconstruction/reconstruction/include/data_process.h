#pragma once
#include "opencv2/core/core.hpp" 
#include <cuda_runtime.h>
#include "pcl/safe_call.hpp"
#include "gpu/vector_operations.hpp"

void bilateralFilter(cudaTextureObject_t depth_frame_in, cudaSurfaceObject_t depth_frame_temp, int width, int height);

void gen_vmap(cudaTextureObject_t _depth_texobj,
	cudaSurfaceObject_t _vmap_surfobj,
	float _d_fx, float _d_fy, float _d_cx, float _d_cy,
	int _depth_width, int _depth_height);

void gen_vmap2(cudaTextureObject_t _depth_texobj,
	cudaSurfaceObject_t _vmap_surfobj,
	float _d_fx, float _d_fy, float _d_cx, float _d_cy,
	int _depth_width, int _depth_height, mat34 camera_pose);

void gen_nmap(cudaTextureObject_t _depth_texobj,
	cudaSurfaceObject_t _nmap_surfobj,
	float _d_fx, float _d_fy, float _d_cx, float _d_cy,
	int _depth_width, int _depth_height);

void gen_nmap2(cudaTextureObject_t _depth_texobj,
	cudaSurfaceObject_t _nmap_surfobj,
	float _d_fx, float _d_fy, float _d_cx, float _d_cy,
	int _depth_width, int _depth_height, mat34 camera_pose);

struct DepthTexSet {
	cudaTextureObject_t m_pDepth;
	cudaTextureObject_t m_pDepthVMap;
	cudaTextureObject_t m_pDepthNMap;
};

class DataProcess
{
public:
//	cv::Mat depth_image;
	int width, height;
	float fx, fy, cx, cy;

	void *m_depth_page_lock;
	DepthTexSet m_depth_tex_set;

	// working cudaArray
	struct {
		cudaArray_t depth_in_array;
		cudaArray_t depth_temp_array; // bilateral filter temp result
		cudaArray_t depth_out_array; // crop depth boundary
	} m_array_set;

	// working cudaTextureObject and cudaSurfaceObject
	struct {
		cudaTextureObject_t depth_in_texobj;
		cudaSurfaceObject_t depth_temp_surfobj;
		cudaTextureObject_t depth_temp_texobj;
		cudaSurfaceObject_t depth_out_surfobj;
		cudaTextureObject_t depth_out_texobj;
	} m_texsurf_set;

	// working cudaArray
	struct {
		cudaArray_t vmap_array;
		cudaArray_t nmap_array;
	} m_arrays;

	// working cudaTextureObject
	struct {
		cudaTextureObject_t vmap_texobj;
		cudaTextureObject_t nmap_texobj;
	} m_texobjs;

	// working cudaSurfaceObject
	struct {
		cudaSurfaceObject_t vmap_surfobj;
		cudaSurfaceObject_t nmap_surfobj;
	} m_surfobjs;

public:
	
	void initialize(int image_width, int image_height);

	DepthTexSet get_map_texture(cv::Mat& org_depth, float _fx, float _fy, float _cx, float _cy);

	DepthTexSet get_map_texture2(cv::Mat& org_depth, float _fx, float _fy, float _cx, float _cy, mat34 camera_pose);

	DataProcess() {}
	~DataProcess();
};
