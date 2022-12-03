#ifndef _SHADING_BASED_WARP_SOLVER_H_
#define _SHADING_BASED_WARP_SOLVER_H_

//#include "camera.hpp"
#include "../warp_field.h"
//#include "./gpu/internal.h"
#include "opencv2/core/core.hpp"
#include <Eigen/Eigen>
#include <vector>
#include <cuda_runtime.h>
#include <pcl/gpu/containers/device_array.h>
#include <cublas_v2.h>

class ShadingBasedWarpSolver {
public:
	ShadingBasedWarpSolver(int _max_gn_iter,
						   int _icp_iter,
						   int _outer_iter,
						   int _frame_idx,
						   int _width, int _height,
						   pcl::device::Intr _camera_intrinsic,
						   Eigen::Matrix4f _camera_pose,
						   const std::string &_results_dir,
 						   cudaTextureObject_t _can_vmap_depth_view,
						   cudaTextureObject_t _can_nmap_depth_view,
///						   cudaTextureObject_t _can_vmap_color_view,
///						   cudaTextureObject_t _can_nmap_color_view,
///						   cudaTextureObject_t _amap_color_view,
						   cudaTextureObject_t _depth_vmap,
						   cudaTextureObject_t _depth_nmap,
///						   cudaTextureObject_t _normalized_color_map,
///						   cudaTextureObject_t _color_x_grad_map,
///						   cudaTextureObject_t _color_y_grad_map,
//						   const Camera &_camera,
						   WarpField &_warp_field,
///						   const pcl::gpu::DeviceArray<float> &_L,
						   cublasHandle_t _cublas_handle,
						   pcl::gpu::DeviceArray<int> &_depth_pairs_occupied_array,
						   pcl::gpu::DeviceArray<int4> &_depth_pairs_pair_array,
						   pcl::gpu::DeviceArray<int> &_depth_pairs_scan_array,
						   pcl::gpu::DeviceArray<int> &_depth_pairs_scan_storage,
						   pcl::gpu::DeviceArray<int4> &_depth_pairs_compact_array,
///						   pcl::gpu::DeviceArray<int> &_albedo_pixel_valid_array,
///						   pcl::gpu::DeviceArray<int2> &_albedo_pixel_pair_array,
///						   pcl::gpu::DeviceArray<int> &_albedo_pixel_scan_array,
///						   pcl::gpu::DeviceArray<int> &_albedo_pixel_scan_storage,
///						   pcl::gpu::DeviceArray<int2> &_albedo_pixel_compact_array,
						   pcl::gpu::DeviceArray<int4> &_depth_pairs_knn_array,
						   pcl::gpu::DeviceArray<float4> &_depth_pairs_weight_array,
///						   pcl::gpu::DeviceArray<int4> &_albedo_pixel_knn_array,
///						   pcl::gpu::DeviceArray<float4> &_albedo_pixel_weight_array,
						   pcl::gpu::DeviceArray<float> &_residual_array,
						   pcl::gpu::DeviceArray<float> &_evaluate_energy_reduce_storage,
						   pcl::gpu::DeviceArray<float> &_total_energy,
						   pcl::gpu::DeviceArray<int> &_Ii_key,
						   pcl::gpu::DeviceArray<int> &_Ii_value,
						   pcl::gpu::DeviceArray<int> &_Ii_sorted_key,
						   pcl::gpu::DeviceArray<int> &_Ii_sorted_value,
						   pcl::gpu::DeviceArray<unsigned char> &_Ii_radixsort_storage,
						   pcl::gpu::DeviceArray<int> &_Ii_offset,
						   pcl::gpu::DeviceArray<int> &_Iij_key,
						   pcl::gpu::DeviceArray<int> &_Iij_value,
						   pcl::gpu::DeviceArray<int> &_Iij_sorted_key,
						   pcl::gpu::DeviceArray<int> &_Iij_sorted_value,
						   pcl::gpu::DeviceArray<unsigned char> &_Iij_radixsort_storage,
						   pcl::gpu::DeviceArray<int> &_Iij_segment_label,
						   pcl::gpu::DeviceArray<int> &_Iij_scan,
						   pcl::gpu::DeviceArray<unsigned char> &_Iij_scan_storage,
						   pcl::gpu::DeviceArray<int> &_compact_Iij_key,
						   pcl::gpu::DeviceArray<int> &_compact_Iij_offset,
						   pcl::gpu::DeviceArray<int> &_row_offset,
						   pcl::gpu::DeviceArray<int> &_row_length,
						   pcl::gpu::DeviceArray<int> &_bin_length,
						   pcl::gpu::DeviceArray<float> &_depth_term_values,
						   pcl::gpu::DeviceArray<float> &_smooth_term_values,
///						   pcl::gpu::DeviceArray<float> &_shading_term_values,
						   pcl::gpu::DeviceArray<float> &_b_values,
						   pcl::gpu::DeviceArray<float> &_Bii_values,
						   pcl::gpu::DeviceArray<float> &_Bij_values,
						   pcl::gpu::DeviceArray<int> &_nonzero_rowscan,
						   pcl::gpu::DeviceArray<float> &_ATA_data,
						   pcl::gpu::DeviceArray<int> &_ATA_colidx,
						   pcl::gpu::DeviceArray<int> &_ATA_rowptr,
						   pcl::gpu::DeviceArray<float> &_ATb_data,
						   pcl::gpu::DeviceArray<float> &_x_pcg,
						   pcl::gpu::DeviceArray<float> &_M_inv,
						   pcl::gpu::DeviceArray<float> &_p,
						   pcl::gpu::DeviceArray<float> &_q,
						   pcl::gpu::DeviceArray<float> &_r,
						   pcl::gpu::DeviceArray<float> &_s,
						   pcl::gpu::DeviceArray<float> &_t,
						   cudaStream_t _depth_data_pairs_stream,
///						   cudaStream_t _valid_albedo_pixels_stream,
						   int *_depth_data_pairs_num,
///						   int *_valid_albedo_pixels_num,
						   pcl::gpu::DeviceArray<float> &_huber_buffer);

	void solve(ITMLib::Objects::ITMScene *scene);

private:
	int width, height;
	pcl::device::Intr camera_intrinsic;
	Eigen::Matrix4f camera_pose;
	cudaTextureObject_t can_vmap_depth_view;
	cudaTextureObject_t can_nmap_depth_view;
///	cudaTextureObject_t can_vmap_color_view;
///	cudaTextureObject_t can_nmap_color_view;
///	cudaTextureObject_t amap_color_view;

	cudaTextureObject_t depth_vmap;
	cudaTextureObject_t depth_nmap;
///	cudaTextureObject_t normalized_color_map;
///	cudaTextureObject_t color_x_grad_map;
///	cudaTextureObject_t color_y_grad_map;

//	const Camera &camera;
	WarpField &warp_field;
///	const pcl::gpu::DeviceArray<float> &L;

	float smooth_coeffs = 5.0f;
	float shading_coeffs = 0.0001f;

	/*max Guass-Newton iteration*/
	int max_gn_iter = 0;

	/*debug values*/
	int icp_iter;
	int frame_idx;
	int outer_iter;

	/*results directory*/
	std::string m_results_dir;

	// cublas handle for pcg solver
	cublasHandle_t m_cublas_handle;

	//////////////////////////////////////////////////////////////////////////
	// preallocated buffers used in solver

	// used in associate_depth_data_pairs_dev
	pcl::gpu::DeviceArray<int> &m_depth_pairs_occupied_array;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_pair_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_storage;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_compact_array;

	// used in find_valid_albedo_pixels_dev
	///ZH
#if 0
	pcl::gpu::DeviceArray<int> &m_albedo_pixel_valid_array;
	pcl::gpu::DeviceArray<int2> &m_albedo_pixel_pair_array;
	pcl::gpu::DeviceArray<int> &m_albedo_pixel_scan_array;
	pcl::gpu::DeviceArray<int> &m_albedo_pixel_scan_storage;
	pcl::gpu::DeviceArray<int2> &m_albedo_pixel_compact_array;
#endif
	// used in query_depth_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_knn_array;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_weight_array;

	// used in query_albedo_data_knn_and_weight
///	pcl::gpu::DeviceArray<int4> &m_albedo_pixel_knn_array;
///	pcl::gpu::DeviceArray<float4> &m_albedo_pixel_weight_array;

	// used in evaluate_energy_device
	pcl::gpu::DeviceArray<float> &m_residual_array;
	pcl::gpu::DeviceArray<float> &m_evaluate_energy_reduce_storage;
	pcl::gpu::DeviceArray<float> &m_total_energy;

	// used in calculate_Ii_and_Iij
	pcl::gpu::DeviceArray<int> &m_Ii_key;
	pcl::gpu::DeviceArray<int> &m_Ii_value;

	pcl::gpu::DeviceArray<int> &m_Ii_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Ii_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Ii_radixsort_storage;
	pcl::gpu::DeviceArray<int> &m_Ii_offset;

	pcl::gpu::DeviceArray<int> &m_Iij_key;
	pcl::gpu::DeviceArray<int> &m_Iij_value;

	pcl::gpu::DeviceArray<int> &m_Iij_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Iij_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_radixsort_storage;

	pcl::gpu::DeviceArray<int> &m_Iij_segment_label;
	pcl::gpu::DeviceArray<int> &m_Iij_scan;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_scan_storage;

	pcl::gpu::DeviceArray<int> &m_compact_Iij_key;
	pcl::gpu::DeviceArray<int> &m_compact_Iij_offset;

	pcl::gpu::DeviceArray<int> &m_row_offset;
	pcl::gpu::DeviceArray<int> &m_row_length;
	pcl::gpu::DeviceArray<int> &m_bin_length;

	// used in construct_ata_atb
	pcl::gpu::DeviceArray<float> &m_depth_term_values;
	pcl::gpu::DeviceArray<float> &m_smooth_term_values;
///	pcl::gpu::DeviceArray<float> &m_shading_term_values;
	pcl::gpu::DeviceArray<float> &m_b_values;
	pcl::gpu::DeviceArray<float> &m_Bii_values;
	pcl::gpu::DeviceArray<float> &m_Bij_values;
	pcl::gpu::DeviceArray<int> &m_nonzero_rowscan;
	pcl::gpu::DeviceArray<float> &m_ATA_data;
	pcl::gpu::DeviceArray<int> &m_ATA_colidx;
	pcl::gpu::DeviceArray<int> &m_ATA_rowptr;
	pcl::gpu::DeviceArray<float> &m_ATb_data;

	// used in pcl_solver
	pcl::gpu::DeviceArray<float> &m_x_pcg;
	pcl::gpu::DeviceArray<float> &m_M_inv;
	pcl::gpu::DeviceArray<float> &m_p;
	pcl::gpu::DeviceArray<float> &m_q;
	pcl::gpu::DeviceArray<float> &m_r;
	pcl::gpu::DeviceArray<float> &m_s;
	pcl::gpu::DeviceArray<float> &m_t;

	// cudaStreams for associate_depth_data_pairs and find_valid_albedo_pixels
	cudaStream_t m_depth_data_pairs_stream;
///	cudaStream_t m_valid_albedo_pixels_stream;

	// page-locked memory for the numbers of depth_data_pairs and valid_albedo_pairs
	int *m_depth_data_pairs_num;
///	int *m_valid_albedo_pixels_num;

	// pre-allocated huber weight buffer for smooth terms
	pcl::gpu::DeviceArray<float> &m_huber_buffer;
};



/***********************************************************************************/
/*                           ShadingBasedWarpSolver2                               */
/***********************************************************************************/

class ShadingBasedWarpSolver2 {
public:
	ShadingBasedWarpSolver2(int _max_gn_iter,
		int _icp_iter,
		int _outer_iter,
		int _frame_idx,
		int _width, int _height,
		pcl::device::Intr _camera_intrinsic_c0,
		pcl::device::Intr _camera_intrinsic_c1,
		Eigen::Matrix4f _camera_pose,
		Eigen::Matrix4f _camera_view0, Eigen::Matrix4f _camera_view1,
		const std::string &_results_dir,
		cudaTextureObject_t _can_vmap_depth_view_c0,
		cudaTextureObject_t _can_nmap_depth_view_c0,
		cudaTextureObject_t _depth_vmap_c0,
		cudaTextureObject_t _depth_nmap_c0,
		cudaTextureObject_t _can_vmap_depth_view_c1,
		cudaTextureObject_t _can_nmap_depth_view_c1,
		cudaTextureObject_t _depth_vmap_c1,
		cudaTextureObject_t _depth_nmap_c1,
		WarpField &_warp_field,
		cublasHandle_t _cublas_handle,
		pcl::gpu::DeviceArray<int> &_depth_pairs_occupied_array,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_pair_array,
		pcl::gpu::DeviceArray<int> &_depth_pairs_scan_array,
		pcl::gpu::DeviceArray<int> &_depth_pairs_scan_storage,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_compact_array,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_cv_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_cn_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_dv_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_dn_f4,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_knn_array,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_weight_array,
		pcl::gpu::DeviceArray<float> &_residual_array,
		pcl::gpu::DeviceArray<float> &_evaluate_energy_reduce_storage,
		pcl::gpu::DeviceArray<float> &_total_energy,
		pcl::gpu::DeviceArray<int> &_Ii_key,
		pcl::gpu::DeviceArray<int> &_Ii_value,
		pcl::gpu::DeviceArray<int> &_Ii_sorted_key,
		pcl::gpu::DeviceArray<int> &_Ii_sorted_value,
		pcl::gpu::DeviceArray<unsigned char> &_Ii_radixsort_storage,
		pcl::gpu::DeviceArray<int> &_Ii_offset,
		pcl::gpu::DeviceArray<int> &_Iij_key,
		pcl::gpu::DeviceArray<int> &_Iij_value,
		pcl::gpu::DeviceArray<int> &_Iij_sorted_key,
		pcl::gpu::DeviceArray<int> &_Iij_sorted_value,
		pcl::gpu::DeviceArray<unsigned char> &_Iij_radixsort_storage,
		pcl::gpu::DeviceArray<int> &_Iij_segment_label,
		pcl::gpu::DeviceArray<int> &_Iij_scan,
		pcl::gpu::DeviceArray<unsigned char> &_Iij_scan_storage,
		pcl::gpu::DeviceArray<int> &_compact_Iij_key,
		pcl::gpu::DeviceArray<int> &_compact_Iij_offset,
		pcl::gpu::DeviceArray<int> &_row_offset,
		pcl::gpu::DeviceArray<int> &_row_length,
		pcl::gpu::DeviceArray<int> &_bin_length,
		pcl::gpu::DeviceArray<float> &_depth_term_values,
		pcl::gpu::DeviceArray<float> &_smooth_term_values,
		pcl::gpu::DeviceArray<float> &_b_values,
		pcl::gpu::DeviceArray<float> &_Bii_values,
		pcl::gpu::DeviceArray<float> &_Bij_values,
		pcl::gpu::DeviceArray<int> &_nonzero_rowscan,
		pcl::gpu::DeviceArray<float> &_ATA_data,
		pcl::gpu::DeviceArray<int> &_ATA_colidx,
		pcl::gpu::DeviceArray<int> &_ATA_rowptr,
		pcl::gpu::DeviceArray<float> &_ATb_data,
		pcl::gpu::DeviceArray<float> &_x_pcg,
		pcl::gpu::DeviceArray<float> &_M_inv,
		pcl::gpu::DeviceArray<float> &_p,
		pcl::gpu::DeviceArray<float> &_q,
		pcl::gpu::DeviceArray<float> &_r,
		pcl::gpu::DeviceArray<float> &_s,
		pcl::gpu::DeviceArray<float> &_t,
		cudaStream_t _depth_data_pairs_stream,
		int *_depth_data_pairs_num,
		pcl::gpu::DeviceArray<float> &_huber_buffer);

	bool solve(ITMLib::Objects::ITMScene *scene);

private:
	int width, height;
	pcl::device::Intr camera_intrinsic_c0;
	pcl::device::Intr camera_intrinsic_c1;
	Eigen::Matrix4f view_camera0 = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f view_camera1;
	Eigen::Matrix4f object_pose;

	cudaTextureObject_t can_vmap_depth_view_c0;
	cudaTextureObject_t can_nmap_depth_view_c0;
	cudaTextureObject_t can_vmap_depth_view_c1;
	cudaTextureObject_t can_nmap_depth_view_c1;

	cudaTextureObject_t depth_vmap_c0;
	cudaTextureObject_t depth_nmap_c0;
	cudaTextureObject_t depth_vmap_c1;
	cudaTextureObject_t depth_nmap_c1;

	//	const Camera &camera;
	WarpField &warp_field;
	///	const pcl::gpu::DeviceArray<float> &L;

	float smooth_coeffs = 60.0f;
	float shading_coeffs = 0.0001f;

	/*max Guass-Newton iteration*/
	int max_gn_iter = 0;

	/*debug values*/
	int icp_iter;
	int frame_idx;
	int outer_iter;

	/*results directory*/
	std::string m_results_dir;

	// cublas handle for pcg solver
	cublasHandle_t m_cublas_handle;

	//////////////////////////////////////////////////////////////////////////
	// preallocated buffers used in solver

	// used in associate_depth_data_pairs_dev
	pcl::gpu::DeviceArray<int> &m_depth_pairs_occupied_array;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_pair_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_storage;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_compact_array;

	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_cv_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_cn_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_dv_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_dn_f4;

	// used in query_depth_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_knn_array;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_weight_array;

	// used in evaluate_energy_device
	pcl::gpu::DeviceArray<float> &m_residual_array;
	pcl::gpu::DeviceArray<float> &m_evaluate_energy_reduce_storage;
	pcl::gpu::DeviceArray<float> &m_total_energy;

	// used in calculate_Ii_and_Iij
	pcl::gpu::DeviceArray<int> &m_Ii_key;
	pcl::gpu::DeviceArray<int> &m_Ii_value;

	pcl::gpu::DeviceArray<int> &m_Ii_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Ii_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Ii_radixsort_storage;
	pcl::gpu::DeviceArray<int> &m_Ii_offset;

	pcl::gpu::DeviceArray<int> &m_Iij_key;
	pcl::gpu::DeviceArray<int> &m_Iij_value;

	pcl::gpu::DeviceArray<int> &m_Iij_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Iij_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_radixsort_storage;

	pcl::gpu::DeviceArray<int> &m_Iij_segment_label;
	pcl::gpu::DeviceArray<int> &m_Iij_scan;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_scan_storage;

	pcl::gpu::DeviceArray<int> &m_compact_Iij_key;
	pcl::gpu::DeviceArray<int> &m_compact_Iij_offset;

	pcl::gpu::DeviceArray<int> &m_row_offset;
	pcl::gpu::DeviceArray<int> &m_row_length;
	pcl::gpu::DeviceArray<int> &m_bin_length;

	// used in construct_ata_atb
	pcl::gpu::DeviceArray<float> &m_depth_term_values;
	pcl::gpu::DeviceArray<float> &m_smooth_term_values;
	///	pcl::gpu::DeviceArray<float> &m_shading_term_values;
	pcl::gpu::DeviceArray<float> &m_b_values;
	pcl::gpu::DeviceArray<float> &m_Bii_values;
	pcl::gpu::DeviceArray<float> &m_Bij_values;
	pcl::gpu::DeviceArray<int> &m_nonzero_rowscan;
	pcl::gpu::DeviceArray<float> &m_ATA_data;
	pcl::gpu::DeviceArray<int> &m_ATA_colidx;
	pcl::gpu::DeviceArray<int> &m_ATA_rowptr;
	pcl::gpu::DeviceArray<float> &m_ATb_data;

	// used in pcl_solver
	pcl::gpu::DeviceArray<float> &m_x_pcg;
	pcl::gpu::DeviceArray<float> &m_M_inv;
	pcl::gpu::DeviceArray<float> &m_p;
	pcl::gpu::DeviceArray<float> &m_q;
	pcl::gpu::DeviceArray<float> &m_r;
	pcl::gpu::DeviceArray<float> &m_s;
	pcl::gpu::DeviceArray<float> &m_t;

	// cudaStreams for associate_depth_data_pairs and find_valid_albedo_pixels
	cudaStream_t m_depth_data_pairs_stream;
	///	cudaStream_t m_valid_albedo_pixels_stream;

	// page-locked memory for the numbers of depth_data_pairs and valid_albedo_pairs
	int *m_depth_data_pairs_num;
	///	int *m_valid_albedo_pixels_num;

	// pre-allocated huber weight buffer for smooth terms
	pcl::gpu::DeviceArray<float> &m_huber_buffer;
};

/***********************************************************************************/
/*                           ShadingBasedWarpSolver3                               */
/***********************************************************************************/

class ShadingBasedWarpSolver3 {
public:
	ShadingBasedWarpSolver3(int _max_gn_iter,
		int _icp_iter,
		int _outer_iter,
		int _frame_idx,
		int _width, int _height,
		pcl::device::Intr _camera_intrinsic_c0,
		pcl::device::Intr _camera_intrinsic_c1,
		Eigen::Matrix4f _camera_pose,
		Eigen::Matrix4f _camera_view0, Eigen::Matrix4f _camera_view1,
		const std::string &_results_dir,
		cudaTextureObject_t _can_vmap_depth_view_c0,
		cudaTextureObject_t _can_nmap_depth_view_c0,
		cudaTextureObject_t _depth_vmap_c0,
		cudaTextureObject_t _depth_nmap_c0,
		cudaTextureObject_t _can_vmap_depth_view_c1,
		cudaTextureObject_t _can_nmap_depth_view_c1,
		cudaTextureObject_t _depth_vmap_c1,
		cudaTextureObject_t _depth_nmap_c1,
		WarpField &_warp_field,
		cublasHandle_t _cublas_handle,
		pcl::gpu::DeviceArray<int> &_depth_pairs_occupied_array,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_pair_array,
		pcl::gpu::DeviceArray<int> &_depth_pairs_scan_array,
		pcl::gpu::DeviceArray<int> &_depth_pairs_scan_storage,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_compact_array,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_cv_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_cn_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_dv_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_dn_f4,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_knn_array,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_weight_array,
		pcl::gpu::DeviceArray<float> &_residual_array,
		pcl::gpu::DeviceArray<float> &_evaluate_energy_reduce_storage,
		pcl::gpu::DeviceArray<float> &_total_energy,
		pcl::gpu::DeviceArray<int> &_Ii_key,
		pcl::gpu::DeviceArray<int> &_Ii_value,
		pcl::gpu::DeviceArray<int> &_Ii_sorted_key,
		pcl::gpu::DeviceArray<int> &_Ii_sorted_value,
		pcl::gpu::DeviceArray<unsigned char> &_Ii_radixsort_storage,
		pcl::gpu::DeviceArray<int> &_Ii_offset,
		pcl::gpu::DeviceArray<int> &_Iij_key,
		pcl::gpu::DeviceArray<int> &_Iij_value,
		pcl::gpu::DeviceArray<int> &_Iij_sorted_key,
		pcl::gpu::DeviceArray<int> &_Iij_sorted_value,
		pcl::gpu::DeviceArray<unsigned char> &_Iij_radixsort_storage,
		pcl::gpu::DeviceArray<int> &_Iij_segment_label,
		pcl::gpu::DeviceArray<int> &_Iij_scan,
		pcl::gpu::DeviceArray<unsigned char> &_Iij_scan_storage,
		pcl::gpu::DeviceArray<int> &_compact_Iij_key,
		pcl::gpu::DeviceArray<int> &_compact_Iij_offset,
		pcl::gpu::DeviceArray<int> &_row_offset,
		pcl::gpu::DeviceArray<int> &_row_length,
		pcl::gpu::DeviceArray<int> &_bin_length,
		pcl::gpu::DeviceArray<float> &_depth_term_values,
		pcl::gpu::DeviceArray<float> &_smooth_term_values,
		pcl::gpu::DeviceArray<float> &_b_values,
		pcl::gpu::DeviceArray<float> &_Bii_values,
		pcl::gpu::DeviceArray<float> &_Bij_values,
		pcl::gpu::DeviceArray<int> &_nonzero_rowscan,
		pcl::gpu::DeviceArray<float> &_ATA_data,
		pcl::gpu::DeviceArray<int> &_ATA_colidx,
		pcl::gpu::DeviceArray<int> &_ATA_rowptr,
		pcl::gpu::DeviceArray<float> &_ATb_data,
		pcl::gpu::DeviceArray<float> &_x_pcg,
		pcl::gpu::DeviceArray<float> &_M_inv,
		pcl::gpu::DeviceArray<float> &_p,
		pcl::gpu::DeviceArray<float> &_q,
		pcl::gpu::DeviceArray<float> &_r,
		pcl::gpu::DeviceArray<float> &_s,
		pcl::gpu::DeviceArray<float> &_t,
		cudaStream_t _depth_data_pairs_stream,
		int *_depth_data_pairs_num,
		pcl::gpu::DeviceArray<float> &_huber_buffer);

	bool solve(ITMLib::Objects::ITMScene *scene, pcl::gpu::DeviceArray<float>& node_smooth_coef);

private:
	int width, height;
	pcl::device::Intr camera_intrinsic_c0;
	pcl::device::Intr camera_intrinsic_c1;
	Eigen::Matrix4f view_camera0 = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f view_camera1;
	Eigen::Matrix4f object_pose;

	cudaTextureObject_t can_vmap_depth_view_c0;
	cudaTextureObject_t can_nmap_depth_view_c0;
	cudaTextureObject_t can_vmap_depth_view_c1;
	cudaTextureObject_t can_nmap_depth_view_c1;

	cudaTextureObject_t depth_vmap_c0;
	cudaTextureObject_t depth_nmap_c0;
	cudaTextureObject_t depth_vmap_c1;
	cudaTextureObject_t depth_nmap_c1;

	//	const Camera &camera;
	WarpField &warp_field;
	///	const pcl::gpu::DeviceArray<float> &L;

	float smooth_coeffs = 15.0f;
	float shading_coeffs = 0.0001f;

	/*max Guass-Newton iteration*/
	int max_gn_iter = 0;

	/*debug values*/
	int icp_iter;
	int frame_idx;
	int outer_iter;

	/*results directory*/
	std::string m_results_dir;

	// cublas handle for pcg solver
	cublasHandle_t m_cublas_handle;

	//////////////////////////////////////////////////////////////////////////
	// preallocated buffers used in solver

	// used in associate_depth_data_pairs_dev
	pcl::gpu::DeviceArray<int> &m_depth_pairs_occupied_array;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_pair_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_storage;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_compact_array;

	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_cv_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_cn_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_dv_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_dn_f4;

	// used in query_depth_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_knn_array;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_weight_array;

	// used in evaluate_energy_device
	pcl::gpu::DeviceArray<float> &m_residual_array;
	pcl::gpu::DeviceArray<float> &m_evaluate_energy_reduce_storage;
	pcl::gpu::DeviceArray<float> &m_total_energy;

	// used in calculate_Ii_and_Iij
	pcl::gpu::DeviceArray<int> &m_Ii_key;
	pcl::gpu::DeviceArray<int> &m_Ii_value;

	pcl::gpu::DeviceArray<int> &m_Ii_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Ii_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Ii_radixsort_storage;
	pcl::gpu::DeviceArray<int> &m_Ii_offset;

	pcl::gpu::DeviceArray<int> &m_Iij_key;
	pcl::gpu::DeviceArray<int> &m_Iij_value;

	pcl::gpu::DeviceArray<int> &m_Iij_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Iij_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_radixsort_storage;

	pcl::gpu::DeviceArray<int> &m_Iij_segment_label;
	pcl::gpu::DeviceArray<int> &m_Iij_scan;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_scan_storage;

	pcl::gpu::DeviceArray<int> &m_compact_Iij_key;
	pcl::gpu::DeviceArray<int> &m_compact_Iij_offset;

	pcl::gpu::DeviceArray<int> &m_row_offset;
	pcl::gpu::DeviceArray<int> &m_row_length;
	pcl::gpu::DeviceArray<int> &m_bin_length;

	// used in construct_ata_atb
	pcl::gpu::DeviceArray<float> &m_depth_term_values;
	pcl::gpu::DeviceArray<float> &m_smooth_term_values;
	///	pcl::gpu::DeviceArray<float> &m_shading_term_values;
	pcl::gpu::DeviceArray<float> &m_b_values;
	pcl::gpu::DeviceArray<float> &m_Bii_values;
	pcl::gpu::DeviceArray<float> &m_Bij_values;
	pcl::gpu::DeviceArray<int> &m_nonzero_rowscan;
	pcl::gpu::DeviceArray<float> &m_ATA_data;
	pcl::gpu::DeviceArray<int> &m_ATA_colidx;
	pcl::gpu::DeviceArray<int> &m_ATA_rowptr;
	pcl::gpu::DeviceArray<float> &m_ATb_data;

	// used in pcl_solver
	pcl::gpu::DeviceArray<float> &m_x_pcg;
	pcl::gpu::DeviceArray<float> &m_M_inv;
	pcl::gpu::DeviceArray<float> &m_p;
	pcl::gpu::DeviceArray<float> &m_q;
	pcl::gpu::DeviceArray<float> &m_r;
	pcl::gpu::DeviceArray<float> &m_s;
	pcl::gpu::DeviceArray<float> &m_t;

	// cudaStreams for associate_depth_data_pairs and find_valid_albedo_pixels
	cudaStream_t m_depth_data_pairs_stream;
	///	cudaStream_t m_valid_albedo_pixels_stream;

	// page-locked memory for the numbers of depth_data_pairs and valid_albedo_pairs
	int *m_depth_data_pairs_num;
	///	int *m_valid_albedo_pixels_num;

	// pre-allocated huber weight buffer for smooth terms
	pcl::gpu::DeviceArray<float> &m_huber_buffer;
};

/***********************************************************************************/
/*                           ShadingBasedWarpSolver4                               */
/***********************************************************************************/

class ShadingBasedWarpSolver4 {
public:
	ShadingBasedWarpSolver4(int _max_gn_iter,
		int _icp_iter,
		int _outer_iter,
		int _frame_idx,
		int _width, int _height,
		pcl::device::Intr _camera_intrinsic_c0,
		pcl::device::Intr _camera_intrinsic_c1,
		Eigen::Matrix4f _camera_pose,
		Eigen::Matrix4f _camera_view0, Eigen::Matrix4f _camera_view1,
		const std::string &_results_dir,
		cudaTextureObject_t _can_vmap_depth_view_c0,
		cudaTextureObject_t _can_nmap_depth_view_c0,
		cudaTextureObject_t _depth_vmap_c0,
		cudaTextureObject_t _depth_nmap_c0,
		cudaTextureObject_t _can_vmap_depth_view_c1,
		cudaTextureObject_t _can_nmap_depth_view_c1,
		cudaTextureObject_t _depth_vmap_c1,
		cudaTextureObject_t _depth_nmap_c1,
		WarpField &_warp_field,
		cublasHandle_t _cublas_handle,
		pcl::gpu::DeviceArray<int> &_depth_pairs_occupied_array,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_pair_array,
		pcl::gpu::DeviceArray<int> &_depth_pairs_scan_array,
		pcl::gpu::DeviceArray<int> &_depth_pairs_scan_storage,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_compact_array,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_cv_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_cn_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_dv_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_dn_f4,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_knn_array,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_weight_array,
		pcl::gpu::DeviceArray<int4> &_interaction_knn_array,
		pcl::gpu::DeviceArray<float4> &_interaction_weight_array,
		pcl::gpu::DeviceArray<float> &_residual_array,
		pcl::gpu::DeviceArray<float> &_evaluate_energy_reduce_storage,
		pcl::gpu::DeviceArray<float> &_total_energy,
		pcl::gpu::DeviceArray<int> &_Ii_key,
		pcl::gpu::DeviceArray<int> &_Ii_value,
		pcl::gpu::DeviceArray<int> &_Ii_sorted_key,
		pcl::gpu::DeviceArray<int> &_Ii_sorted_value,
		pcl::gpu::DeviceArray<unsigned char> &_Ii_radixsort_storage,
		pcl::gpu::DeviceArray<int> &_Ii_offset,
		pcl::gpu::DeviceArray<int> &_Iij_key,
		pcl::gpu::DeviceArray<int> &_Iij_value,
		pcl::gpu::DeviceArray<int> &_Iij_sorted_key,
		pcl::gpu::DeviceArray<int> &_Iij_sorted_value,
		pcl::gpu::DeviceArray<unsigned char> &_Iij_radixsort_storage,
		pcl::gpu::DeviceArray<int> &_Iij_segment_label,
		pcl::gpu::DeviceArray<int> &_Iij_scan,
		pcl::gpu::DeviceArray<unsigned char> &_Iij_scan_storage,
		pcl::gpu::DeviceArray<int> &_compact_Iij_key,
		pcl::gpu::DeviceArray<int> &_compact_Iij_offset,
		pcl::gpu::DeviceArray<int> &_row_offset,
		pcl::gpu::DeviceArray<int> &_row_length,
		pcl::gpu::DeviceArray<int> &_bin_length,
		pcl::gpu::DeviceArray<float> &_depth_term_values,
		pcl::gpu::DeviceArray<float> &_smooth_term_values,
		pcl::gpu::DeviceArray<float> &_interaction_term_values,
		pcl::gpu::DeviceArray<float> &_b_values,
		pcl::gpu::DeviceArray<float> &_Bii_values,
		pcl::gpu::DeviceArray<float> &_Bij_values,
		pcl::gpu::DeviceArray<int> &_nonzero_rowscan,
		pcl::gpu::DeviceArray<float> &_ATA_data,
		pcl::gpu::DeviceArray<int> &_ATA_colidx,
		pcl::gpu::DeviceArray<int> &_ATA_rowptr,
		pcl::gpu::DeviceArray<float> &_ATb_data,
		pcl::gpu::DeviceArray<float> &_x_pcg,
		pcl::gpu::DeviceArray<float> &_M_inv,
		pcl::gpu::DeviceArray<float> &_p,
		pcl::gpu::DeviceArray<float> &_q,
		pcl::gpu::DeviceArray<float> &_r,
		pcl::gpu::DeviceArray<float> &_s,
		pcl::gpu::DeviceArray<float> &_t,
		cudaStream_t _depth_data_pairs_stream,
		int *_depth_data_pairs_num,
		pcl::gpu::DeviceArray<float> &_huber_buffer);

	bool solve(ITMLib::Objects::ITMScene *scene, pcl::gpu::DeviceArray<float>& node_smooth_coef, pcl::gpu::DeviceArray<float4>& interaction_warped_vertice, pcl::gpu::DeviceArray<float4>& interaction_warped_normal,
		pcl::gpu::DeviceArray<float4>& interaction_cano_vertice, pcl::gpu::DeviceArray<float4>& interaction_cano_normal,
		pcl::gpu::DeviceArray<unsigned char>& interaction_finger_idx, pcl::gpu::DeviceArray<float4>& joint_positions, pcl::gpu::DeviceArray<float>& joint_radius);

private:
	int width, height;
	pcl::device::Intr camera_intrinsic_c0;
	pcl::device::Intr camera_intrinsic_c1;
	Eigen::Matrix4f view_camera0 = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f view_camera1;
	Eigen::Matrix4f object_pose;

	cudaTextureObject_t can_vmap_depth_view_c0;
	cudaTextureObject_t can_nmap_depth_view_c0;
	cudaTextureObject_t can_vmap_depth_view_c1;
	cudaTextureObject_t can_nmap_depth_view_c1;

	cudaTextureObject_t depth_vmap_c0;
	cudaTextureObject_t depth_nmap_c0;
	cudaTextureObject_t depth_vmap_c1;
	cudaTextureObject_t depth_nmap_c1;

	//	const Camera &camera;
	WarpField &warp_field;
	///	const pcl::gpu::DeviceArray<float> &L;

	float smooth_coeffs = 15.0f;
	float shading_coeffs = 0.0001f;
	float interaction_coeffs = 0.16f;

	/*max Guass-Newton iteration*/
	int max_gn_iter = 0;

	/*debug values*/
	int icp_iter;
	int frame_idx;
	int outer_iter;

	/*results directory*/
	std::string m_results_dir;

	// cublas handle for pcg solver
	cublasHandle_t m_cublas_handle;

	//////////////////////////////////////////////////////////////////////////
	// preallocated buffers used in solver

	// used in associate_depth_data_pairs_dev
	pcl::gpu::DeviceArray<int> &m_depth_pairs_occupied_array;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_pair_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_storage;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_compact_array;

	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_cv_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_cn_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_dv_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_dn_f4;

	// used in query_depth_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_knn_array;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_weight_array;

	// used in interaction_depth_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> &m_interaction_knn_array;
	pcl::gpu::DeviceArray<float4> &m_interaction_weight_array;

	// used in evaluate_energy_device
	pcl::gpu::DeviceArray<float> &m_residual_array;
	pcl::gpu::DeviceArray<float> &m_evaluate_energy_reduce_storage;
	pcl::gpu::DeviceArray<float> &m_total_energy;

	// used in calculate_Ii_and_Iij
	pcl::gpu::DeviceArray<int> &m_Ii_key;
	pcl::gpu::DeviceArray<int> &m_Ii_value;

	pcl::gpu::DeviceArray<int> &m_Ii_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Ii_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Ii_radixsort_storage;
	pcl::gpu::DeviceArray<int> &m_Ii_offset;

	pcl::gpu::DeviceArray<int> &m_Iij_key;
	pcl::gpu::DeviceArray<int> &m_Iij_value;

	pcl::gpu::DeviceArray<int> &m_Iij_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Iij_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_radixsort_storage;

	pcl::gpu::DeviceArray<int> &m_Iij_segment_label;
	pcl::gpu::DeviceArray<int> &m_Iij_scan;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_scan_storage;

	pcl::gpu::DeviceArray<int> &m_compact_Iij_key;
	pcl::gpu::DeviceArray<int> &m_compact_Iij_offset;

	pcl::gpu::DeviceArray<int> &m_row_offset;
	pcl::gpu::DeviceArray<int> &m_row_length;
	pcl::gpu::DeviceArray<int> &m_bin_length;

	// used in construct_ata_atb
	pcl::gpu::DeviceArray<float> &m_depth_term_values;
	pcl::gpu::DeviceArray<float> &m_smooth_term_values;
	pcl::gpu::DeviceArray<float> &m_interaction_term_values;
	///	pcl::gpu::DeviceArray<float> &m_shading_term_values;
	pcl::gpu::DeviceArray<float> &m_b_values;
	pcl::gpu::DeviceArray<float> &m_Bii_values;
	pcl::gpu::DeviceArray<float> &m_Bij_values;
	pcl::gpu::DeviceArray<int> &m_nonzero_rowscan;
	pcl::gpu::DeviceArray<float> &m_ATA_data;
	pcl::gpu::DeviceArray<int> &m_ATA_colidx;
	pcl::gpu::DeviceArray<int> &m_ATA_rowptr;
	pcl::gpu::DeviceArray<float> &m_ATb_data;

	// used in pcl_solver
	pcl::gpu::DeviceArray<float> &m_x_pcg;
	pcl::gpu::DeviceArray<float> &m_M_inv;
	pcl::gpu::DeviceArray<float> &m_p;
	pcl::gpu::DeviceArray<float> &m_q;
	pcl::gpu::DeviceArray<float> &m_r;
	pcl::gpu::DeviceArray<float> &m_s;
	pcl::gpu::DeviceArray<float> &m_t;

	// cudaStreams for associate_depth_data_pairs and find_valid_albedo_pixels
	cudaStream_t m_depth_data_pairs_stream;
	///	cudaStream_t m_valid_albedo_pixels_stream;

	// page-locked memory for the numbers of depth_data_pairs and valid_albedo_pairs
	int *m_depth_data_pairs_num;
	///	int *m_valid_albedo_pixels_num;

	// pre-allocated huber weight buffer for smooth terms
	pcl::gpu::DeviceArray<float> &m_huber_buffer;
};

/***********************************************************************************/
/*                           ShadingBasedWarpSolver4                               */
/***********************************************************************************/

class ShadingBasedWarpSolver5 {
public:
	ShadingBasedWarpSolver5(int _max_gn_iter,
		int _icp_iter,
		int _outer_iter,
		int _frame_idx,
		int _width, int _height,
		pcl::device::Intr _camera_intrinsic_c0,
		//pcl::device::Intr _camera_intrinsic_c1,
		Eigen::Matrix4f _camera_pose,
		Eigen::Matrix4f _camera_view0, /*Eigen::Matrix4f _camera_view1,*/
		const std::string &_results_dir,
		cudaTextureObject_t _can_vmap_depth_view_c0,
		cudaTextureObject_t _can_nmap_depth_view_c0,
		cudaTextureObject_t _depth_vmap_c0,
		cudaTextureObject_t _depth_nmap_c0,
		/*cudaTextureObject_t _can_vmap_depth_view_c1,
		cudaTextureObject_t _can_nmap_depth_view_c1,
		cudaTextureObject_t _depth_vmap_c1,
		cudaTextureObject_t _depth_nmap_c1,*/
		WarpField &_warp_field,
		cublasHandle_t _cublas_handle,
		pcl::gpu::DeviceArray<int> &_depth_pairs_occupied_array,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_pair_array,
		pcl::gpu::DeviceArray<int> &_depth_pairs_scan_array,
		pcl::gpu::DeviceArray<int> &_depth_pairs_scan_storage,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_compact_array,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_cv_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_cn_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_dv_f4,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_dn_f4,
		pcl::gpu::DeviceArray<int4> &_depth_pairs_knn_array,
		pcl::gpu::DeviceArray<float4> &_depth_pairs_weight_array,
		pcl::gpu::DeviceArray<int4> &_interaction_knn_array,
		pcl::gpu::DeviceArray<float4> &_interaction_weight_array,
		pcl::gpu::DeviceArray<float> &_residual_array,
		pcl::gpu::DeviceArray<float> &_evaluate_energy_reduce_storage,
		pcl::gpu::DeviceArray<float> &_total_energy,
		pcl::gpu::DeviceArray<int> &_Ii_key,
		pcl::gpu::DeviceArray<int> &_Ii_value,
		pcl::gpu::DeviceArray<int> &_Ii_sorted_key,
		pcl::gpu::DeviceArray<int> &_Ii_sorted_value,
		pcl::gpu::DeviceArray<unsigned char> &_Ii_radixsort_storage,
		pcl::gpu::DeviceArray<int> &_Ii_offset,
		pcl::gpu::DeviceArray<int> &_Iij_key,
		pcl::gpu::DeviceArray<int> &_Iij_value,
		pcl::gpu::DeviceArray<int> &_Iij_sorted_key,
		pcl::gpu::DeviceArray<int> &_Iij_sorted_value,
		pcl::gpu::DeviceArray<unsigned char> &_Iij_radixsort_storage,
		pcl::gpu::DeviceArray<int> &_Iij_segment_label,
		pcl::gpu::DeviceArray<int> &_Iij_scan,
		pcl::gpu::DeviceArray<unsigned char> &_Iij_scan_storage,
		pcl::gpu::DeviceArray<int> &_compact_Iij_key,
		pcl::gpu::DeviceArray<int> &_compact_Iij_offset,
		pcl::gpu::DeviceArray<int> &_row_offset,
		pcl::gpu::DeviceArray<int> &_row_length,
		pcl::gpu::DeviceArray<int> &_bin_length,
		pcl::gpu::DeviceArray<float> &_depth_term_values,
		pcl::gpu::DeviceArray<float> &_smooth_term_values,
		pcl::gpu::DeviceArray<float> &_interaction_term_values,
		pcl::gpu::DeviceArray<float> &_b_values,
		pcl::gpu::DeviceArray<float> &_Bii_values,
		pcl::gpu::DeviceArray<float> &_Bij_values,
		pcl::gpu::DeviceArray<int> &_nonzero_rowscan,
		pcl::gpu::DeviceArray<float> &_ATA_data,
		pcl::gpu::DeviceArray<int> &_ATA_colidx,
		pcl::gpu::DeviceArray<int> &_ATA_rowptr,
		pcl::gpu::DeviceArray<float> &_ATb_data,
		pcl::gpu::DeviceArray<float> &_x_pcg,
		pcl::gpu::DeviceArray<float> &_M_inv,
		pcl::gpu::DeviceArray<float> &_p,
		pcl::gpu::DeviceArray<float> &_q,
		pcl::gpu::DeviceArray<float> &_r,
		pcl::gpu::DeviceArray<float> &_s,
		pcl::gpu::DeviceArray<float> &_t,
		cudaStream_t _depth_data_pairs_stream,
		int *_depth_data_pairs_num,
		pcl::gpu::DeviceArray<float> &_huber_buffer);

	bool solve(ITMLib::Objects::ITMScene *scene, pcl::gpu::DeviceArray<float>& node_smooth_coef, pcl::gpu::DeviceArray<float4>& interaction_warped_vertice, pcl::gpu::DeviceArray<float4>& interaction_warped_normal,
		pcl::gpu::DeviceArray<float4>& interaction_cano_vertice, pcl::gpu::DeviceArray<float4>& interaction_cano_normal,
		pcl::gpu::DeviceArray<unsigned char>& interaction_finger_idx, pcl::gpu::DeviceArray<float4>& joint_positions, pcl::gpu::DeviceArray<float>& joint_radius, bool& stop_itr, int camera_use);

private:
	int width, height;
	pcl::device::Intr camera_intrinsic_c0;
	pcl::device::Intr camera_intrinsic_c1;
	Eigen::Matrix4f view_camera0 = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f view_camera1;
	Eigen::Matrix4f object_pose;

	cudaTextureObject_t can_vmap_depth_view_c0;
	cudaTextureObject_t can_nmap_depth_view_c0;
	cudaTextureObject_t can_vmap_depth_view_c1;
	cudaTextureObject_t can_nmap_depth_view_c1;

	cudaTextureObject_t depth_vmap_c0;
	cudaTextureObject_t depth_nmap_c0;
	cudaTextureObject_t depth_vmap_c1;
	cudaTextureObject_t depth_nmap_c1;

	//	const Camera &camera;
	WarpField &warp_field;
	///	const pcl::gpu::DeviceArray<float> &L;

	float smooth_coeffs = 15.0f;
	float shading_coeffs = 0.0001f;
	float interaction_coeffs = 0.20f;//0.3f

	/*max Guass-Newton iteration*/
	int max_gn_iter = 0;

	/*debug values*/
	int icp_iter;
	int frame_idx;
	int outer_iter;

	/*results directory*/
	std::string m_results_dir;

	// cublas handle for pcg solver
	cublasHandle_t m_cublas_handle;

	//////////////////////////////////////////////////////////////////////////
	// preallocated buffers used in solver

	// used in associate_depth_data_pairs_dev
	pcl::gpu::DeviceArray<int> &m_depth_pairs_occupied_array;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_pair_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_storage;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_compact_array;

	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_cv_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_cn_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_dv_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_dn_f4;

	// used in query_depth_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_knn_array;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_weight_array;

	// used in interaction_depth_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> &m_interaction_knn_array;
	pcl::gpu::DeviceArray<float4> &m_interaction_weight_array;

	// used in evaluate_energy_device
	pcl::gpu::DeviceArray<float> &m_residual_array;
	pcl::gpu::DeviceArray<float> &m_evaluate_energy_reduce_storage;
	pcl::gpu::DeviceArray<float> &m_total_energy;

	// used in calculate_Ii_and_Iij
	pcl::gpu::DeviceArray<int> &m_Ii_key;
	pcl::gpu::DeviceArray<int> &m_Ii_value;

	pcl::gpu::DeviceArray<int> &m_Ii_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Ii_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Ii_radixsort_storage;
	pcl::gpu::DeviceArray<int> &m_Ii_offset;

	pcl::gpu::DeviceArray<int> &m_Iij_key;
	pcl::gpu::DeviceArray<int> &m_Iij_value;

	pcl::gpu::DeviceArray<int> &m_Iij_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Iij_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_radixsort_storage;

	pcl::gpu::DeviceArray<int> &m_Iij_segment_label;
	pcl::gpu::DeviceArray<int> &m_Iij_scan;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_scan_storage;

	pcl::gpu::DeviceArray<int> &m_compact_Iij_key;
	pcl::gpu::DeviceArray<int> &m_compact_Iij_offset;

	pcl::gpu::DeviceArray<int> &m_row_offset;
	pcl::gpu::DeviceArray<int> &m_row_length;
	pcl::gpu::DeviceArray<int> &m_bin_length;

	// used in construct_ata_atb
	pcl::gpu::DeviceArray<float> &m_depth_term_values;
	pcl::gpu::DeviceArray<float> &m_smooth_term_values;
	pcl::gpu::DeviceArray<float> &m_interaction_term_values;
	///	pcl::gpu::DeviceArray<float> &m_shading_term_values;
	pcl::gpu::DeviceArray<float> &m_b_values;
	pcl::gpu::DeviceArray<float> &m_Bii_values;
	pcl::gpu::DeviceArray<float> &m_Bij_values;
	pcl::gpu::DeviceArray<int> &m_nonzero_rowscan;
	pcl::gpu::DeviceArray<float> &m_ATA_data;
	pcl::gpu::DeviceArray<int> &m_ATA_colidx;
	pcl::gpu::DeviceArray<int> &m_ATA_rowptr;
	pcl::gpu::DeviceArray<float> &m_ATb_data;

	// used in pcl_solver
	pcl::gpu::DeviceArray<float> &m_x_pcg;
	pcl::gpu::DeviceArray<float> &m_M_inv;
	pcl::gpu::DeviceArray<float> &m_p;
	pcl::gpu::DeviceArray<float> &m_q;
	pcl::gpu::DeviceArray<float> &m_r;
	pcl::gpu::DeviceArray<float> &m_s;
	pcl::gpu::DeviceArray<float> &m_t;

	// cudaStreams for associate_depth_data_pairs and find_valid_albedo_pixels
	cudaStream_t m_depth_data_pairs_stream;
	///	cudaStream_t m_valid_albedo_pixels_stream;

	// page-locked memory for the numbers of depth_data_pairs and valid_albedo_pairs
	int *m_depth_data_pairs_num;
	///	int *m_valid_albedo_pixels_num;

	// pre-allocated huber weight buffer for smooth terms
	pcl::gpu::DeviceArray<float> &m_huber_buffer;
};


/***********************************************************************************/
/*                               NonRigidWarpSolver6                               */
/***********************************************************************************/

class NonRigidWarpSolver_handblock {
public:
	NonRigidWarpSolver_handblock(int _max_gn_iter,
						int _icp_iter,
						int _outer_iter,
						int _frame_idx,
						int _width, int _height,
						pcl::device::Intr _camera_intrinsic_c0,
						//pcl::device::Intr _camera_intrinsic_c1,
						Eigen::Matrix4f _camera_pose,
						Eigen::Matrix4f _camera_view0, /*Eigen::Matrix4f _camera_view1,*/
						const std::string &_results_dir,
						cudaTextureObject_t _can_vmap_depth_view_c0,
						cudaTextureObject_t _can_nmap_depth_view_c0,
						cudaTextureObject_t _depth_vmap_c0,
						cudaTextureObject_t _depth_nmap_c0,
						/*cudaTextureObject_t _can_vmap_depth_view_c1,
						cudaTextureObject_t _can_nmap_depth_view_c1,
						cudaTextureObject_t _depth_vmap_c1,
						cudaTextureObject_t _depth_nmap_c1,*/
						WarpField &_warp_field,
						cublasHandle_t _cublas_handle,
						pcl::gpu::DeviceArray<int> &_depth_pairs_occupied_array,
						pcl::gpu::DeviceArray<int4> &_depth_pairs_pair_array,
						pcl::gpu::DeviceArray<int> &_depth_pairs_scan_array,
						pcl::gpu::DeviceArray<int> &_depth_pairs_scan_storage,
						pcl::gpu::DeviceArray<int4> &_depth_pairs_compact_array,
						pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_cv_f4,
						pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_cn_f4,
						pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_dv_f4,
						pcl::gpu::DeviceArray<float4> &_depth_pairs_compact_array_dn_f4,
						pcl::gpu::DeviceArray<int4> &_depth_pairs_knn_array,
						pcl::gpu::DeviceArray<float4> &_depth_pairs_weight_array,
						pcl::gpu::DeviceArray<int4> &_interaction_knn_array,
						pcl::gpu::DeviceArray<float4> &_interaction_weight_array,
						pcl::gpu::DeviceArray<float> &_residual_array,
						pcl::gpu::DeviceArray<float> &_evaluate_energy_reduce_storage,
						pcl::gpu::DeviceArray<float> &_total_energy,
						pcl::gpu::DeviceArray<int> &_Ii_key,
						pcl::gpu::DeviceArray<int> &_Ii_value,
						pcl::gpu::DeviceArray<int> &_Ii_sorted_key,
						pcl::gpu::DeviceArray<int> &_Ii_sorted_value,
						pcl::gpu::DeviceArray<unsigned char> &_Ii_radixsort_storage,
						pcl::gpu::DeviceArray<int> &_Ii_offset,
						pcl::gpu::DeviceArray<int> &_Iij_key,
						pcl::gpu::DeviceArray<int> &_Iij_value,
						pcl::gpu::DeviceArray<int> &_Iij_sorted_key,
						pcl::gpu::DeviceArray<int> &_Iij_sorted_value,
						pcl::gpu::DeviceArray<unsigned char> &_Iij_radixsort_storage,
						pcl::gpu::DeviceArray<int> &_Iij_segment_label,
						pcl::gpu::DeviceArray<int> &_Iij_scan,
						pcl::gpu::DeviceArray<unsigned char> &_Iij_scan_storage,
						pcl::gpu::DeviceArray<int> &_compact_Iij_key,
						pcl::gpu::DeviceArray<int> &_compact_Iij_offset,
						pcl::gpu::DeviceArray<int> &_row_offset,
						pcl::gpu::DeviceArray<int> &_row_length,
						pcl::gpu::DeviceArray<int> &_bin_length,
						pcl::gpu::DeviceArray<float> &_depth_term_values,
						pcl::gpu::DeviceArray<float> &_smooth_term_values,
						pcl::gpu::DeviceArray<float> &_interaction_term_values,
						pcl::gpu::DeviceArray<float> &_b_values,
						pcl::gpu::DeviceArray<float> &_Bii_values,
						pcl::gpu::DeviceArray<float> &_Bij_values,
						pcl::gpu::DeviceArray<int> &_nonzero_rowscan,
						pcl::gpu::DeviceArray<float> &_ATA_data,
						pcl::gpu::DeviceArray<int> &_ATA_colidx,
						pcl::gpu::DeviceArray<int> &_ATA_rowptr,
						pcl::gpu::DeviceArray<float> &_ATb_data,
						pcl::gpu::DeviceArray<float> &_x_pcg,
						pcl::gpu::DeviceArray<float> &_M_inv,
						pcl::gpu::DeviceArray<float> &_p,
						pcl::gpu::DeviceArray<float> &_q,
						pcl::gpu::DeviceArray<float> &_r,
						pcl::gpu::DeviceArray<float> &_s,
						pcl::gpu::DeviceArray<float> &_t,
						cudaStream_t _depth_data_pairs_stream,
						int *_depth_data_pairs_num,
						pcl::gpu::DeviceArray<float> &_huber_buffer);

	bool solve(ITMLib::Objects::ITMScene *scene, pcl::gpu::DeviceArray<float>& node_smooth_coef, pcl::gpu::DeviceArray<float4>& interaction_warped_vertice, pcl::gpu::DeviceArray<float4>& interaction_warped_normal,
			   pcl::gpu::DeviceArray<float4>& interaction_cano_vertice, pcl::gpu::DeviceArray<float4>& interaction_cano_normal, pcl::gpu::DeviceArray<float4>& hand_joints_positions, pcl::gpu::DeviceArray<float>& hand_joints_radius,
			   pcl::gpu::DeviceArray<int3>& interaction_sphere_block, pcl::gpu::DeviceArray<float3>& interaction_sphere_coordinate, pcl::gpu::DeviceArray<unsigned char>& interaction_vertex_block_idx, bool& stop_itr);

private:
	int width, height;
	pcl::device::Intr camera_intrinsic_c0;
	pcl::device::Intr camera_intrinsic_c1;
	Eigen::Matrix4f view_camera0 = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f view_camera1;
	Eigen::Matrix4f object_pose;

	cudaTextureObject_t can_vmap_depth_view_c0;
	cudaTextureObject_t can_nmap_depth_view_c0;
	cudaTextureObject_t can_vmap_depth_view_c1;
	cudaTextureObject_t can_nmap_depth_view_c1;

	cudaTextureObject_t depth_vmap_c0;
	cudaTextureObject_t depth_nmap_c0;
	cudaTextureObject_t depth_vmap_c1;
	cudaTextureObject_t depth_nmap_c1;

	//	const Camera &camera;
	WarpField &warp_field;
	///	const pcl::gpu::DeviceArray<float> &L;

	float smooth_coeffs = 15.0f;
	float shading_coeffs = 0.0001f;
	float interaction_coeffs = 1.0f;//0.25f

	/*max Guass-Newton iteration*/
	int max_gn_iter = 0;

	/*debug values*/
	int icp_iter;
	int frame_idx;
	int outer_iter;

	/*results directory*/
	std::string m_results_dir;

	// cublas handle for pcg solver
	cublasHandle_t m_cublas_handle;

	//////////////////////////////////////////////////////////////////////////
	// preallocated buffers used in solver

	// used in associate_depth_data_pairs_dev
	pcl::gpu::DeviceArray<int> &m_depth_pairs_occupied_array;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_pair_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_array;
	pcl::gpu::DeviceArray<int> &m_depth_pairs_scan_storage;
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_compact_array;

	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_cv_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_cn_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_dv_f4;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_compact_array_dn_f4;

	// used in query_depth_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> &m_depth_pairs_knn_array;
	pcl::gpu::DeviceArray<float4> &m_depth_pairs_weight_array;

	// used in interaction_depth_data_knn_and_weight
	pcl::gpu::DeviceArray<int4> &m_interaction_knn_array;
	pcl::gpu::DeviceArray<float4> &m_interaction_weight_array;

	// used in evaluate_energy_device
	pcl::gpu::DeviceArray<float> &m_residual_array;
	pcl::gpu::DeviceArray<float> &m_evaluate_energy_reduce_storage;
	pcl::gpu::DeviceArray<float> &m_total_energy;

	// used in calculate_Ii_and_Iij
	pcl::gpu::DeviceArray<int> &m_Ii_key;
	pcl::gpu::DeviceArray<int> &m_Ii_value;

	pcl::gpu::DeviceArray<int> &m_Ii_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Ii_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Ii_radixsort_storage;
	pcl::gpu::DeviceArray<int> &m_Ii_offset;

	pcl::gpu::DeviceArray<int> &m_Iij_key;
	pcl::gpu::DeviceArray<int> &m_Iij_value;

	pcl::gpu::DeviceArray<int> &m_Iij_sorted_key;
	pcl::gpu::DeviceArray<int> &m_Iij_sorted_value;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_radixsort_storage;

	pcl::gpu::DeviceArray<int> &m_Iij_segment_label;
	pcl::gpu::DeviceArray<int> &m_Iij_scan;
	pcl::gpu::DeviceArray<unsigned char> &m_Iij_scan_storage;

	pcl::gpu::DeviceArray<int> &m_compact_Iij_key;
	pcl::gpu::DeviceArray<int> &m_compact_Iij_offset;

	pcl::gpu::DeviceArray<int> &m_row_offset;
	pcl::gpu::DeviceArray<int> &m_row_length;
	pcl::gpu::DeviceArray<int> &m_bin_length;

	// used in construct_ata_atb
	pcl::gpu::DeviceArray<float> &m_depth_term_values;
	pcl::gpu::DeviceArray<float> &m_smooth_term_values;
	pcl::gpu::DeviceArray<float> &m_interaction_term_values;
	///	pcl::gpu::DeviceArray<float> &m_shading_term_values;
	pcl::gpu::DeviceArray<float> &m_b_values;
	pcl::gpu::DeviceArray<float> &m_Bii_values;
	pcl::gpu::DeviceArray<float> &m_Bij_values;
	pcl::gpu::DeviceArray<int> &m_nonzero_rowscan;
	pcl::gpu::DeviceArray<float> &m_ATA_data;
	pcl::gpu::DeviceArray<int> &m_ATA_colidx;
	pcl::gpu::DeviceArray<int> &m_ATA_rowptr;
	pcl::gpu::DeviceArray<float> &m_ATb_data;

	// used in pcl_solver
	pcl::gpu::DeviceArray<float> &m_x_pcg;
	pcl::gpu::DeviceArray<float> &m_M_inv;
	pcl::gpu::DeviceArray<float> &m_p;
	pcl::gpu::DeviceArray<float> &m_q;
	pcl::gpu::DeviceArray<float> &m_r;
	pcl::gpu::DeviceArray<float> &m_s;
	pcl::gpu::DeviceArray<float> &m_t;

	// cudaStreams for associate_depth_data_pairs and find_valid_albedo_pixels
	cudaStream_t m_depth_data_pairs_stream;
	///	cudaStream_t m_valid_albedo_pixels_stream;

	// page-locked memory for the numbers of depth_data_pairs and valid_albedo_pairs
	int *m_depth_data_pairs_num;
	///	int *m_valid_albedo_pixels_num;

	// pre-allocated huber weight buffer for smooth terms
	pcl::gpu::DeviceArray<float> &m_huber_buffer;
};


#endif

