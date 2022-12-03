/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef PCL_KINFU_INTERNAL_HPP_
#define PCL_KINFU_INTERNAL_HPP_

#include <pcl/gpu/containers/device_array.h>
//#include <pcl/gpu/utils/safe_call.hpp>
#include "safe_call.hpp"
#include "../gpu/vector_operations.hpp"
//#include "../image_texture.hpp"
#include <cuda.h>
#include "../gpu/dual_quaternion.hpp"
#include "../gpu/constants.h"
#include "../Sparse/ITMScene.h"

namespace pcl
{
  namespace device
  {
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Types
    typedef unsigned short PCLushort;
    typedef DeviceArray2D<float> MapArr;
    typedef DeviceArray2D<PCLushort> DepthMap;
    typedef float4 PointType;

    //TSDF fixed point divisor (if old format is enabled)
    const int DIVISOR = 32767;     // SHRT_MAX;

	//Should be multiple of 32
    enum { VOLUME_X = RESOLUTION_X, VOLUME_Y = RESOLUTION_Y, VOLUME_Z = RESOLUTION_Z };
	//enum{ VOLUME_X = 256, VOLUME_Y = 256, VOLUME_Z = 256 };
	
	/*max weight of tsdf*/
	enum { TSDF_MAX_WEIGHT = 1 << 7 };

	/*max weight of albedo*/
	enum { ALBEDO_MAX_WEIGHT = 1 << 7 };

    const float VOLUME_SIZE = 1.0f; // in meters

    /** \brief Camera intrinsics structure
      */ 
    struct Intr
    {
      float fx, fy, cx, cy;
      Intr () {}
      Intr (float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

      Intr operator()(int level_index) const
      { 
        int div = 1 << level_index; 
        return (Intr (fx / div, fy / div, cx / div, cy / div));
      }
    };

    /** \brief 3x3 Matrix for device code
      */ 
    struct Mat33
    {
      float3 data[3];
    };

    /** \brief Light source collection
      */ 
    struct LightSource
    {
      float3 pos[1];
      int number;
    };

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TSDF volume functions            

    /** \brief Perform tsdf volume initialization
      *  \param[out] array volume to be initialized
      */
    void
    initVolume(PtrStep<short2> array);

	/*initialize both sdf volume and albedo volume*/
	void
	initVolume(PtrStep<short2> sdf_volume, PtrStep<float4> albedo_volume);

	/*initialize only albedo volume*/
	void init_color_volume(PtrStep<float4> albedo_volume);

    //first version
    /** \brief Performs Tsfg volume uptation (extra obsolete now)
      * \param[in] depth_raw Kinect depth image
      * \param[in] intr camera intrinsics
      * \param[in] volume_size size of volume in mm
      * \param[in] Rcurr_inv inverse rotation for current camera pose
      * \param[in] tcurr translation for current camera pose
      * \param[in] tranc_dist tsdf truncation distance
      * \param[in] volume tsdf volume to be updated
      */

	/*integrate tsdf only for the first frame (without any transformation)*/
	void integrate_first_depth(//intrusive_ptr<ImageTexture> _depth_texture_ptr,
							   pcl::gpu::DeviceArray2D<unsigned short> _depth_frame,
							   const Intr &_depth_intr,
							   const mat33 &_w2d_r, const float3 &_w2d_t,
							   PtrStep<short2> _tsdf_volume,
							   float _trunc_dist, float3 _vol_size);

	/*update tsdf volume for the first time, update voxel sdf and color simultaneously*/
	void
	integrateTsdfVolume(const PtrStepSz<PCLushort> &depth_frame, const PtrStepSz<uchar3> &color_frame,
						const Intr &depth_intr, const Intr &color_intr, const mat33 &w2d_r, const float3 &w2d_t,
						const mat33 &d2c_r, const float3 &d2c_t, float tranc_dist, PtrStep<short2> tsdf_volume,
						PtrStep<uchar4> color_volume, float3 volume_size);

	void
	integrateTsdfVolume_nonrigid(const PtrStepSz<PCLushort>& depth_raw, const Intr& intr, const float3& volume_size,
								 const mat33& w2c_r, const float3& w2c_t, float tranc_dist, PtrStep<short2> volume,
								 const PtrSz<int> &_voxel_knn_index, const PtrSz<float> &_voxel_knn_dist,
								 const PtrSz<mat34> &_node_se3);

	void
	integrateTsdfVolume_nonrigid_weighted(const PtrStepSz<PCLushort>& depth_raw,
										  const Intr& intr, const float3& volume_size,
										  const mat33& w2c_r, const float3& w2c_t, float tranc_dist,
										  PtrStep<short2> volume,
										  const PtrSz<ushort4> &_voxel_knn_index,
										  const PtrSz<float3> &_node_coords,
										  const PtrSz<mat34> &_node_se3,
										  const PtrStep<float3> &_depth_nmap);

	/*integrate tsdf*/
	pcl::gpu::DeviceArray<int>
	integrate_tsdf(cudaTextureObject_t _depth_frame,
				   cudaTextureObject_t _depth_nmap,
				   int _depth_width, int _depth_height,
				   const Intr &_depth_intr,
				   const mat33 &_w2d_r, const float3 &_w2d_t,
				   const mat33 &_d2c_r, const float3 &_d2c_t,
				   PtrStep<short2> _tsdf_volume,
				   int3 _vol_res, float3 _vol_size,
				   float _trunc_dist, float _shell_dist,
				   const PtrSz<ushort4> &_knn_field,
				   const PtrSz<float3> &_node_coords,
				   const PtrSz<mat34> &_node_se3,
				   pcl::gpu::DeviceArray<int> &_voxel_candidates_buffer);

	// merged integrate_tsdf version (with ytrock)
#if 0
	pcl::gpu::DeviceArray<int>
	integrate_tsdf_merged_ytrock(cudaTextureObject_t _depth_frame,
								 int _depth_width, int _depth_height,
								 const Intr &_depth_intr,
								 const mat33 &_w2d_r, const float3 &_w2d_t,
								 const mat33 &_d2c_r, const float3 &_d2c_t,
								 pcl::gpu::DeviceArray2D<int> _tsdf_volume,
								 int3 _vol_res, float3 _vol_size,
								 float _trunc_dist, float _shell_dist,
								 const pcl::gpu::DeviceArray<ushort4> &_knn_field,
								 //const pcl::gpu::DeviceArray<float3> &_node_coords,
								 //const pcl::gpu::DeviceArray<mat34> &_node_se3,
								 const pcl::gpu::DeviceArray<float> _node_coords,
								 const pcl::gpu::DeviceArray<float> _node_se3,
								 pcl::gpu::DeviceArray<int> &_voxel_candidates_buffer);
#else
//	pcl::gpu::DeviceArray<int>
	void
	integrate_tsdf_merged_ytrock(cudaTextureObject_t _depth_frame,
								 int _depth_width, int _depth_height,
								 const Intr &_depth_intr,
								 const mat33 &_w2d_r, const float3 &_w2d_t,
///								 const mat33 &_d2c_r, const float3 &_d2c_t,
								 ITMLib::Objects::ITMScene *scene,//pcl::gpu::DeviceArray2D<int> _tsdf_volume,
								 float3 _voxel_size,//int3 _vol_res, float3 _vol_size,
///								 float _trunc_dist, float _shell_dist,
								 const pcl::gpu::DeviceArray<ushort4> &_knn_field,
								 const pcl::gpu::DeviceArray<float4> _node_coords,
								 const pcl::gpu::DeviceArray<DualQuaternion> _node_se3,
								 const pcl::gpu::DeviceArray<bool> &_flag_volume,
								 const pcl::gpu::DeviceArray<float4> &_weight_volume,
								 const pcl::gpu::DeviceArray<short4> &_valid_voxelblock, const int weight_limit1, const int weight_limit2, mat34 rigidpart_DynamicObject);
								 /*pcl::gpu::DeviceArray<int> &_voxel_candidates_buffer*/

	void
		integrate_tsdf_merged_ytrock2(cudaTextureObject_t _depth_frame,
			int _depth_width, int _depth_height,
			const Intr &_depth_intr,
			const mat33 &_w2d_r, const float3 &_w2d_t,
			///								 const mat33 &_d2c_r, const float3 &_d2c_t,
			ITMLib::Objects::ITMScene *scene,//pcl::gpu::DeviceArray2D<int> _tsdf_volume,
			float3 _voxel_size,//int3 _vol_res, float3 _vol_size,
							   ///								 float _trunc_dist, float _shell_dist,
			const pcl::gpu::DeviceArray<ushort4> &_knn_field,
			const pcl::gpu::DeviceArray<float4> _node_coords,
			const pcl::gpu::DeviceArray<DualQuaternion> _node_se3,
			const pcl::gpu::DeviceArray<bool> &_flag_volume,
			const pcl::gpu::DeviceArray<float4> &_weight_volume,
			const pcl::gpu::DeviceArray<short4> &_valid_voxelblock, 
			const pcl::gpu::DeviceArray<int> &_validblock_idx,const int validblock_num,
			const int weight_limit1, const int weight_limit2, mat34 rigidpart_DynamicObject);
	/*pcl::gpu::DeviceArray<int> &_voxel_candidates_buffer*/


#endif

	void rigid_fusion(cudaTextureObject_t _depth_frame,
		int _depth_width, int _depth_height,
		const Intr &_depth_intr,
		const mat33 &_w2d_r, const float3 &_w2d_t,
		ITMLib::Objects::ITMScene *scene,
		float3 _voxel_size,
		const pcl::gpu::DeviceArray<short4> &_valid_voxelblock);

	void non_rigid_fusion(cudaTextureObject_t _depth_frame,
		int _depth_width, int _depth_height,
		const Intr &_depth_intr,
		const mat33 &_w2d_r, const float3 &_w2d_t,
		ITMLib::Objects::ITMScene *scene,
		float3 _voxel_size,
		const pcl::gpu::DeviceArray<ushort4> &_knn_field,
		const pcl::gpu::DeviceArray<float4> _node_coords,
		const pcl::gpu::DeviceArray<DualQuaternion> _node_se3,
		const pcl::gpu::DeviceArray<bool> &_flag_volume,
		const pcl::gpu::DeviceArray<float4> &_weight_volume,
		const pcl::gpu::DeviceArray<short4> &_valid_voxelblock);

	// merged integrate_tsdf version with ytrock using driver api
	pcl::gpu::DeviceArray<int>
	integrate_tsdf_merged_ytrock_driver_api(CUfunction _kernel,
											CUdeviceptr _max_voxel_num,
											CUdeviceptr _voxel_num,
											cudaTextureObject_t _depth_frame,
											int _depth_width, int _depth_height,
											const Intr &_depth_intr,
											const mat33 &_w2d_r, const float3 &_w2d_t,
											const mat33 &_d2c_r, const float3 &_d2c_t,
											pcl::gpu::DeviceArray2D<int> _tsdf_volume,
											int3 _vol_res, float3 _vol_size,
											float _trunc_dist, float _shell_dist,
											const pcl::gpu::DeviceArray<ushort4> &_knn_field,
											const pcl::gpu::DeviceArray<float4> _node_coords,
											const pcl::gpu::DeviceArray<DualQuaternion> _node_se3,
											const pcl::gpu::DeviceArray<bool> &_flag_volume,
											const pcl::gpu::DeviceArray<float4> &_weight_volume,
											pcl::gpu::DeviceArray<int> &_voxel_candidates_buffer);

	/*update voxel albedo*/
	void update_albedo(const PtrStepSz<unsigned short> &_depth_frame,
					   const PtrStepSz<uchar3> &_color_frame,
					   const Intr &_depth_intr,
					   const Intr &_color_intr,
					   const mat33 &_w2d_r, const float3 &_w2d_t,
					   const mat33 &_d2c_r, const float3 &_d2c_t,
					   const PtrStep<short2> _tsdf_volume,
					   PtrStep<float4> _albedo_volume,
					   const int3 &_vol_res, const float3 &_vol_size,
					   float _shell_dist, float _trunc_dist,
					   const PtrSz<float3> &_node_coords,
					   const PtrSz<mat34> &_node_se3,
					   const PtrSz<ushort4> &_knn_field,
					   const std::vector<float> &_light_coeffs);

	/*integrate voxel albedo*/
	void integrate_albedo(const PtrStepSz<unsigned short> &_depth_frame,
						  const PtrStepSz<uchar3> &_color_frame,
						  const Intr &_depth_intr,
						  const Intr &_color_intr,
						  const mat33 &_w2d_r, const float3 &_w2d_t,
						  const mat33 &_d2c_r, const float3 &_d2c_t,
						  const PtrStep<short2> _tsdf_volume,
						  PtrStep<float4> _albedo_volume,
						  const int3 &_vol_res, const float3 &_vol_size,
						  float _shell_dist, float _trunc_dist,
						  const PtrSz<float3> &_node_coords,
						  const PtrSz<mat34> &_node_se3,
						  const PtrSz<ushort4> &_knn_field,
						  const std::vector<float> &_light_coeffs,
						  PtrStep<float4> _albedo_volume_copy);

	// integrate tsdf (rigid motion, typically for the first frame) and
	// find all candidates (voxel normal array, voxel color array, and voxel index array)
	void integrate_tsdf_and_find_candidates(const PtrStepSz<PCLushort> &_depth_frame,
											const PtrStepSz<uchar3> &_color_frame,
											const Intr &_depth_intr, const Intr &_color_intr,
											const mat33 &_w2d_r, const float3 &_w2d_t,
											const mat33 &_d2c_r, const float3 &_d2c_t,
											PtrStep<short2> _tsdf_volume,
											float3 _vol_size, float _trunc_dist, float _shell_dist,
											pcl::gpu::DeviceArray<float3> &_voxel_coords,
											pcl::gpu::DeviceArray<float3> &_voxel_normals,
											pcl::gpu::DeviceArray<uchar3> &_voxel_colors,
											pcl::gpu::DeviceArray<int> &_voxel_indices,
											int &_candidate_num);

	/*tsdf update and voxel color update (nonrigid motion)*/
	void
	integrateTsdfVolume_color(const PtrStepSz<PCLushort>& depth_raw, const Intr& intr, const float3& volume_size,
							  const mat33& w2c_r, const float3& w2c_t, float tranc_dist, PtrStep<short2> volume,
							  const PtrSz<ushort4> &_voxel_knn_index, const PtrSz<float3> &_node_coords,
							  const PtrSz<mat34> &_node_se3, const PtrStep<float3> &_depth_nmap,
							  const Intr color_intr, const mat33 d2c_r, const float3 d2c_t,
							  PtrStep<uchar4> color_volume, const PtrStepSz<uchar4> &_color_map, int _max_weight);

	/*tsdf update, and update voxel color using only current frame*/
	void integrate_tsdf_refresh_color(const PtrStepSz<PCLushort>& depth_raw, const Intr& intr, const float3& volume_size,
									  const mat33& w2c_r, const float3& w2c_t, float tranc_dist, PtrStep<short2> volume,
									  const PtrSz<ushort4> &_voxel_knn_index, const PtrSz<float3> &_node_coords,
									  const PtrSz<mat34> &_node_se3, const PtrStep<float3> &_depth_nmap,
									  const Intr color_intr, const mat33 d2c_r, const float3 d2c_t,
									  PtrStep<uchar4> color_volume, const PtrStepSz<uchar3> &_color_map);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Raycast and view generation        
    /** \brief Generation vertex and normal maps from volume for current camera pose
      * \param[in] intr camera intrinsices
      * \param[in] Rcurr current rotation
      * \param[in] tcurr current translation
      * \param[in] tranc_dist volume truncation distance
      * \param[in] volume_size volume size in mm
      * \param[in] volume tsdf volume
      * \param[out] vmap output vertex map
      * \param[out] nmap output normals map
      */
    void 
    raycast (const Intr& intr, const Mat33& Rcurr, const float3& tcurr, float tranc_dist, const float3& volume_size, 
             const PtrStep<short2>& volume, MapArr& vmap, MapArr& nmap);

    /** \brief Renders 3D image of the scene
      * \param[in] vmap vetex map
      * \param[in] nmap normals map
      * \param[in] light poase of light source
      * \param[out] dst buffer where image is generated
      */
    void 
    generateImage (const MapArr& vmap, const MapArr& nmap, const LightSource& light, PtrStepSz<uchar3> dst);


    /** \brief Renders depth image from give pose
      * \param[in] R_inv inverse camera rotation
      * \param[in] t camera translation
      * \param[in] vmap vertex map
      * \param[out] dst buffer where depth is generated
      */
    void
    generateDepth (const Mat33& R_inv, const float3& t, const MapArr& vmap, DepthMap& dst);

     /** \brief Paints 3D view with color map
      * \param[in] colors rgb color frame from OpenNI   
      * \param[out] dst output 3D view
      * \param[in] colors_weight weight for colors   
      */
    void 
    paint3DView(const PtrStep<uchar3>& colors, PtrStepSz<uchar3> dst, float colors_weight = 0.5f);

    /** \brief Performs resize of vertex map to next pyramid level by averaging each four points
      * \param[in] input vertext map
      * \param[out] output resized vertex map
      */
    void 
    resizeVMap (const MapArr& input, MapArr& output);
    
    /** \brief Performs resize of vertex map to next pyramid level by averaging each four normals
      * \param[in] input normal map
      * \param[out] output vertex map
      */
    void 
    resizeNMap (const MapArr& input, MapArr& output);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Cloud extraction 

    /** \brief Perform point cloud extraction from tsdf volume
      * \param[in] volume tsdf volume 
      * \param[in] volume_size size of the volume
      * \param[out] output buffer large enought to store point cloud
      * \return number of point stored to passed buffer
      */ 
    PCL_EXPORTS size_t 
    extractCloud (const PtrStep<short2>& volume, const float3& volume_size, PtrSz<PointType> output);

    /** \brief Performs normals computation for given poins using tsdf volume
      * \param[in] volume tsdf volume
      * \param[in] volume_size volume size
      * \param[in] input points where normals are computed
      * \param[out] output normals. Could be float4 or float8. If for a point normal can't be computed, such normal is marked as nan.
      */ 
    template<typename NormalType> 
    void 
    extractNormals (const PtrStep<short2>& volume, const float3& volume_size, const PtrSz<PointType>& input, NormalType* output);

	/*average vertex normal within a small window*/
	pcl::gpu::DeviceArray<float4>
		extract_smooth_normals(const PtrStep<short2> &_tsdf_volume,
						   const int3 &_vol_res, const float3 &_vol_size,
						   const pcl::gpu::DeviceArray<float4> &_valid_can_vertices,
						   pcl::gpu::DeviceArray<float4> &_smooth_normals_buffer);

	/*extract voxel normal using central differentiation*/
	pcl::gpu::DeviceArray<float4>
	extract_normals(const PtrStep<short2> &_tsdf_volume,
					const int3 &_vol_res, const float3 &_vol_size,
					const pcl::gpu::DeviceArray<float4> &_valid_can_vertices,
					pcl::gpu::DeviceArray<float4> &_can_normals_buffer);

    /** \brief Performs colors exctraction from color volume
      * \param[in] color_volume color volume
      * \param[in] volume_size volume size
      * \param[in] points points for which color are computed
      * \param[out] colors output array with colors.
      */
    void 
    exctractColors(const PtrStep<uchar4>& color_volume, const float3& volume_size, const PtrSz<PointType>& points, uchar4* colors);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    struct float8  { float x, y, z, w, c1, c2, c3, c4; };
    struct float12 { float x, y, z, w, normal_x, normal_y, normal_z, n4, c1, c2, c3, c4; };

    /** \brief Conversion from SOA to AOS
      * \param[in] vmap SOA map
      * \param[out] output Array of 3D points. Can be float4 or float8.
      */
    template<typename T> 
    void 
    convert (const MapArr& vmap, DeviceArray2D<T>& output);

    /** \brief Merges pcl::PointXYZ and pcl::Normal to PointNormal
      * \param[in] cloud points cloud
      * \param[in] normals normals cloud
      * \param[out] output array of PointNomals.
      */
    void 
    mergePointNormal(const DeviceArray<float4>& cloud, const DeviceArray<float8>& normals, const DeviceArray<float12>& output);

    /** \brief  Check for qnan (unused now) 
      * \param[in] value
      */
    inline bool 
    valid_host (float value)
    {
      return *reinterpret_cast<int*>(&value) != 0x7fffffff; //QNAN
    }

    /** \brief synchronizes CUDA execution in null stream */
    inline 
    void 
    sync () { cudaSafeCall (cudaStreamSynchronize(0)); }


    template<class D, class Matx> D&
    device_cast (Matx& matx)
    {
      return (*reinterpret_cast<D*>(matx.data ()));
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Marching cubes implementation

    /** \brief Binds marching cubes tables to texture references */
    void 
    bindTextures(const int *edgeBuf, const int *triBuf, const int *numVertsBuf);            
    
    /** \brief Unbinds */
    void 
    unbindTextures();
    
    /** \brief Scans tsdf volume and retrieves occuped voxes
      * \param[in] volume tsdf volume
      * \param[out] occupied_voxels buffer for occuped voxels. The function fulfills first row with voxel ids and second row with number of vertextes.
      * \return number of voxels in the buffer
      */
    int
    getOccupiedVoxels(const PtrStep<short2> &tsdf_volume,
					  DeviceArray2D<int> &occupied_voxels);

    /** \brief Computes total number of vertexes for all voxels and offsets of vertexes in final triangle array
      * \param[out] occupied_voxels buffer with occuped voxels. The function fulfills 3nd only with offsets      
      * \return total number of vertexes
      */
    int
    computeOffsetsAndTotalVertexes(DeviceArray2D<int>& occupied_voxels);

    /** \brief Generates final triangle array
      * \param[in] volume tsdf volume
      * \param[in] occupied_voxels occuped voxel ids (first row), number of vertexes(second row), offsets(third row).
      * \param[in] volume_size volume size in meters
      * \param[out] output triangle array            
      */
    void
    generateTriangles(const PtrStep<short2>& volume, const DeviceArray2D<int>& occupied_voxels, const float3& volume_size, DeviceArray<PointType>& output);
  }
}

#endif /* PCL_KINFU_INTERNAL_HPP_ */
