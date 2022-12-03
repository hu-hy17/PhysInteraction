// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once
#include "ITMScene.h"
#include <pcl/gpu/containers/device_array.h>
#include "ITMMeshingEngine.h"
//#include "pcl/safe_call.hpp"
#include "../gpu/vector_operations.hpp"


namespace ITMLib
{
	namespace Engine
	{

		class ITMMeshingEngine_CUDA
		{
		private:
			unsigned int  *noTriangles_device;
			unsigned int *noVisibleBlock_device;
			ORUVector4s *visibleBlockGlobalPos_device;

		public:
//			pcl::gpu::DeviceArray<float4> MeshScene(pcl::gpu::DeviceArray<float4> &m_can_vertices_buffer, pcl::gpu::DeviceArray<float4> &m_warped_vertices_buffer, const ITMLib::Objects::ITMScene *scene);
			std::pair<pcl::gpu::DeviceArray<float4>, pcl::gpu::DeviceArray<float4>> MeshScene(pcl::gpu::DeviceArray<float4> &m_can_vertices_buffer, pcl::gpu::DeviceArray<float4> &m_live_vertices_buffer, const ITMLib::Objects::ITMScene *scene, mat34 object_pose, const int weight_thr);

			pcl::gpu::DeviceArray<float4> MeshScene_canonical_weight(pcl::gpu::DeviceArray<float4> &m_can_vertices_buffer, const ITMLib::Objects::ITMScene *scene, const int weight_thr/*, cudaStream_t m_object_stream*/);

			pcl::gpu::DeviceArray<float4> MeshScene_FirstFrame(pcl::gpu::DeviceArray<float4> &m_can_vertices_buffer, const ITMLib::Objects::ITMScene *scene);

//			pcl::gpu::DeviceArray<float4> MeshScene_RigidPart(pcl::gpu::DeviceArray<float4> &m_can_vertices_buffer_RigidPart, pcl::gpu::DeviceArray<float4> &m_warped_vertices_buffer, const ITMLib::Objects::ITMScene *scene);

			pcl::gpu::DeviceArray<float4> MeshScene_DevidePart(pcl::gpu::DeviceArray<float4> &m_can_vertices_buffer_DevidePart, const ITMLib::Objects::ITMScene *scene, const pcl::gpu::DeviceArray<short4> &d_valid_voxel_block, short VBw);

			pcl::gpu::DeviceArray<float4> NormalExtraction(pcl::gpu::DeviceArray<float4> &m_can_normals_buffer, pcl::gpu::DeviceArray<float4> &m_warped_normals_buffer, const pcl::gpu::DeviceArray<float4> &_valid_can_vertices, const ITMLib::Objects::ITMScene *scene);
			
			std::pair<pcl::gpu::DeviceArray<float4>, pcl::gpu::DeviceArray<float4>> NormalExtraction_DevidePart(pcl::gpu::DeviceArray<float4> &m_can_normals_buffer, pcl::gpu::DeviceArray<float4> &m_live_normals_buffer, const pcl::gpu::DeviceArray<float4> &_valid_can_vertices,  const ITMLib::Objects::ITMScene *scene, mat33 object_pose);
			
			pcl::gpu::DeviceArray<float4> NormalExtraction_canonical(pcl::gpu::DeviceArray<float4> &m_can_normals_buffer, const pcl::gpu::DeviceArray<float4> &_valid_can_vertices, const ITMLib::Objects::ITMScene *scene/*, cudaStream_t m_object_stream*/);
			
			ITMMeshingEngine_CUDA(void);
			~ITMMeshingEngine_CUDA(void);
		};
	}
}
