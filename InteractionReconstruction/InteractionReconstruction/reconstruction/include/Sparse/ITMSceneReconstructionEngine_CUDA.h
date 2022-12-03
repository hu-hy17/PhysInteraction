// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <math.h>

#include "SparseVoxel.h"

#include "ITMScene.h"
#include "../pcl/internal.h"
#include "ITMMath.h"
#include "ITMRenderState_VH.h"
#include "ITMSceneReconstructionEngine.h"

using namespace ITMLib::Objects;
using namespace pcl::device;

		class ITMSceneReconstructionEngine_CUDA
		{
		private:
			void *allocationTempData_device;
			void *allocationTempData_host;
			unsigned char *entriesAllocType_device;
			ORUVector4s *blockCoords_device;

		public:
			void ResetScene(ITMScene *scene);

			void AllocateSceneFromDepth(ITMScene *scene, pcl::gpu::DeviceArray2D<unsigned short> _depth_frame, int width, int height, const Intr &_depth_intr, const ORUMatrix4f DepthPose, const ITMRenderState_VH *renderState, bool onlyUpdateVisibleList = false);

			void IntegrateIntoScene(ITMScene *scene, pcl::gpu::DeviceArray2D<unsigned short> _depth_frame, int width, int height, const Intr &_depth_intr, const ORUMatrix4f DepthPose,
				const ITMRenderState_VH *renderState);

			ITMSceneReconstructionEngine_CUDA(void);
			~ITMSceneReconstructionEngine_CUDA(void);
		};



