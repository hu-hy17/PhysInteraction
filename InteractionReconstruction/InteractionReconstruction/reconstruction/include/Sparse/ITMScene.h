// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "SparseVoxel.h"

#include "ITMSceneParams.h"
#include "ITMLocalVBA.h"
#include "ITMGlobalCache.h"
///#include "ITMGlobalCache.h"

namespace ITMLib
{
	namespace Objects
	{
		/** \brief
		Represents the 3D world model as a hash of small voxel
		blocks
		*/

		class ITMScene
		{
		public:
			bool useSwapping;

			/** Scene parameters like voxel size etc. */
			const ITMSceneParams *sceneParams;

			/** Hash table to reference the 8x8x8 blocks */
			ITMVoxelBlockHash index;

			/** Current local content of the 8x8x8 voxel blocks -- stored host or device */
			ITMLocalVBA<ITMVoxel> localVBA;

			/*ZH the d_knn_index */
//			pcl::gpu::DeviceArray<ushort4> d_knn_index;

			/** Global content of the 8x8x8 voxel blocks -- stored on host only */
			ITMGlobalCache<ITMVoxel> *globalCache;

			ITMScene(const ITMSceneParams *sceneParams, bool useSwapping, MemoryDeviceType memoryType)
				: index(memoryType), localVBA(memoryType, index.getNumAllocatedVoxelBlocks(), index.getVoxelBlockSize())
			{
				this->sceneParams = sceneParams;
				this->useSwapping = useSwapping;
				if (useSwapping) globalCache = new ITMGlobalCache<ITMVoxel>();
				

			}

			~ITMScene(void)
			{
///				if (useSwapping) delete globalCache;
			}

			// Suppress the default copy constructor and assignment operator
			ITMScene(const ITMScene&);
			ITMScene& operator=(const ITMScene&);
		};
	}
}