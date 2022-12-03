// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdlib.h>

#include "MemoryBlock.h"
#include "ITMMath.h"
#include "SparseVoxel.h"


		/** \brief
		    Stores the render state used by the SceneReconstruction 
			and visualisation engines, as used by voxel hashing.
		*/
		class ITMRenderState_VH
		{
		private:
			MemoryDeviceType memoryType;

			/** A list of "visible entries", that are currently
			being processed by the tracker.
			*/
			ORUtils::MemoryBlock<int> *visibleEntryIDs;

			/** A list of "visible entries", that are
			currently being processed by integration
			and tracker.
			*/
			ORUtils::MemoryBlock<ORUuchar> *entriesVisibleType;
            
		public:
			/** Number of entries in the live list. */
			int noVisibleEntries;
            
			//ITMRenderState_VH(int noTotalEntries, const Vector2i & imgSize, float vf_min, float vf_max, MemoryDeviceType memoryType = MEMORYDEVICE_CPU)
			ITMRenderState_VH(int noTotalEntries, MemoryDeviceType memoryType = MEMORYDEVICE_CUDA)
            {
				this->memoryType = memoryType;

				visibleEntryIDs = new ORUtils::MemoryBlock<int>(SDF_LOCAL_BLOCK_NUM, memoryType);
				entriesVisibleType = new ORUtils::MemoryBlock<ORUuchar>(noTotalEntries, memoryType);
				
				noVisibleEntries = 0;
            }
            
			~ITMRenderState_VH()
            {
				delete visibleEntryIDs;
				delete entriesVisibleType;
            }

			/** Get the list of "visible entries", that are currently
			processed by the tracker.
			*/
			const int *GetVisibleEntryIDs(void) const { return visibleEntryIDs->GetData(memoryType); }
			int *GetVisibleEntryIDs(void) { return visibleEntryIDs->GetData(memoryType); }

			/** Get the list of "visible entries", that are
			currently processed by integration and tracker.
			*/
			ORUuchar *GetEntriesVisibleType(void) { return entriesVisibleType->GetData(memoryType); }

#ifdef COMPILE_WITH_METAL
			const void* GetVisibleEntryIDs_MB(void) { return visibleEntryIDs->GetMetalBuffer(); }
			const void* GetEntriesVisibleType_MB(void) { return entriesVisibleType->GetMetalBuffer(); }
#endif
		};

