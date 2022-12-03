// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#ifndef __METALC__
#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#else
#include <assert.h>
#endif // NDEBUG
#endif

/// Kinect2 support is disabled by default (to not add the Kinect2 SDK dependency)
#ifndef COMPILE_WITHOUT_Kinect2
#define COMPILE_WITHOUT_Kinect2
#endif

#ifndef COMPILE_WITHOUT_CUDA
#include <cuda_runtime.h>

#ifndef ITMSafeCall
#define ITMSafeCall ORcudaSafeCall
#endif

#endif

#include "PlatformIndependence.h"
#include "ITMMath.h"

//////////////////////////////////////////////////////////////////////////
// Voxel Hashing definition and helper functions
//////////////////////////////////////////////////////////////////////////

#define SDF_BLOCK_SIZE 8				// SDF block size
#define SDF_BLOCK_SIZE3 512				// SDF_BLOCK_SIZE3 = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE
#define SDF_LOCAL_BLOCK_NUM 0x20000		// Number of locally stored blocks, currently 2^17   //ZH 0x40000

#define SDF_GLOBAL_BLOCK_NUM 0x120000	// Number of globally stored blocks: SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE    //ZH 0x120000
#define SDF_TRANSFER_BLOCK_NUM 0x1000	// Maximum number of blocks transfered in one swap operation                  //ZH 0x1000

#define SDF_BUCKET_NUM 0x100000			// Number of Hash Bucket, should be 2^n and bigger than SDF_LOCAL_BLOCK_NUM, SDF_HASH_MASK = SDF_BUCKET_NUM - 1  //ZH 0x100000
#define SDF_HASH_MASK 0xfffff			// Used for get hashing value of the bucket index,  SDF_HASH_MASK = SDF_BUCKET_NUM - 1     //ZH 0xfffff
#define SDF_EXCESS_LIST_SIZE 0x20000	// 0x20000 Size of excess list, used to handle collisions. Also max offset (unsigned short) value.           //ZH 0x20000

#define VOXEL_SIZE 2.0f//mm
#define rectify_constant 0.5f//mm

#define POINTS_PER_TRIANGLE 3
#define MAX_TRIANGLE_NUM 4 * 1000 * 1000
#define DEFAULT_TRIANGLES_BUFFER_SIZE  MAX_TRIANGLE_NUM*POINTS_PER_TRIANGLE

//////////////////////////////////////////////////////////////////////////
// Voxel Hashing data structures
//////////////////////////////////////////////////////////////////////////

/** \brief
A single entry in the hash table.
*/
struct ITMHashEntry
{
	/** Position of the corner of the 8x8x8 volume, that identifies the entry. */
	ORUVector3s pos;
	/** Offset in the excess list. */
	int offset;
	/** Pointer to the voxel block array.
	- >= 0 identifies an actual allocated entry in the voxel block array
	- -1 identifies an entry that has been removed (swapped out)
	- <-1 identifies an unallocated block
	*/
	int ptr;
};

struct ITMHashSwapState
{
	/// 0 - most recent data is on host, data not currently in active
	///     memory
	/// 1 - data both on host and in active memory, information has not
	///     yet been combined
	/// 2 - most recent data is in active memory, should save this data
	///     back to host at some point
	ORUuchar state;
};

#include "ITMVoxelBlockHash.h"

/** \brief
Stores the information of a single voxel in the volume
*/
struct ITMVoxel_f_rgb
{
	_CPU_AND_GPU_CODE_ static float SDF_initialValue() { return 1.0f; }
	_CPU_AND_GPU_CODE_ static float SDF_valueToFloat(float x) { return x; }
	_CPU_AND_GPU_CODE_ static float SDF_floatToValue(float x) { return x; }

	static const CONSTPTR(bool) hasColorInformation = true;

	/** Value of the truncated signed distance transformation. */
	float sdf;
	/** Number of fused observations that make up @p sdf. */
	ORUuchar w_depth;
	/** RGB colour information stored for this voxel. */
	ORUVector3u clr;
	/** Number of observations that made up @p clr. */
	ORUuchar w_color;

	_CPU_AND_GPU_CODE_ ITMVoxel_f_rgb()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
		clr = (ORUuchar)0;
		w_color = 0;
	}
};

/** \brief
Stores the information of a single voxel in the volume
*/
struct ITMVoxel_s_rgb
{
	_CPU_AND_GPU_CODE_ static short SDF_initialValue() { return 32767; }
	_CPU_AND_GPU_CODE_ static float SDF_valueToFloat(float x) { return (float)(x) / 32767.0f; }
	_CPU_AND_GPU_CODE_ static short SDF_floatToValue(float x) { return (short)((x)* 32767.0f); }

	static const CONSTPTR(bool) hasColorInformation = true;

	/** Value of the truncated signed distance transformation. */
	short sdf;
	/** Number of fused observations that make up @p sdf. */
	ORUuchar w_depth;
	/** Padding that may or may not improve performance on certain GPUs */
	//uchar pad;
	/** RGB colour information stored for this voxel. */
	ORUVector3u clr;
	/** Number of observations that made up @p clr. */
	ORUuchar w_color;

	_CPU_AND_GPU_CODE_ ITMVoxel_s_rgb()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
		clr = (ORUuchar)0;
		w_color = 0;
	}
};

struct ITMVoxel_s
{
	_CPU_AND_GPU_CODE_ static short SDF_initialValue() { return 32767; }
	_CPU_AND_GPU_CODE_ static float SDF_valueToFloat(float x) { return (float)(x) / 32767.0f; }
	_CPU_AND_GPU_CODE_ static short SDF_floatToValue(float x) { return (short)((x)* 32767.0f); }

	static const CONSTPTR(bool) hasColorInformation = false;

	/** Value of the truncated signed distance transformation. */
	short sdf;

//	short d_knn_index_ZH;
	/** Number of fused observations that make up @p sdf. */
	ORUuchar w_depth;
	/** Padding that may or may not improve performance on certain GPUs */
	//uchar pad;

	/*ZH the node index and its weight*/
//	ushort4 d_knn_index_ZH;
//	float4 d_weight_sparse;


	_CPU_AND_GPU_CODE_ ITMVoxel_s()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
//		d_knn_index_ZH.x = 0;
//		d_knn_index_ZH.y = 0;
//		d_knn_index_ZH.z = 0;
//		d_knn_index_ZH.w = 0;
//		d_knn_index_ZH = 0;

	}
};

struct ITMVoxel_f
{
	_CPU_AND_GPU_CODE_ static float SDF_initialValue() { return 1.0f; }
	_CPU_AND_GPU_CODE_ static float SDF_valueToFloat(float x) { return x; }
	_CPU_AND_GPU_CODE_ static float SDF_floatToValue(float x) { return x; }

	static const CONSTPTR(bool) hasColorInformation = false;

	/** Value of the truncated signed distance transformation. */
	float sdf;
	/** Number of fused observations that make up @p sdf. */
	ORUuchar w_depth;
	/** Padding that may or may not improve performance on certain GPUs */
	//uchar pad;

	_CPU_AND_GPU_CODE_ ITMVoxel_f()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
	}
};

/** This chooses the information stored at each voxel. At the moment, valid
options are ITMVoxel_s, ITMVoxel_f, ITMVoxel_s_rgb and ITMVoxel_f_rgb
*/
//ZH this is important. We can change this to store color information
typedef ITMVoxel_s ITMVoxel;

/** This chooses the way the voxels are addressed and indexed. At the moment,
valid options are ITMVoxelBlockHash and ITMPlainVoxelArray.
*/
typedef ITMLib::Objects::ITMVoxelBlockHash ITMVoxelIndex;
//typedef ITMLib::Objects::ITMPlainVoxelArray ITMVoxelIndex;


#if 0

#include "../../ORUtils/Image.h"

//////////////////////////////////////////////////////////////////////////
// Do not change below this point
//////////////////////////////////////////////////////////////////////////
#ifndef ITMFloatImage
#define ITMFloatImage ORUtils::Image<float>
#endif

#ifndef ITMFloat2Image
#define ITMFloat2Image ORUtils::Image<Vector2f>
#endif

#ifndef ITMFloat4Image
#define ITMFloat4Image ORUtils::Image<Vector4f>
#endif

#ifndef ITMShortImage
#define ITMShortImage ORUtils::Image<short>
#endif

#ifndef ITMShort3Image
#define ITMShort3Image ORUtils::Image<Vector3s>
#endif

#ifndef ITMShort4Image
#define ITMShort4Image ORUtils::Image<Vector4s>
#endif

#ifndef ITMUShortImage
#define ITMUShortImage ORUtils::Image<ushort>
#endif

#ifndef ITMUIntImage
#define ITMUIntImage ORUtils::Image<uint>
#endif

#ifndef ITMIntImage
#define ITMIntImage ORUtils::Image<int>
#endif

#ifndef ITMUCharImage
#define ITMUCharImage ORUtils::Image<uchar>
#endif

#ifndef ITMUChar4Image
#define ITMUChar4Image ORUtils::Image<Vector4u>
#endif

#ifndef ITMBoolImage
#define ITMBoolImage ORUtils::Image<bool>
#endif


#endif

//debug
#ifndef DEBUGBREAK
#define DEBUGBREAK \
{ \
	int ryifrklaeybfcklarybckyar=0; \
	ryifrklaeybfcklarybckyar++; \
}
#endif

#ifndef TRACKER_ITERATION_TYPE
#define TRACKER_ITERATION_TYPE
/// The tracker iteration type used to define the tracking iteration regime
typedef enum
{
	TRACKER_ITERATION_ROTATION = 1,
	TRACKER_ITERATION_TRANSLATION = 2,
	TRACKER_ITERATION_BOTH = 3,
	TRACKER_ITERATION_NONE = 4
} TrackerIterationType;
#endif
