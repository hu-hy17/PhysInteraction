// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

//#include "../../Utils/ITMLibDefines.h"
//#include "ITMPixelUtils.h"

template<typename T> _CPU_AND_GPU_CODE_ inline int hashIndex(const THREADPTR(T) & blockPos) {
	return (((ORUuint)blockPos.x * 73856093u) ^ ((ORUuint)blockPos.y * 19349669u) ^ ((ORUuint)blockPos.z * 83492791u)) & (ORUuint)SDF_HASH_MASK;
}

_CPU_AND_GPU_CODE_ inline int pointToVoxelBlockPos(const THREADPTR(ORUVector3i) & point, THREADPTR(ORUVector3i) &blockPos) {
	blockPos.x = ((point.x < 0) ? point.x - SDF_BLOCK_SIZE + 1 : point.x) / SDF_BLOCK_SIZE;
	blockPos.y = ((point.y < 0) ? point.y - SDF_BLOCK_SIZE + 1 : point.y) / SDF_BLOCK_SIZE;
	blockPos.z = ((point.z < 0) ? point.z - SDF_BLOCK_SIZE + 1 : point.z) / SDF_BLOCK_SIZE;

	//Vector3i locPos = point - blockPos * SDF_BLOCK_SIZE;
	//return locPos.x + locPos.y * SDF_BLOCK_SIZE + locPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
	return point.x + (point.y - blockPos.x) * SDF_BLOCK_SIZE + (point.z - blockPos.y) * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE - blockPos.z * SDF_BLOCK_SIZE3;
}

_CPU_AND_GPU_CODE_ inline int findVoxel(const CONSTPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexData) *voxelIndex, const THREADPTR(ORUVector3i) & point,
	THREADPTR(bool) &isFound, THREADPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexCache) & cache)
{
	ORUVector3i blockPos;
	short linearIdx = pointToVoxelBlockPos(point, blockPos);

	if IS_EQUAL3(blockPos, cache.blockPos)
	{
		isFound = true;
		return cache.blockPtr + linearIdx;
	}

	int hashIdx = hashIndex(blockPos);

	while (true) 
	{
		ITMHashEntry hashEntry = voxelIndex[hashIdx];

		if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= 0)
		{
			isFound = true;
			cache.blockPos = blockPos; cache.blockPtr = hashEntry.ptr * SDF_BLOCK_SIZE3;
			return cache.blockPtr + linearIdx;
		}

		if (hashEntry.offset < 1) break;
		hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
	}

	isFound = false;
	return -1;
}

_CPU_AND_GPU_CODE_ inline int findVoxel(const CONSTPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexData) *voxelIndex, ORUVector3i point, THREADPTR(bool) &isFound)
{
	ITMLib::Objects::ITMVoxelBlockHash::IndexCache cache;
	return findVoxel(voxelIndex, point, isFound, cache);
}

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline TVoxel readVoxel(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexData) *voxelIndex,
	const THREADPTR(ORUVector3i) & point, THREADPTR(bool) &isFound, THREADPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexCache) & cache)
{
//	int voxelAddress = findVoxel(voxelIndex, point, isFound, cache);
//	return isFound ? voxelData[voxelAddress] : TVoxel();
	ORUVector3i blockPos;
	int linearIdx = pointToVoxelBlockPos(point, blockPos);//ZH: this is the lenearIdx in one voxel  block

	if IS_EQUAL3(blockPos, cache.blockPos)
	{
		isFound = true;
		return voxelData[cache.blockPtr + linearIdx];
	}

	int hashIdx = hashIndex(blockPos);

	while (true) 
	{
		ITMHashEntry hashEntry = voxelIndex[hashIdx];

		if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= 0)
		{
			isFound = true;
			cache.blockPos = blockPos; cache.blockPtr = hashEntry.ptr * SDF_BLOCK_SIZE3;//the cache here is used to store the voxel block pointor will be refrenced with high frequency(because the related voxels are organized in one voxel block)
			return voxelData[cache.blockPtr + linearIdx];
		}

		if (hashEntry.offset < 1) break;
		hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
	}

	isFound = false;
	return TVoxel();
}

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline TVoxel readVoxel(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexData) *voxelIndex,
	ORUVector3i point, THREADPTR(bool) &isFound)
{
	ITMLib::Objects::ITMVoxelBlockHash::IndexCache cache;
	return readVoxel(voxelData, voxelIndex, point, isFound, cache);
}

template<class TVoxel, class TIndex, class TCache>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_interpolated(const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(TIndex) *voxelIndex, ORUVector3f point, THREADPTR(bool) &isFound, THREADPTR(TCache) & cache)
{
	float res1, res2, v1, v2;
	ORUVector3f coeff; ORUVector3i pos; TO_INT_FLOOR3(pos, coeff, point);

	v1 = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 0, 0), isFound, cache).sdf;
	v2 = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 0, 0), isFound, cache).sdf;
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	v1 = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 1, 0), isFound, cache).sdf;
	v2 = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 1, 0), isFound, cache).sdf;
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	v1 = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 0, 1), isFound, cache).sdf;
	v2 = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 0, 1), isFound, cache).sdf;
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	v1 = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 1, 1), isFound, cache).sdf;
	v2 = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 1, 1), isFound, cache).sdf;
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	isFound = true;
	return TVoxel::SDF_valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline ORUVector4f readFromSDF_color4u_interpolated(const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(typename TIndex::IndexData) *voxelIndex, const THREADPTR(ORUVector3f) & point, 
	THREADPTR(typename TIndex::IndexCache) & cache)
{
	TVoxel resn; ORUVector3f ret = 0.0f; ORUVector4f ret4; bool isFound;
	ORUVector3f coeff; ORUVector3i pos; TO_INT_FLOOR3(pos, coeff, point);

	resn = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 0, 0), isFound, cache);
	ret += (1.0f - coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * resn.clr.toFloat();

	resn = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 0, 0), isFound, cache);
	ret += (coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * resn.clr.toFloat();

	resn = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 1, 0), isFound, cache);
	ret += (1.0f - coeff.x) * (coeff.y) * (1.0f - coeff.z) * resn.clr.toFloat();

	resn = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 1, 0), isFound, cache);
	ret += (coeff.x) * (coeff.y) * (1.0f - coeff.z) * resn.clr.toFloat();

	resn = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 0, 1), isFound, cache);
	ret += (1.0f - coeff.x) * (1.0f - coeff.y) * coeff.z * resn.clr.toFloat();

	resn = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 0, 1), isFound, cache);
	ret += (coeff.x) * (1.0f - coeff.y) * coeff.z * resn.clr.toFloat();;

	resn = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 1, 1), isFound, cache);
	ret += (1.0f - coeff.x) * (coeff.y) * coeff.z * resn.clr.toFloat();

	resn = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 1, 1), isFound, cache);
	ret += (coeff.x) * (coeff.y) * coeff.z * resn.clr.toFloat();

	ret4.x = ret.x; ret4.y = ret.y; ret4.z = ret.z; ret4.w = 255.0f;

	return ret4 / 255.0f;
}

template<class TVoxel>///template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline ORUVector3f computeSingleNormalFromSDF(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(ITMLib::Objects::ITMVoxelBlockHash::IndexData/*TIndex*/) *voxelIndex, const THREADPTR(ORUVector3f) &point)//(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(TIndex) *voxelIndex, const THREADPTR(Vector3f) &point)
{
	bool isFound;

	ORUVector3f ret;
	ORUVector3f coeff; ORUVector3i pos; TO_INT_FLOOR3(pos, coeff, point);
	ORUVector3f ncoeff(1.0f - coeff.x, 1.0f - coeff.y, 1.0f - coeff.z);

	// all 8 values are going to be reused several times
	ORUVector4f front, back;
	front.x = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 0, 0), isFound).sdf;
	front.y = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 0, 0), isFound).sdf;
	front.z = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 1, 0), isFound).sdf;
	front.w = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 1, 0), isFound).sdf;
	back.x  = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 0, 1), isFound).sdf;
	back.y  = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 0, 1), isFound).sdf;
	back.z  = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 1, 1), isFound).sdf;
	back.w  = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 1, 1), isFound).sdf;

	ORUVector4f tmp;
	float p1, p2, v1;
	// gradient x
	p1 = front.x * ncoeff.y * ncoeff.z +
	     front.z *  coeff.y * ncoeff.z +
	     back.x  * ncoeff.y *  coeff.z +
	     back.z  *  coeff.y *  coeff.z;
	tmp.x = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(-1, 0, 0), isFound).sdf;
	tmp.y = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(-1, 1, 0), isFound).sdf;
	tmp.z = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(-1, 0, 1), isFound).sdf;
	tmp.w = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(-1, 1, 1), isFound).sdf;
	p2 = tmp.x * ncoeff.y * ncoeff.z +
	     tmp.y *  coeff.y * ncoeff.z +
	     tmp.z * ncoeff.y *  coeff.z +
	     tmp.w *  coeff.y *  coeff.z;
	v1 = p1 * coeff.x + p2 * ncoeff.x;

	p1 = front.y * ncoeff.y * ncoeff.z +
	     front.w *  coeff.y * ncoeff.z +
	     back.y  * ncoeff.y *  coeff.z +
	     back.w  *  coeff.y *  coeff.z;
	tmp.x = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(2, 0, 0), isFound).sdf;
	tmp.y = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(2, 1, 0), isFound).sdf;
	tmp.z = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(2, 0, 1), isFound).sdf;
	tmp.w = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(2, 1, 1), isFound).sdf;
	p2 = tmp.x * ncoeff.y * ncoeff.z +
	     tmp.y *  coeff.y * ncoeff.z +
	     tmp.z * ncoeff.y *  coeff.z +
	     tmp.w *  coeff.y *  coeff.z;

	ret.x = TVoxel::SDF_valueToFloat(p1 * ncoeff.x + p2 * coeff.x - v1);

	// gradient y
	p1 = front.x * ncoeff.x * ncoeff.z +
	     front.y *  coeff.x * ncoeff.z +
	     back.x  * ncoeff.x *  coeff.z +
	     back.y  *  coeff.x *  coeff.z;
	tmp.x = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, -1, 0), isFound).sdf;
	tmp.y = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, -1, 0), isFound).sdf;
	tmp.z = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, -1, 1), isFound).sdf;
	tmp.w = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, -1, 1), isFound).sdf;
	p2 = tmp.x * ncoeff.x * ncoeff.z +
	     tmp.y *  coeff.x * ncoeff.z +
	     tmp.z * ncoeff.x *  coeff.z +
	     tmp.w *  coeff.x *  coeff.z;
	v1 = p1 * coeff.y + p2 * ncoeff.y;

	p1 = front.z * ncoeff.x * ncoeff.z +
	     front.w *  coeff.x * ncoeff.z +
	     back.z  * ncoeff.x *  coeff.z +
	     back.w  *  coeff.x *  coeff.z;
	tmp.x = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 2, 0), isFound).sdf;
	tmp.y = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 2, 0), isFound).sdf;
	tmp.z = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 2, 1), isFound).sdf;
	tmp.w = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 2, 1), isFound).sdf;
	p2 = tmp.x * ncoeff.x * ncoeff.z +
	     tmp.y *  coeff.x * ncoeff.z +
	     tmp.z * ncoeff.x *  coeff.z +
	     tmp.w *  coeff.x *  coeff.z;

	ret.y = TVoxel::SDF_valueToFloat(p1 * ncoeff.y + p2 * coeff.y - v1);

	// gradient z
	p1 = front.x * ncoeff.x * ncoeff.y +
	     front.y *  coeff.x * ncoeff.y +
	     front.z * ncoeff.x *  coeff.y +
	     front.w *  coeff.x *  coeff.y;
	tmp.x = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 0, -1), isFound).sdf;
	tmp.y = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 0, -1), isFound).sdf;
	tmp.z = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 1, -1), isFound).sdf;
	tmp.w = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 1, -1), isFound).sdf;
	p2 = tmp.x * ncoeff.x * ncoeff.y +
	     tmp.y *  coeff.x * ncoeff.y +
	     tmp.z * ncoeff.x *  coeff.y +
	     tmp.w *  coeff.x *  coeff.y;
	v1 = p1 * coeff.z + p2 * ncoeff.z;

	p1 = back.x * ncoeff.x * ncoeff.y +
	     back.y *  coeff.x * ncoeff.y +
	     back.z * ncoeff.x *  coeff.y +
	     back.w *  coeff.x *  coeff.y;
	tmp.x = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 0, 2), isFound).sdf;
	tmp.y = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 0, 2), isFound).sdf;
	tmp.z = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(0, 1, 2), isFound).sdf;
	tmp.w = readVoxel(voxelData, voxelIndex, pos + ORUVector3i(1, 1, 2), isFound).sdf;
	p2 = tmp.x * ncoeff.x * ncoeff.y +
	     tmp.y *  coeff.x * ncoeff.y +
	     tmp.z * ncoeff.x *  coeff.y +
	     tmp.w *  coeff.x *  coeff.y;

	ret.z = TVoxel::SDF_valueToFloat(p1 * ncoeff.z + p2 * coeff.z - v1);

	return ret;
}

template<bool hasColor,class TVoxel,class TIndex> struct VoxelColorReader;

template<class TVoxel, class TIndex>
struct VoxelColorReader<false,TVoxel,TIndex> {
	_CPU_AND_GPU_CODE_ static ORUVector4f interpolate(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
		const THREADPTR(ORUVector3f) & point)
	{ return ORUVector4f(0.0f,0.0f,0.0f,0.0f); }
};

template<class TVoxel, class TIndex>
struct VoxelColorReader<true,TVoxel,TIndex> {
	_CPU_AND_GPU_CODE_ static ORUVector4f interpolate(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
		const THREADPTR(ORUVector3f) & point)
	{
		typename TIndex::IndexCache cache;
		return readFromSDF_color4u_interpolated<TVoxel,TIndex>(voxelData, voxelIndex, point, cache);
	}
};
