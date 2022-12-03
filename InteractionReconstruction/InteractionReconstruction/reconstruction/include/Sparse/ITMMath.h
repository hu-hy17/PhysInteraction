// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

///#include "../../ORUtils/MathUtils.h"

#ifndef NULL
#define NULL 0
#endif

#ifndef __METALC__

typedef unsigned char ORUuchar;
typedef unsigned short ORUushort;
typedef unsigned int ORUuint;
typedef unsigned long ORUulong;

#include "Vector.h"
#include "Matrix.h"


typedef class ORUtils::Matrix3<float> ORUMatrix3f;//modify this "Matrix3f" to "ORUMatrix3f"
typedef class ORUtils::Matrix4<float> ORUMatrix4f;

typedef class ORUtils::Vector2<short> ORUVector2s;
typedef class ORUtils::Vector2<int> ORUVector2i;
typedef class ORUtils::Vector2<float> ORUVector2f;
typedef class ORUtils::Vector2<double> ORUVector2d;

typedef class ORUtils::Vector3<short> ORUVector3s;
typedef class ORUtils::Vector3<double> ORUVector3d;
typedef class ORUtils::Vector3<int> ORUVector3i;
typedef class ORUtils::Vector3<ORUuint> ORUVector3ui;
typedef class ORUtils::Vector3<ORUuchar> ORUVector3u;
typedef class ORUtils::Vector3<float> ORUVector3f;

typedef class ORUtils::Vector4<float> ORUVector4f;
typedef class ORUtils::Vector4<int> ORUVector4i;
typedef class ORUtils::Vector4<short> ORUVector4s;
typedef class ORUtils::Vector4<ORUuchar> ORUVector4u;

typedef class ORUtils::Vector6<float> ORUVector6f;


#ifndef TO_INT_ROUND3
#define TO_INT_ROUND3(x) (x).toIntRound()
#endif

#ifndef TO_INT_ROUND4
#define TO_INT_ROUND4(x) (x).toIntRound()
#endif

#ifndef TO_INT_FLOOR3
#define TO_INT_FLOOR3(inted, coeffs, in) inted = (in).toIntFloor(coeffs)
#endif

#ifndef TO_SHORT_FLOOR3
#define TO_SHORT_FLOOR3(x) (x).toShortFloor()
#endif

#ifndef TO_UCHAR3
#define TO_UCHAR3(x) (x).toUChar()
#endif

#ifndef TO_FLOAT3
#define TO_FLOAT3(x) (x).toFloat()
#endif

#ifndef TO_VECTOR3
#define TO_VECTOR3(a) (a).toVector3()
#endif

#ifndef IS_EQUAL3
#define IS_EQUAL3(a,b) (((a).x == (b).x) && ((a).y == (b).y) && ((a).z == (b).z))
#endif

#else

using namespace metal;

typedef float3x3 Matrix3f;
typedef float4x4 Matrix4f;

typedef short2 Vector2s;
typedef int2 Vector2i;
typedef float2 Vector2f;

typedef short3 Vector3s;
typedef int3 Vector3i;
typedef uint3 Vector3ui;
typedef uchar3 Vector3u;
typedef float3 Vector3f;

typedef float4 Vector4f;
typedef int4 Vector4i;
typedef short4 Vector4s;
typedef uchar4 Vector4u;

#ifndef TO_INT_ROUND3
#define TO_INT_ROUND3(x) (static_cast<int3>(round(x)))
#endif

#ifndef TO_INT_ROUND4
#define TO_INT_ROUND4(x) (static_cast<int4>(round(x)))
#endif

#ifndef TO_INT_FLOOR3
#define TO_INT_FLOOR3(inted, coeffs, in) { Vector3f flored(floor(in.x), floor(in.y), floor(in.z)); coeffs = in - flored; inted = Vector3i((int)flored.x, (int)flored.y, (int)flored.z); }
#endif

#ifndef TO_SHORT_FLOOR3
#define TO_SHORT_FLOOR3(x) (static_cast<short3>(floor(x)))
#endif

#ifndef TO_UCHAR3
#define TO_UCHAR3(x) (static_cast<uchar3>(x))
#endif

#ifndef TO_FLOAT3
#define TO_FLOAT3(x) (static_cast<float3>(x))
#endif

#ifndef TO_VECTOR3
#define TO_VECTOR3(a) ((a).xyz)
#endif

#ifndef IS_EQUAL3
#define IS_EQUAL3(a,b) (((a).x == (b).x) && ((a).y == (b).y) && ((a).z == (b).z))
#endif

#endif
