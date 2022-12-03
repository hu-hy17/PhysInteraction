#pragma once

#ifndef _DUAL_QUATERNION_HPP_
#define _DUAL_QUATERNION_HPP_

#include <vector_types.h>
#include <vector_functions.h>
#include "vector_operations.hpp"

/*dual quaternion class for CUDA kernel*/
struct Quaternion
{
	__host__ __device__ Quaternion() {}
	__host__ __device__ Quaternion(float _w, float _x, float _y, float _z) : q0(make_float4(_x, _y, _z, _w)) {}
	__host__ __device__ Quaternion(const float4 &_q) : q0(_q) {}
	__host__ __device__ Quaternion(const mat33 &_rot)
	{
		float tr = _rot.m00() + _rot.m11() + _rot.m22();
		if (tr > 0) {
			float s = sqrtf(tr + 1.0f) * 2;
			q0.w = s*0.25f;
			q0.x = (_rot.m21() - _rot.m12()) / s;
			q0.y = (_rot.m02() - _rot.m20()) / s;
			q0.z = (_rot.m10() - _rot.m01()) / s;
		}
		else if ((_rot.m00() > _rot.m11()) && (_rot.m00() > _rot.m22())) {
			float s = sqrtf(1.0f + _rot.m00() - _rot.m11() - _rot.m22()) * 2;
			q0.w = (_rot.m21() - _rot.m12()) / s;
			q0.x = 0.25f*s;
			q0.y = (_rot.m01() + _rot.m10()) / s;
			q0.z = (_rot.m02() + _rot.m20()) / s;
		}
		else if (_rot.m11() > _rot.m22()) {
			float s = sqrtf(1.0f + _rot.m11() - _rot.m00() - _rot.m22()) * 2;
			q0.w = (_rot.m02() - _rot.m20()) / s;
			q0.x = (_rot.m01() + _rot.m10()) / s;
			q0.y = 0.25f*s;
			q0.z = (_rot.m12() + _rot.m21()) / s;
		}
		else {
			float s = sqrtf(1.0f + _rot.m22() - _rot.m00() - _rot.m11()) * 2;
			q0.w = (_rot.m10() - _rot.m01()) / s;
			q0.x = (_rot.m02() + _rot.m20()) / s;
			q0.y = (_rot.m12() + _rot.m21()) / s;
			q0.z = 0.25f*s;
		}
	}

	__host__ __device__ float& x() { return q0.x; }
	__host__ __device__ float& y() { return q0.y; }
	__host__ __device__ float& z() { return q0.z; }
	__host__ __device__ float& w() { return q0.w; }

	__host__ __device__ const float& x() const { return q0.x; }
	__host__ __device__ const float& y() const { return q0.y; }
	__host__ __device__ const float& z() const { return q0.z; }
	__host__ __device__ const float& w() const { return q0.w; }

	__host__ __device__ Quaternion conjugate() const { return Quaternion(q0.w, -q0.x, -q0.y, -q0.z); }
	__host__ __device__ float square_norm() const { return q0.w*q0.w + q0.x*q0.x + q0.y*q0.y + q0.z*q0.z; }
	__host__ __device__ float norm() const { return sqrtf(square_norm()); }
	__host__ __device__ float dot(const Quaternion &_quat) const { return q0.w*_quat.w() + q0.x*_quat.x() + q0.y*_quat.y() + q0.z*_quat.z(); }
	__host__ __device__ void normalize() { q0 = ::normalized(q0); }
	__host__ __device__ Quaternion normalized() const { Quaternion q(*this); q.normalize(); return q; }

	__host__ __device__ mat33 matrix() const
	{
		/*normalize quaternion before converting to so3 matrix*/
		Quaternion q(*this);
		q.normalize();

		mat33 rot;
		rot.m00() = 1 - 2 * q.y()*q.y() - 2 * q.z()*q.z();
		rot.m01() = 2 * q.x()*q.y() - 2 * q.z()*q.w();
		rot.m02() = 2 * q.x()*q.z() + 2 * q.y()*q.w();
		rot.m10() = 2 * q.x()*q.y() + 2 * q.z()*q.w();
		rot.m11() = 1 - 2 * q.x()*q.x() - 2 * q.z()*q.z();
		rot.m12() = 2 * q.y()*q.z() - 2 * q.x()*q.w();
		rot.m20() = 2 * q.x()*q.z() - 2 * q.y()*q.w();
		rot.m21() = 2 * q.y()*q.z() + 2 * q.x()*q.w();
		rot.m22() = 1 - 2 * q.x()*q.x() - 2 * q.y()*q.y();
		return rot;
	}

	__host__ __device__ float3 vec() const { return make_float3(q0.x, q0.y, q0.z); }

	float4 q0;
};

__host__ __device__ __forceinline__ Quaternion operator+(const Quaternion &_left, const Quaternion &_right)
{
	return{ _left.w() + _right.w(), _left.x() + _right.x(), _left.y() + _right.y(), _left.z() + _right.z() };
}

__host__ __device__ __forceinline__ Quaternion operator*(float _scalar, const Quaternion &_quat)
{
	return{ _scalar*_quat.w(), _scalar*_quat.x(), _scalar*_quat.y(), _scalar*_quat.z() };
}

__host__ __device__ __forceinline__ Quaternion operator*(const Quaternion &_quat, float _scalar)
{
	return _scalar * _quat;
}

__host__ __device__ __forceinline__ Quaternion operator*(const Quaternion &_q0, const Quaternion &_q1)
{
	Quaternion q;
	q.w() = _q0.w()*_q1.w() - _q0.x()*_q1.x() - _q0.y()*_q1.y() - _q0.z()*_q1.z();
	q.x() = _q0.w()*_q1.x() + _q0.x()*_q1.w() + _q0.y()*_q1.z() - _q0.z()*_q1.y();
	q.y() = _q0.w()*_q1.y() - _q0.x()*_q1.z() + _q0.y()*_q1.w() + _q0.z()*_q1.x();
	q.z() = _q0.w()*_q1.z() + _q0.x()*_q1.y() - _q0.y()*_q1.x() + _q0.z()*_q1.w();

	return q;
}

struct DualNumber {
	__host__ __device__ DualNumber() : q0(0), q1(0) {}
	__host__ __device__ DualNumber(float _q0, float _q1) : q0(_q0), q1(_q1) {}

	__host__ __device__ DualNumber operator+(const DualNumber &_dn) const
	{
		return{ q0 + _dn.q0, q1 + _dn.q1 };
	}

	__host__ __device__ DualNumber& operator+=(const DualNumber &_dn)
	{
		*this = *this + _dn;
		return *this;
	}

	__host__ __device__ DualNumber operator*(const DualNumber &_dn) const
	{
		return{ q0*_dn.q0, q0*_dn.q1 + q1*_dn.q0 };
	}

	__host__ __device__ DualNumber& operator*=(const DualNumber &_dn)
	{
		*this = *this * _dn;
		return *this;
	}

	__host__ __device__ DualNumber reciprocal() const
	{
		return{ 1.0f / q0, -q1 / (q0*q0) };
	}

	__host__ __device__ DualNumber sqrt() const
	{
		return{ sqrtf(q0), q1 / (2 * sqrtf(q0)) };
	}

	float q0, q1;
};

// Forward declaration
struct DualQuaternion;
__host__ __device__ DualQuaternion operator*(const DualNumber &_dn, const DualQuaternion &_dq);

struct DualQuaternion {

	__host__ __device__ DualQuaternion() {}
	__host__ __device__ DualQuaternion(const Quaternion &_q0, const Quaternion &_q1) : q0(_q0), q1(_q1) {}
	__host__ __device__ DualQuaternion(const mat34 &T)
	{
		mat33 r = T.rot;
		float3 t = T.trans;
		DualQuaternion rot_part(Quaternion(r), Quaternion(0, 0, 0, 0));
		DualQuaternion vec_part(Quaternion(1, 0, 0, 0), Quaternion(0, 0.5f*t.x, 0.5f*t.y, 0.5f*t.z));
		*this = vec_part * rot_part;
	}

	__host__ __device__ DualQuaternion operator+(const DualQuaternion &_dq) const
	{
		Quaternion quat0(q0 + _dq.q0);
		Quaternion quat1(q1 + _dq.q1);
		return{ quat0, quat1 };
	}

	__host__ __device__ DualQuaternion operator*(const DualQuaternion &_dq) const
	{
		Quaternion quat0(q0*_dq.q0);
		Quaternion quat1(q1*_dq.q0 + q0*_dq.q1);
		return{ quat0, quat1 };
	}

	__host__ __device__ DualQuaternion operator*(const float &_w) const
	{
		return{ _w * q0, _w * q1 };
	}

	__host__ __device__ float3 operator*(const float3 &_p) const
	{
		float3 vec0 = q0.vec();
		float3 vec1 = q1.vec();
		return _p + 2 * (cross(vec0, cross(vec0, _p) + q0.w() * _p) + vec1 * q0.w() - vec0 * q1.w() + cross(vec0, vec1));
	}

	__host__ __device__ float3 rotate(const float3 &_p) const
	{
		float3 vec0 = q0.vec();
		return _p + 2 * cross(vec0, cross(vec0, _p) + q0.w() * _p);
	}

	__host__ __device__ DualQuaternion& operator+=(const DualQuaternion &_dq)
	{
		*this = *this + _dq;
		return *this;
	}

	__host__ __device__ DualQuaternion& operator*=(const DualQuaternion &_dq)
	{
		*this = *this * _dq;
		return *this;
	}

	__host__ __device__ DualQuaternion operator*(const DualNumber &_dn) const
	{
		return _dn * *this;
	}

	__host__ __device__ DualQuaternion& operator*=(const DualNumber &_dn)
	{
		*this = *this * _dn;
		return *this;
	}

	__host__ __device__ operator DualNumber() const
	{
		return DualNumber(q0.w(), q1.w());
	}

	__host__ __device__ DualQuaternion conjugate() const
	{
		return{ q0.conjugate(), q1.conjugate() };
	}

	__host__ __device__ DualNumber squared_norm() const
	{
		return *this * this->conjugate();
	}

	__host__ __device__ DualNumber norm() const
	{
		float a0 = q0.norm();
		float a1 = q0.dot(q1) / q0.norm();
		return{ a0, a1 };
	}

	__host__ __device__ DualQuaternion inverse() const
	{
		return this->conjugate() * this->squared_norm().reciprocal();
	}

	__host__ __device__ void normalize()
	{
		*this = *this * this->norm().reciprocal();
	}

	__host__ __device__ DualQuaternion normalized() const
	{
		return *this * this->norm().reciprocal();
	}

	__host__ __device__ operator mat34() const
	{
		mat33 r;
		float3 t;
		DualQuaternion quat_normalized = this->normalized();
		r = quat_normalized.q0.matrix();
		Quaternion vec_part = 2.0f*quat_normalized.q1*quat_normalized.q0.conjugate();
		t = vec_part.vec();

		return mat34(r, t);
	}

	Quaternion q0, q1;
};

__host__ __device__ __forceinline__ DualQuaternion operator*(const DualNumber &_dn, const DualQuaternion &_dq)
{
	Quaternion quat0 = _dn.q0*_dq.q0;
	Quaternion quat1 = _dn.q0*_dq.q1 + _dn.q1*_dq.q0;
	return{ quat0, quat1 };
}

#endif