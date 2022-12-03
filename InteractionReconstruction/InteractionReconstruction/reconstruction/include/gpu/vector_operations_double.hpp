#ifndef _VECTOR_OPERATION_DOUBLE_HPP_
#define _VECTOR_OPERATION_DOUBLE_HPP_

#include <vector_functions.h>

__host__ __device__ __forceinline__ double
dot(const double3& v1, const double3& v2)
{
	return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
}

__host__ __device__ __forceinline__ double3&
operator+=(double3& vec, const double& v)
{
	vec.x += v;  vec.y += v;  vec.z += v; return vec;
}

__host__ __device__ __forceinline__ double3
operator+(const double3& v1, const double3& v2)
{
	return make_double3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ __forceinline__ double3&
operator*=(double3& vec, const double& v)
{
	vec.x *= v;  vec.y *= v;  vec.z *= v; return vec;
}

__host__ __device__ __forceinline__ double3
operator-(const double3& v1, const double3& v2)
{
	return make_double3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ __forceinline__ double3
operator*(const double& v, const double3& v1)
{
	return make_double3(v * v1.x, v * v1.y, v * v1.z);
}

__host__ __device__ __forceinline__ double3
operator*(const double3& v1, const double& v)
{
	return make_double3(v1.x * v, v1.y * v, v1.z * v);
}

__host__ __device__ __forceinline__ double
fabs_sum(const double3 &v)
{
	return fabs(v.x) + fabs(v.y) + fabs(v.z);
}

__host__ __device__ __forceinline__ double
squared_norm(const double3 &v)
{
	return dot(v, v);
}

__host__ __device__ __forceinline__ double
norm(const double3& v)
{
	return sqrt(dot(v, v));
}

#if defined(__CUDACC__)
__host__ __device__ __forceinline__ double3
normalized(const double3& v)
{
	return v * rsqrt(dot(v, v));
}
#else
__host__ __device__ __forceinline__ double3
normalized(const double3 &v)
{
	return v * (1.0/sqrt(dot(v, v)));
}
#endif

__host__ __device__ __forceinline__ double3
cross(const double3& v1, const double3& v2)
{
	return make_double3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

__host__ __device__ __forceinline__ double
dot(const double4 &v1, const double4 &v2)
{
	return v1.w*v2.w+v1.x*v2.x+v1.y*v2.y+v1.z*v2.z;
}

__host__ __device__ __forceinline__ double
squared_norm(const double4 &v)
{
	return dot(v, v);
}

__host__ __device__ __forceinline__ double
norm(const double4 &v4)
{
	return sqrt(squared_norm(v4));
}

__host__ __device__ __forceinline__ double4 operator*(const double4 &v, double s)
{
	return make_double4(v.x*s, v.y*s, v.z*s, v.w*s);
}

__host__ __device__ __forceinline__ double4 operator*(double s, const double4 &v)
{
	return make_double4(s*v.x, s*v.y, s*v.z, s*v.w);
}

#if defined(__CUDACC__)
__host__ __device__ __forceinline__ double4
normalized(const double4& v)
{
	return v * rsqrt(dot(v, v));
}
#else

__host__ __device__ __forceinline__ double4
normalized(const double4 &v)
{
	return v * (1.0f/sqrt(dot(v, v)));
}
#endif

__host__ __device__ __forceinline__ double3
operator-(const double3 &v)
{
	return make_double3(-v.x, -v.y, -v.z);
}

__host__ __device__ __forceinline__ double4
operator-(const double4 &v)
{
	return make_double4(-v.x, -v.y, -v.z, -v.w);
}

struct mat33d {
	__host__ __device__ mat33d() {}
	__host__ __device__ mat33d(const double3 &_a0, const double3 &_a1, const double3 &_a2) { cols[0] = _a0; cols[1] = _a1; cols[2] = _a2; }
	__host__ __device__ mat33d(const double *_data)
	{
		/*_data MUST have at least 9 double elements, ctor does not check range*/
		cols[0] = make_double3(_data[0], _data[1], _data[2]);
		cols[1] = make_double3(_data[3], _data[4], _data[5]);
		cols[2] = make_double3(_data[6], _data[7], _data[8]);
	}
	__host__ __device__ const double& m00() const { return cols[0].x; }
	__host__ __device__ const double& m10() const { return cols[0].y; }
	__host__ __device__ const double& m20() const { return cols[0].z; }
	__host__ __device__ const double& m01() const { return cols[1].x; }
	__host__ __device__ const double& m11() const { return cols[1].y; }
	__host__ __device__ const double& m21() const { return cols[1].z; }
	__host__ __device__ const double& m02() const { return cols[2].x; }
	__host__ __device__ const double& m12() const { return cols[2].y; }
	__host__ __device__ const double& m22() const { return cols[2].z; }

	__host__ __device__ double& m00() { return cols[0].x; }
	__host__ __device__ double& m10() { return cols[0].y; }
	__host__ __device__ double& m20() { return cols[0].z; }
	__host__ __device__ double& m01() { return cols[1].x; }
	__host__ __device__ double& m11() { return cols[1].y; }
	__host__ __device__ double& m21() { return cols[1].z; }
	__host__ __device__ double& m02() { return cols[2].x; }
	__host__ __device__ double& m12() { return cols[2].y; }
	__host__ __device__ double& m22() { return cols[2].z; }

	__host__ __device__ mat33d transpose() const
	{
		double3 row0 = make_double3(cols[0].x, cols[1].x, cols[2].x);
		double3 row1 = make_double3(cols[0].y, cols[1].y, cols[2].y);
		double3 row2 = make_double3(cols[0].z, cols[1].z, cols[2].z);
		return mat33d(row0, row1, row2);
	}

	__host__ __device__ mat33d operator* (const mat33d &_mat) const
	{
		mat33d mat;
		mat.m00() = m00()*_mat.m00() + m01()*_mat.m10() + m02()*_mat.m20();
		mat.m01() = m00()*_mat.m01() + m01()*_mat.m11() + m02()*_mat.m21();
		mat.m02() = m00()*_mat.m02() + m01()*_mat.m12() + m02()*_mat.m22();
		mat.m10() = m10()*_mat.m00() + m11()*_mat.m10() + m12()*_mat.m20();
		mat.m11() = m10()*_mat.m01() + m11()*_mat.m11() + m12()*_mat.m21();
		mat.m12() = m10()*_mat.m02() + m11()*_mat.m12() + m12()*_mat.m22();
		mat.m20() = m20()*_mat.m00() + m21()*_mat.m10() + m22()*_mat.m20();
		mat.m21() = m20()*_mat.m01() + m21()*_mat.m11() + m22()*_mat.m21();
		mat.m22() = m20()*_mat.m02() + m21()*_mat.m12() + m22()*_mat.m22();
		return mat;
	}

	__host__ __device__ mat33d operator+ (const mat33d &_mat) const
	{
		mat33d mat_sum;
		mat_sum.m00() = m00() + _mat.m00();
		mat_sum.m01() = m01() + _mat.m01();
		mat_sum.m02() = m02() + _mat.m02();

		mat_sum.m10() = m10() + _mat.m10();
		mat_sum.m11() = m11() + _mat.m11();
		mat_sum.m12() = m12() + _mat.m12();

		mat_sum.m20() = m20() + _mat.m20();
		mat_sum.m21() = m21() + _mat.m21();
		mat_sum.m22() = m22() + _mat.m22();

		return mat_sum;
	}

	__host__ __device__ mat33d operator- (const mat33d &_mat) const
	{
		mat33d mat_diff;
		mat_diff.m00() = m00() - _mat.m00();
		mat_diff.m01() = m01() - _mat.m01();
		mat_diff.m02() = m02() - _mat.m02();

		mat_diff.m10() = m10() - _mat.m10();
		mat_diff.m11() = m11() - _mat.m11();
		mat_diff.m12() = m12() - _mat.m12();

		mat_diff.m20() = m20() - _mat.m20();
		mat_diff.m21() = m21() - _mat.m21();
		mat_diff.m22() = m22() - _mat.m22();

		return mat_diff;			
	}

	__host__ __device__ mat33d operator-() const
	{
		mat33d mat_neg;
		mat_neg.m00() = -m00();
		mat_neg.m01() = -m01();
		mat_neg.m02() = -m02();

		mat_neg.m10() = -m10();
		mat_neg.m11() = -m11();
		mat_neg.m12() = -m12();

		mat_neg.m20() = -m20();
		mat_neg.m21() = -m21();
		mat_neg.m22() = -m22();

		return mat_neg;
	}

	__host__ __device__ mat33d& operator*= (const mat33d &_mat)
	{
		*this = *this * _mat;
		return *this;
	}

	__host__ __device__ double3 operator* (const double3 &_vec) const
	{
		double x = m00()*_vec.x + m01()*_vec.y + m02()*_vec.z;
		double y = m10()*_vec.x + m11()*_vec.y + m12()*_vec.z;
		double z = m20()*_vec.x + m21()*_vec.y + m22()*_vec.z;
		return make_double3(x, y, z);
	}

	__host__ __device__ void set_identity()
	{
		cols[0] = make_double3(1, 0, 0);
		cols[1] = make_double3(0, 1, 0);
		cols[2] = make_double3(0, 0, 1);
	}

	__host__ __device__ static mat33d identity()
	{
		mat33d idmat;
		idmat.set_identity();
		return idmat;
	}

	double3 cols[3]; /*colume major*/
};

/*rotation and translation*/
struct mat34d {
	__host__ __device__ mat34d() {}
	__host__ __device__ mat34d(const mat33d &_rot, const double3 &_trans) : rot(_rot), trans(_trans) {}
	__host__ __device__ static mat34d identity()
	{
		return mat34d(mat33d::identity(), make_double3(0, 0, 0));
	}

	__host__ __device__ mat34d operator* (const mat34d &_right_se3) const
	{
		mat34d se3;
		se3.rot = rot*_right_se3.rot;
		se3.trans = rot*_right_se3.trans + trans;
		return se3;
	}

	__host__ __device__ mat34d& operator*= (const mat34d &_right_se3)
	{
		*this = *this * _right_se3;
		return *this;
	}

	mat33d rot;
	double3 trans;
};

/*outer production of two double3*/
__host__ __device__ __forceinline__ mat33d
outer_prod(const double3 &v0, const double3 &v1)
{
	return mat33d(v0*v1.x, v0*v1.y, v0*v1.z);
}

// symmetric inverse of mat33
__host__ __device__ __forceinline__
mat33d sym_inv(const mat33d &_A)
{
	double det = _A.m00()*_A.m11()*_A.m22() +
			   2*_A.m01()*_A.m02()*_A.m12() -
			    (_A.m00()*_A.m12()*_A.m12() +
			     _A.m11()*_A.m02()*_A.m02() +
				 _A.m22()*_A.m01()*_A.m01());

	mat33d A_inv;

	if (fabs(det) < 1e-10) {
		A_inv.set_identity();
	}
	else {
		double det_inv = 1/det;
		A_inv.m00() = det_inv*(_A.m11()*_A.m22() - _A.m12()*_A.m12());
		A_inv.m11() = det_inv*(_A.m00()*_A.m22() - _A.m02()*_A.m02());
		A_inv.m22() = det_inv*(_A.m00()*_A.m11() - _A.m01()*_A.m01());
		A_inv.m01() = A_inv.m10() = det_inv*(_A.m02()*_A.m12() - _A.m01()*_A.m22());
		A_inv.m02() = A_inv.m20() = det_inv*(_A.m01()*_A.m12() - _A.m02()*_A.m11());
		A_inv.m12() = A_inv.m21() = det_inv*(_A.m02()*_A.m01() - _A.m00()*_A.m12());
	}

	return A_inv;
}

#endif