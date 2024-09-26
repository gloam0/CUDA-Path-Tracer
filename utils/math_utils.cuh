#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <curand_kernel.h>

#include "math_constants.h"
#include "common.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
constexpr double deg_to_rad = M_PI / 180.;

////////////////////////////////////////////////////////////////////////////////////////////////
// Operator overloads

/* float3 */
__host__ __device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ __forceinline__ float3 operator*(const float3& a, const float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ float3 operator*(const float b, const float3& a) {
    return a * b;
}

__host__ __device__ __forceinline__ float3 operator/(const float3& a, const float b) {
    return a * (1/b);
}

__host__ __device__ __forceinline__ void operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__host__ __device__ __forceinline__ void operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__host__ __device__ __forceinline__ void operator*=(float3& a, const float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__host__ __device__ __forceinline__ void operator/=(float3& a, const float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

/* double3 */
__host__ __device__ __forceinline__ double3 operator+(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ __forceinline__ double3 operator-(const double3& a) {
    return make_double3(-a.x, -a.y, -a.z);
}

__host__ __device__ __forceinline__ double3 operator*(const double3& a, const float b) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ double3 operator*(const float b, const double3& a) {
    return a * b;
}

__host__ __device__ __forceinline__ double3 operator*(const double3& a, const double b) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ double3 operator*(const double b, const double3& a) {
    return a * b;
}

__host__ __device__ __forceinline__ double3 operator/(const double3& a, const float b) {
    return a * (1/b);
}

__host__ __device__ __forceinline__ double3 operator/(const float b, const double3& a) {
    return a * (1/b);
}

__host__ __device__ __forceinline__ void operator+=(double3& a, const double3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__host__ __device__ __forceinline__ void operator-=(double3& a, const double3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__host__ __device__ __forceinline__ void operator*=(double3& a, const float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__host__ __device__ __forceinline__ void operator/=(double3& a, const float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Vector math

__host__ __device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__ float dot(const double3& a, const double3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__ float3 cross(const float3& a, const float3& b) {
    return float3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__host__ __device__ __forceinline__ double3 cross(const double3& a, const double3& b) {
    return double3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__host__ __device__ __forceinline__ float3 elem_product(const float3& a, const float3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ __forceinline__ float3 normalize(const vec3& a) {
#ifdef __CUDA_ARCH__
    return a * rsqrtf(dot(a, a));
#else
    return a * (1.0f / sqrtf(dot(a, a)));
#endif
}

__host__ __device__ __forceinline__ double3 normalize(const double3& a) {
#ifdef __CUDA_ARCH__
    return a * rsqrtf(dot(a, a));
#else
    return a * (1.0f / sqrtf(dot(a, a)));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Quaternion math
// Derived from explanation at: https://www.3dgep.com/understanding-quaternions/
struct quaternion {
    float s;
    vec3 v;
};

inline quaternion operator*(const quaternion& a, const quaternion& b) {
    return quaternion{
        a.s * b.s - dot(a.v, b.v),
        a.s * b.v + b.s * a.v + cross(a.v, b.v)
    };
}

inline quaternion operator*(const quaternion& a, float b) {
    return quaternion{a.s * b, a.v * b};
}

inline quaternion operator/(const quaternion& a, float b) {
    return quaternion{a.s / b, a.v / b};
}

inline quaternion& operator*=(quaternion& a, const quaternion& b) {
    a = a * b;
    return a;
}

inline quaternion normalize(const quaternion& q) {
    float norm = sqrt(q.s * q.s + dot(q.v, q.v));
    return q / norm;
}

/// Make a rotor quaternion characterized by an axis and an angle delta in radians
inline quaternion make_rotor_quaternion(const vec3& axis, float theta) {
    vec3 u_ax = normalize(axis);
    double half_theta = theta * 0.5;
    return quaternion{cosf(half_theta), sinf(half_theta) * u_ax}; // Guaranteed to be normalized
}

/// Rotate vector with rotation characterized by rotor quaternion q
inline vec3 rotate_v(const vec3& v, const quaternion& q) {
    quaternion v_q{0.f, v};
    quaternion q_conj{q.s, -q.v};
    quaternion v_q_rot = q * v_q * q_conj;
    return v_q_rot.v;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Random generation

__device__ inline unsigned int xorshift32_i(unsigned int* seed) {
    unsigned int x = *seed;
    if (x == 0) x = 1;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *seed = x;
    return x;
}

__device__ inline float xorshift32_f_norm(unsigned int* seed) {
    // avoid fmul/fdiv conversion; mask mantissa, set exponent to 1.0, reinterpret bits
    // [3F800000 : 3FFFFFFFh] = [1.f : 2.f)
    return __uint_as_float((xorshift32_i(seed) & 0x007FFFFF) | 0x3F800000) - 1.f;
}

/// Generate a random vector with length < 1.
__device__ inline vec3 random_unit_vector(unsigned int* seed) {
    vec3 v;
    float len_squared;
    do {  /* Using xorshift, so long sequences of values satisfying
           * len_squared >= 1.f should not be possible */
        v.x = 2.f * xorshift32_f_norm(seed) - 1.f;
        v.y = 2.f * xorshift32_f_norm(seed) - 1.f;
        v.z = 2.f * xorshift32_f_norm(seed) - 1.f;
        len_squared = dot(v, v);
    } while (len_squared >= 1.f || len_squared < render::self_intersect_eps_sq);

    float inv_len = rsqrtf(len_squared);
    return vec3{v.x * inv_len, v.y * inv_len, v.z * inv_len};
}

/// Determine if the length of vector v is less than a minimum threshold.
__device__ inline bool near_zero(const vec3& v) {
    return dot(v, v) < render::self_intersect_eps_sq;
}

////////////////////////////////////////////////////////////////////////////////////////////////
#endif //MATH_UTILS_H
