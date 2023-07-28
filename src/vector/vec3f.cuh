#ifndef VEC3F_CUH
#define VEC3F_CUH

#include <cmath>
#include <curand_kernel.h>
#include <iostream>

#include "../global.cuh"

class vec3f {
public:
  float e[3];

public:
  __host__ __device__ vec3f() {}

  __host__ __device__ vec3f(float xx) {
    e[0] = xx;
    e[1] = xx;
    e[2] = xx;
  }
  __host__ __device__ vec3f(float xx, float yy, float zz) {
    e[0] = xx;
    e[1] = yy;
    e[2] = zz;
  }
  __host__ __device__ inline float x() const { return e[0]; }
  __host__ __device__ inline float y() const { return e[1]; }
  __host__ __device__ inline float z() const { return e[2]; }
  __host__ __device__ inline float r() const { return e[0]; }
  __host__ __device__ inline float g() const { return e[1]; }
  __host__ __device__ inline float b() const { return e[2]; }
  __host__ __device__ inline float operator[](int i) const { return e[i]; }
  __host__ __device__ inline float &operator[](int i) { return e[i]; };
  __host__ __device__ inline vec3f operator-() const {
    return {-e[0], -e[1], -e[2]};
  }

  __host__ __device__ inline vec3f operator*(const float &r) const {
    return {e[0] * r, e[1] * r, e[2] * r};
  }
  __host__ __device__ inline vec3f operator/(const float &r) const {
    return *this * (1 / r);
  }
  __host__ __device__ inline vec3f operator+(const vec3f &v) const {
    return {e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2]};
  }
  __host__ __device__ inline vec3f operator-(const vec3f &v) const {
    return {e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]};
  }
  __host__ __device__ inline vec3f operator*(const vec3f &v) const {
    return {e[0] * v.e[0], e[1] * v.e[1], e[2] * v.e[2]};
  }
  __host__ __device__ inline vec3f &operator+=(const vec3f &v) {
    *this = *this + v;
    return *this;
  }
  __host__ __device__ inline vec3f &operator*=(const float &r) {
    *this = *this * r;
    return *this;
  }
  __host__ __device__ inline vec3f &operator/=(const float &r) {
    *this = *this / r;
    return *this;
  }

  __host__ __device__ inline float sum() { return e[0] + e[1] + e[2]; }
  __host__ __device__ inline float length_squared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }
  __host__ __device__ inline float length() const {
    return std::sqrt(length_squared());
  }
  __host__ __device__ inline void sqrt() {
    e[0] = std::sqrt(e[0]);
    e[1] = std::sqrt(e[1]);
    e[2] = std::sqrt(e[2]);
  }
  __host__ __device__ inline bool near_zero() {
    return (std::fabs(e[0]) < esp) && (std::fabs(e[1]) < esp) &&
           (std::fabs(e[2]) < esp);
  }
};
__host__ __device__ inline vec3f operator*(const float &r, const vec3f &v) {
  return v * r;
}

inline std::istream &operator>>(std::istream &is, vec3f &v) {
  is >> v.e[0] >> v.e[1] >> v.e[2];
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const vec3f &v) {
  os << v.e[0] << " " << v.e[1] << " " << v.e[2];
  return os;
}
__host__ __device__ inline float dot(const vec3f &a, const vec3f &b) {
  return (a * b).sum();
}
__host__ __device__ inline vec3f cross(const vec3f &a, const vec3f &b) {
  // clang-fomat off
  return {a.e[1] * b.e[2] - a.e[2] * b.e[1], a.e[2] * b.e[0] - a.e[0] * b.e[2],
          a.e[0] * b.e[1] - a.e[1] * b.e[0]};
  // clang-fomat on
}
__host__ __device__ inline vec3f unit_vector(vec3f v) { return v / v.length(); }
__host__ __device__ inline vec3f reflect(const vec3f &v, const vec3f &n) {
  return v - 2 * dot(v, n) * n;
}
__host__ __device__ inline vec3f refract(const vec3f &uv, const vec3f &n,
                                         float etai_over_etat) {
  auto cos_theta = min(dot(-uv, n), 1.0f);
  vec3f r_out_perp = etai_over_etat * (uv + cos_theta * n);
  vec3f r_out_parallel =
      -std::sqrt(std::abs(1.0f - r_out_perp.length_squared())) * n;
  return r_out_perp + r_out_parallel;
}

__device__ float inline random_float(curandState &state, float min, float max) {
  return curand_uniform(&state) * (max - min) + min;
}

__device__ auto random_in_unit_disk(curandState &state) -> vec3f {
  while (true) {
    auto p = vec3f(random_float(state, -1.0f, 1.0f),
                   random_float(state, -1.0f, 1.0f), 0);
    if (p.length_squared() >= 1)
      continue;
    return p;
  }
}

__device__ inline vec3f random_vec3f(curandState &state) {
  return {curand_uniform(&state), curand_uniform(&state),
          curand_uniform(&state)};
}
__device__ inline static auto random_vec3f(curandState &state, float min,
                                           float max) -> vec3f {
  return {random_float(state, min, max), random_float(state, min, max),
          random_float(state, min, max)};
}

__device__ inline auto random_in_unit_sphere(curandState &state) -> vec3f {
  while (true) {
    auto p = random_vec3f(state, -1, 1);
    if (p.length_squared() >= 1)
      continue;
    return p;
  }
}
__device__ inline auto random_unit_vector(curandState &state) -> vec3f {
  return unit_vector(random_in_unit_sphere(state));
}
__device__ inline auto random_in_hemisphere(const vec3f &normal,
                                            curandState &state) -> vec3f {
  // 判断与法线的位置关系
  vec3f in_unit_sphere = random_in_unit_sphere(state);
  if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
    return in_unit_sphere;
  else
    return -in_unit_sphere;
}

inline auto lerp(const vec3f &a, const vec3f &b, const float &t) -> vec3f {
  return a * (1 - t) + b * t;
}

// point3 and color can use vector to representation
using point3 = vec3f;
using color = vec3f;

#endif