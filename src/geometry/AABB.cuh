#ifndef AABB_HPP
#define AABB_HPP

#include "../vector/vec3f.cuh"
#include "ray.cuh"

// Axis-Aligned Bounding Boxes
// 三维空间的盒子，最小坐标 + 最大坐标 的 矩形
class aabb {
public:
  point3 minimum, maximum;

public:
  __device__ aabb() {
    minimum = {0.0f, 0.0f, 0.0f};
    maximum = {0.0f, 0.0f, 0.0f};
  }
  //    __device__ aabb(const point3 &p) : minimum(p), maximum(p) {}

  __device__ aabb(const point3 &a, const point3 &b) : minimum(a), maximum(b) {}

  __device__ point3 min() const { return minimum; };

  __device__ point3 max() const { return maximum; };
  // Andrew Kensler 优化
  __device__ bool hit(const ray &r, float t_min, float t_max) const {
    for (int i = 0; i < 3; i++) {
      auto invD = 1.0f / r.direction()[i];
      auto t0 = (minimum[i] - r.origin()[i]) * invD;
      auto t1 = (maximum[i] - r.origin()[i]) * invD;
      if (invD < 0.0f)
        swap(t0, t1);
      t_min = t0 > t_min ? t0 : t_min;
      t_max = t1 < t_max ? t1 : t_max;
      if (t_max <= t_min)
        return false;
    }
    return true;
  }
};
// 包围和
__device__ inline aabb surrounding_box(aabb box0, aabb box1) {
  // clang-format off
    point3 small(min(box0.min().x(), box1.min().x()),
                 min(box0.min().y(), box1.min().y()),
                 min(box0.min().z(), box1.min().z()));

    point3 big(max(box0.max().x(), box1.max().x()),
               max(box0.max().y(), box1.max().y()),
               max(box0.max().z(), box1.max().z()));
  // clang-format on
  return {small, big};
}

#endif