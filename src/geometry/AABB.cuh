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
//  aabb() = default;
    __host__ __device__ aabb(const point3 &p) : minimum(p), maximum(p) {}

    __host__ __device__ aabb(point3 a, point3 b) : minimum(std::move(a)), maximum(std::move(b)) {}

    __host__ __device__ aabb(const aabb &other) = default;

    __host__ __device__ point3 min() const {
        return minimum;
    };

    __host__ __device__ point3 max() const {
        return maximum;
    };
    // old version
    __host__ __device__ bool hit_old(const ray &r, float t_min, float t_max) {
        // 判断光线 与 AABB 是否相交，判断与三个面的交面，是否有重合
        for (int i = 0; i < 3; i++) {
            auto t0 =
                    std::min((minimum[i] - r.origin()[i]) / r.direction()[i],
                             (maximum[i] - r.origin()[i]) / r.direction()[i]);
            auto t1 =
                    std::max((minimum[i] - r.origin()[i]) / r.direction()[i],
                             (maximum[i] - r.origin()[i]) / r.direction()[i]);
            t_min = std::max(t0, t_min);
            t_max = std::min(t1, t_max);
            if (t_max <= t_min)
                return false;
        }
        return true;
    }
    // Andrew Kensler 优化
    __host__ __device__  bool hit(const ray &r, float t_min, float t_max) const {
        for (int i = 0; i < 3; i++) {
            auto invD = 1.0f / r.direction()[i];
            auto t0 = (minimum[i] - r.origin()[i]) * invD;
            auto t1 = (maximum[i] - r.origin()[i]) * invD;
            if (invD < 0.0f)
                std::swap(t0, t1);
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min)
                return false;
        }
        return true;
    }
};
// 包围和
__host__ __device__  inline aabb surrounding_box(aabb box0, aabb box1) {
    // clang-format off
    point3 small(std::min(box0.min().x(), box1.min().x()),
                 std::min(box0.min().y(), box1.min().y()),
                 std::min(box0.min().z(), box1.min().z()));

    point3 big(std::max(box0.max().x(), box1.max().x()),
               std::max(box0.max().y(), box1.max().y()),
               std::max(box0.max().z(), box1.max().z()));
    // clang-format on
    return {small, big};
}

#endif