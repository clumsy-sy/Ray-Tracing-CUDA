#ifndef LAMBERTIAN_HPP
#define LAMBERTIAN_HPP

#include "../geometry/hittable.cuh"
#include "material.cuh"

class lambertian : public material {
public:
  color albedo; // 以某种概率分布衰减，albedo / p

public:
  __device__ lambertian(const color &a) : albedo(a) {}

  __device__ bool __device__ scatter(const ray &, const hit_record &rec,
                                     color &attenuation, ray &scattered,
                                     curandState &state) const override {
    // 得到一个在交点的单位相切圆上的随机散射方向向量
    auto scatter_direction = rec.normal + random_unit_vector(state);

    // Catch degenerate scatter direction
    if (scatter_direction.near_zero())
      scatter_direction = rec.normal;

    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;
  }
};

#endif