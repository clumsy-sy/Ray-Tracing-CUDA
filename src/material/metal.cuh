#ifndef METAL_HPP
#define METAL_HPP

#include "../geometry/hittable.cuh"
#include "material.cuh"

class metal : public material {
public:
  color albedo;
  float fuzz; // 模糊参数 0 为不扰动

public:
  __device__ metal(const color &a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

  __device__ bool scatter(const ray &r_in, const hit_record &rec,
                          color &attenuation, ray &scattered,
                          curandState &state) const override {
    vec3f reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(state));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }
};

#endif