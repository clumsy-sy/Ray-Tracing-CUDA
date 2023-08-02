#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "../geometry/ray.cuh"

struct hit_record;

/*
  base class: attenuation(颜色衰减) 产生的 scattered（散射光）
*/
class material {
public:
  __device__ virtual bool scatter(const ray &r_in, const hit_record &rec,
                                  color &attenuation, ray &scattered,
                                  curandState &state) const = 0;
};

#endif