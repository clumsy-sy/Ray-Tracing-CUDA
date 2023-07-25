#ifndef DIELECTRIC_HPP
#define DIELECTRIC_HPP

#include "material.cuh"
#include "../vector/Vec3dx4.hpp"
#include "../geometry/hittable.hpp"

class dielectric : public material {
public:
  float ir; // 折射率

public:
  dielectric(float index_of_refraction) : ir(index_of_refraction) {}

  auto scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const -> bool override {
    attenuation = color(1.0, 1.0, 1.0);
    float refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

    Vec3d unit_direction = unit_vector(r_in.direction());

    float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    Vec3d direction;

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float()) // Schlick Approximation
      direction = reflect(unit_direction, rec.normal);
    else
      direction = refract(unit_direction, rec.normal, refraction_ratio);

    scattered = ray(rec.p, direction);

    return true;
  }

private:
  static auto reflectance(float cosine, float ref_idx) -> float {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
  }
};

#endif