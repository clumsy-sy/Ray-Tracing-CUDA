#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include "../vector/vec3f.cuh"
#include "AABB.cuh"
#include "ray.cuh"

class material;

/*
  光线与物体交的记录
*/
struct hit_record {
  point3 p{};          // 交点坐标
  vec3f normal{};      // 法线
  material *mat_ptr{}; // 材质
  float t{};           // 光线的 t
                       //    float u{}, v{};                       // 材质
  bool front_face{};   // 朝向

  __device__ inline void set_face_normal(const ray &r,
                                         const vec3f &outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class hittable {
public:
  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec) const = 0;

  __device__ virtual bool bounding_box(aabb &output_box) const = 0;
};

#endif