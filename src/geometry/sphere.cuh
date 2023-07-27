#ifndef SPHERE_H
#define SPHERE_H

#include <cmath>

#include "../global.cuh"
#include "hittable.cuh"

class sphere : public hittable {

public:
  point3 center;
  float radius{}, radius2{};
  std::shared_ptr<material> mat_ptr;

private:
  __host__ __device__ static void get_sphere_uv(const point3 &p, float &u, float &v) {
    /*
      纹理坐标 $(u, v) \in [0, 1]$ 需要被映射到球面上，使用极坐标就是 $(\theta, \phi)$
      u = \phi / 2 * PI , v = \theta / PI
      由于圆是 (x, y, z) 形式存储，所以需要从笛卡尔坐标系转为极坐标系
      https://raytracing.github.io/books/RayTracingTheNextWeek.html#solidtextures/texturecoordinatesforspheres
    */
    float theta = std::acos(-p.y());
    float phi = std::atan2(-p.z(), p.x()) + PI;

    u = phi / (2 * PI);
    v = theta / PI;
  }

public:
  sphere() = default;
    __host__ __device__ sphere(point3 c, float r, std::shared_ptr<material> m)
      : center(std::move(c)), radius(r), radius2(r * r), mat_ptr(std::move(m)){};

    __host__ __device__ bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const override;
    __host__ __device__ auto bounding_box(aabb &output_box) const -> bool override;
};

__host__ __device__ auto sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const -> bool {
  vec3f oc = r.origin() - center;
  auto a = r.direction().length_squared();
  auto half_b = dot(oc, r.direction());
  auto c = oc.length_squared() - radius2;
  // Find the nearest root that lies in the acceptable range.
  float x0, x1, root;
  if (solveQuadratic_halfb(a, half_b, c, x0, x1)) {
    if (x0 > t_min && x0 < t_max)
      root = x0;
    else if (x1 > t_min && x1 < t_max)
      root = x1;
    else
      return false;
  } else
    return false;

  rec.t = root;
  rec.p = r.at(rec.t);
  vec3f &&outward_normal = (rec.p - center) / radius;
  rec.set_face_normal(r, outward_normal);
  get_sphere_uv(outward_normal, rec.u, rec.v);
  rec.mat_ptr = mat_ptr;

  return true;
}

__host__ __device__  auto sphere::bounding_box(aabb &output_box) const -> bool {
  // 圆的 AABB 就是(圆心 - r)三个方向 和 （圆心 + r）三个方向
  output_box = aabb(center - vec3f(radius, radius, radius), center + vec3f(radius, radius, radius));
  return true;
}

#endif