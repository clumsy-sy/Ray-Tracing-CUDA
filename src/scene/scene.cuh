#ifndef SCENE_HPP
#define SCENE_HPP

#include "../geometry/hittablelist.cuh"
#include "balls_world.cuh"

__host__ inline auto choose_scene(uint32_t opt, float &aspect_ratio, int &image_width, float &vfov, point3 &lookfrom,
    point3 &lookat, color &background) -> hittable_list {
  lookfrom = point3(13, 2, 3);
  // lookfrom = point3(4, 4, 20);
  lookat = point3(0, 0, 0);
  background = color(0.70, 0.80, 1.00);

  switch (opt) {
  case 1:
    return random_balls_world();

  default:
    return random_balls_world();
  }
}

#endif