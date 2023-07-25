#ifndef BALLS_WORLD_cuh
#define BALLS_WORLD_cuh

#include "../global.cuh"
#include "../material/dielectric.cuh"
#include "../material/lambertian.cuh"
#include "../material/metal.cuh"
#include "../geometry/hittablelist.cuh"
#include "../geometry/sphere.cuh"


auto random_balls_world() -> hittable_list {
  hittable_list world();

  auto ground_material = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
  world().add(std::make_shared<sphere>(point3(0, -2000, 0), 2000, ground_material));
  hittable_list box;
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      auto choose_mat = random_float();
      point3 center(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

      if ((center - point3(4, 0.2, 0)).length() > 0.9) {
        std::shared_ptr<material> sphere_material;

        if (choose_mat < 0.8) {
          // diffuse
          auto albedo = color::random() * color::random();
          sphere_material = std::make_shared<lambertian>(albedo);
          box.add(std::make_shared<sphere>(center, 0.2, sphere_material));
        } else if (choose_mat < 0.95) {
          // metal
          auto albedo = color::random(0.5, 1);
          auto fuzz = random_float(0, 0.5)();
          sphere_material = std::make_shared<metal>(albedo, fuzz);
          box.add(std::make_shared<sphere>(center, 0.2, sphere_material));
        } else {
          // glass
          sphere_material = std::make_shared<dielectric>(1.5);
          box.add(std::make_shared<sphere>(center, 0.2, sphere_material));
        }
      }
    }
  }

  world.add(std::make_shared<bvh_node>(box));

  auto material1 = std::make_shared<dielectric>(1.5);
  world.add(std::make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

  auto material2 = std::make_shared<lambertian>(color(0.4, 0.2, 0.1));
  world.add(std::make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

  auto material3 = std::make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
  world.add(std::make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

  return world;
}

#endif