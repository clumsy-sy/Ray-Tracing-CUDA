#ifndef BALLS_WORLD_cuh
#define BALLS_WORLD_cuh

#include "../geometry/BVH.cuh"
#include "../geometry/hittablelist.cuh"
#include "../geometry/sphere.cuh"
#include "../global.cuh"
#include "../material/dielectric.cuh"
#include "../material/lambertian.cuh"
#include "../material/metal.cuh"
#include "../vector/vec3f.cuh"
#include <curand_kernel.h>

__global__ void random_balls_world(hittable **total) {
  *total = new hittable_list;
  hittable_list *world = (hittable_list *)*total;
  printf("world size: %u, idx : %u, begin : %llu\n", world->size, world->idx,
         world->list);
  material *ground_material = new lambertian(color(0.5, 0.5, 0.5));
  hittable *ground = new sphere(point3(0, -2000, 0), 2000, ground_material);
  world->push_back(ground);

  // small balls

  hittable_list *box = new hittable_list;
  curandState *rand_state = new curandState;
  curand_init(0, blockDim.x * blockIdx.x + threadIdx.x, 0, rand_state);
  //    __syncthreads();

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      auto choose_mat = curand_uniform(rand_state);
      point3 center(a + 0.9f * curand_uniform(rand_state), 0.2f,
                    b + 0.9f * curand_uniform(rand_state));

      if ((center - point3(4, 0.2, 0)).length() > 0.9) {
        material *sphere_material;

        if (choose_mat < 0.8) {
          // diffuse
          auto albedo = random_vec3f(*rand_state) * random_vec3f(*rand_state);
          sphere_material = new lambertian(albedo);
          //                    world->push_back(new sphere(center, 0.2,
          //                    sphere_material));
          box->push_back(new sphere(center, 0.2, sphere_material));
        } else if (choose_mat < 0.95) {
          // metal
          auto albedo = random_vec3f(*rand_state, 0.5, 1);
          auto fuzz = random_float(*rand_state, 0, 0.5);
          sphere_material = new metal(albedo, fuzz);
          //                    world->push_back(new sphere(center, 0.2,
          //                    sphere_material));
          box->push_back(new sphere(center, 0.2, sphere_material));
        } else {
          // glass
          sphere_material = new dielectric(1.5);
          //                    world->push_back(new sphere(center, 0.2,
          //                    sphere_material));
          box->push_back(new sphere(center, 0.2, sphere_material));
        }
      }
    }
  }
  aabb **lrbox = (aabb **)malloc(2 * sizeof(aabb *));
  lrbox[0] = new aabb;
  lrbox[1] = new aabb;
  //    world->push_back(box);
  world->push_back(new bvh_node(box->list, 0, box->idx, lrbox, *rand_state));

  material *material1 = new dielectric(1.5f);
  hittable *ball1 = new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, material1);
  world->push_back(ball1);

  material *material2 = new lambertian(color(0.4f, 0.2f, 0.1f));
  hittable *ball2 = new sphere(point3(-4.0f, 1.0f, 0.0f), 1.0f, material2);
  world->push_back(ball2);

  auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0f);
  hittable *ball3 = new sphere(point3(4.0f, 1.0f, 0.0f), 1.0f, material3);
  world->push_back(ball3);

  printf("total: sphere : %u\n", world->idx);
  //    for (uint32_t i = 0; i < box->idx; i++) {
  //        printf("ball%d: %.5f %.5f %.5f\n", i, ((sphere *)
  //        box->list[i])->center[0],
  //               ((sphere *) box->list[i])->center[1], ((sphere *)
  //               box->list[i])->center[2]);
  //    }
}

#endif