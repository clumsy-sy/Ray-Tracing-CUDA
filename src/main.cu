#include "cudaGlobal.cuh"
#include "external/BMP.cuh"
#include "global.cuh"
#include "renderer/Renderer.cuh"
#include "scene/balls_world.cuh"
#include "vector/vec3f.cuh"
#include <iostream>

int main(int argc, const char **argv) {
  std::string photoname = "image2.bmp";
  if (argc >= 2) {
    photoname = std::string(argv[1]);
  }
  // photo size
  float aspect_ratio = 3.0 / 2.0;
  uint32_t image_width = 1200;

  point3 lookfrom(13, 2, 3), lookat(0, 0, 0);
  float vfov = 20.0;

  size_t limit = 4096;
  cudaDeviceSetLimit(cudaLimitStackSize, limit);
  hittable **world = nullptr;
  cudaMalloc((void **)&world, sizeof(hittable **));
  random_balls_world<<<1, 1>>>(world);
  cudaDeviceSynchronize();
  cudaDeviceSetLimit(cudaLimitStackSize, 1024);

  // Camera
  vec3f vup(0, 1, 0);
  float dist_to_focus = (lookfrom - lookat).length();
  float aperture = 0.1;

  camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture,
             dist_to_focus);
  // Renderer
  Renderer renderer(world, aspect_ratio, image_width, cam);
  renderer.set_photo_name(photoname);
  renderer.set_samples_per_pixel(64);
  renderer.set_max_depth(5);
  renderer.render();
  return 0;
}
