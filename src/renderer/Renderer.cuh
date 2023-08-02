#ifndef RENDERER_HPP
#define RENDERER_HPP

#include "../camera/camera.cuh"
#include "../external/BMP.cuh"
#include "../geometry/hittablelist.cuh"
#include "../geometry/ray.cuh"
#include "../global.cuh"
#include "../material/material.cuh"
#include <curand_kernel.h>

__constant__ uint32_t msg[4];

__global__ void rand_init(curandState *rand_state, long clock_for_rand) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  curand_init(clock_for_rand, x, 0, &rand_state[x]);
}

__device__ color ray_color(const ray &r, hittable **world, uint32_t depth,
                           curandState &state) {
  if (depth <= 0)
    return {0, 0, 0};
  hit_record rec;
  if ((hittable *)(*world)->hit(r, 0.001, infinity, rec)) {
    ray scattered;
    color attenuation;
    if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, state))
      return attenuation * ray_color(scattered, world, depth - 1, state);
    return {0, 0, 0};
  }
  vec3f unit_direction = unit_vector(r.direction());
  float t = 0.5f * (unit_direction.y() + 1.0);
  return (1.0f - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

__device__ __forceinline__ color warpReduceSum(color res) {
  for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    res[0] += __shfl_down_sync(0xffffffff, res[0], offset, warpSize);
    res[1] += __shfl_down_sync(0xffffffff, res[1], offset, warpSize);
    res[2] += __shfl_down_sync(0xffffffff, res[2], offset, warpSize);
  }
  return res;
}

__global__ void draw(curandState *rand_state, camera cam, hittable **world,
                     float *photo) {
  uint32_t i = blockIdx.x;
  uint32_t j = blockIdx.y;

  uint32_t now = j * (msg[0] + 1) + i;

  color res(0.0f, 0.0f, 0.0f);
  for (uint32_t s = 0; s < msg[2]; ++s) {
    float u =
        ((float)i + curand_uniform(&rand_state[now + threadIdx.x * 32 + s])) /
        (float)msg[0];
    float v =
        ((float)j + curand_uniform(&rand_state[now + threadIdx.x * 32 + s])) /
        (float)msg[1];
    ray r = cam.get_ray(u, v, rand_state[now]);
    res += ray_color(r, world, msg[3], rand_state[now]);
  }
  res = warpReduceSum(res);
  if (threadIdx.x == 0) {
    uint32_t idx = now * 3;
    photo[idx + 0] = res[0];
    photo[idx + 1] = res[1];
    photo[idx + 2] = res[2];
  }
}

class Renderer {
public:
  std::string photoname = "image.bmp";
  camera cam;
  hittable **world;
  float aspect_ratio = 16.0 / 9.0;
  uint32_t image_width = 1200;
  uint32_t image_height = static_cast<uint32_t>(image_width / aspect_ratio);
  uint32_t samples_per_pixel = 64; // 单素采样数
  uint32_t max_depth = 5;          // 光线递归深度
  uint32_t threadsPerBlock = 256;  // 线程数
  uint32_t rayPerThread = samples_per_pixel / 32;
  uint32_t blocksPerGrid =
      (image_height * image_width + threadsPerBlock - 1) / threadsPerBlock;
  //    color background = color(0, 0, 0); // 背景辐射

public:
  Renderer() = default;

  Renderer(hittable **hitlist, float ratio, uint32_t width, camera c)
      : world(hitlist), aspect_ratio(ratio), image_width(width), cam(c) {
    image_height = static_cast<uint32_t>(image_width / aspect_ratio);
  }

  auto set_camera(camera &c) { cam = c; }

  auto set_photo_name(std::string name) { photoname = std::move(name); }

  auto set_samples_per_pixel(uint32_t samples) {
    samples_per_pixel = samples;
    rayPerThread = samples_per_pixel / 32;
  }

  auto set_max_depth(uint32_t depth) { max_depth = depth; }

  auto set_threadsPerBlock(uint32_t num) {
    threadsPerBlock = num;
    blocksPerGrid =
        (image_height * image_width + threadsPerBlock - 1) / threadsPerBlock;
  }

  auto render() {
    bmp::bitmap photo(image_width, image_height); // photo
    std::cout << "Photo Size : " << image_width << ' ' << image_height << "\n";

    uint32_t PhotoSize = image_width * image_height * 3 * sizeof(float);
    // CUDA
    cudaError_t err = cudaSuccess;

    // uint32_t msg
    /*
     * msg[0] = image_width - 1; msg[1] = image_height - 1;
     * msg[2] = rayPerThread; msg[3] = max_depth;
     */
    uint32_t msgHost[4] = {image_width - 1, image_height - 1, rayPerThread,
                           max_depth};
    cudaMemcpyToSymbol(msg, msgHost, 4 * sizeof(uint32_t));

    // CUDA rand
    long clock_for_rand = clock();
    curandState *rand_state;
    err =
        cudaMalloc((void **)&rand_state,
                   image_height * image_width * sizeof(curandState) + 32 * 32);
    CudaAllocErrorMSG(err, "rand_state");
    rand_init<<<blocksPerGrid, threadsPerBlock>>>(rand_state, clock_for_rand);

    // CUDA alloc
    float *photoInGPU = nullptr;
    err = cudaMalloc((void **)&photoInGPU, PhotoSize);
    CudaAllocErrorMSG(err, "photoInGPU");

    draw<<<dim3{image_width, image_height}, 32>>>(rand_state, cam, world,
                                                  photoInGPU);
    //        draw<<<dim3{10, 10}, 32>>>(rand_state, cam, world, photoInGPU);

    // CUAD world

    cudaDeviceSynchronize();

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(photo.fimage.data(), photoInGPU, PhotoSize,
                     cudaMemcpyDeviceToHost);

    CudaCopyErrorMSG(err, "Device to Host");

    cudaFree(photoInGPU);

    photo.floatToUint(samples_per_pixel);

    photo.generate(photoname);
  }
};

//
//__global__ void draw(uint32_t image_width, uint32_t image_height,
//                     uint32_t samples_per_pixel, curandState *rand_state,
//                     camera cam, hittable **world, uint32_t max_depth,
//                     uint8_t *photo) {
//    uint32_t now = blockDim.x * blockIdx.x + threadIdx.x;
//    uint32_t i = now % image_width;
//    uint32_t j = now / image_width;
//
//    color res(0.0f, 0.0f, 0.0f);
//    for (uint32_t s = 0; s < samples_per_pixel; ++s) {
//        float u = ((float)i + curand_uniform(&rand_state[now])) /
//                  (float)(image_width - 1);
//        float v = ((float)j + curand_uniform(&rand_state[now])) /
//                  (float)(image_height - 1);
//        ray r = cam.get_ray(u, v, rand_state[now]);
//        res += ray_color(r, world, max_depth, rand_state[now]);
//    }
//
//    float scale = 1.0f / (float)samples_per_pixel;
//    res *= scale;
//    res.sqrt();
//    uint32_t idx = now * 3;
//    photo[idx + 0] = static_cast<uint8_t>(255.999 * res.b());
//    photo[idx + 1] = static_cast<uint8_t>(255.999 * res.g());
//    photo[idx + 2] = static_cast<uint8_t>(255.999 * res.r());
//}

#endif