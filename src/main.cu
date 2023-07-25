#include "external/BMP.cuh"
#include "geometry/ray.cuh"
#include "global.cuh"
#include "vector/vec3f.cuh"
#include <cuda_profiler_api.h>
#include <curand_kernel.h>
#include <iostream>

__host__ void CudaAllocErrorMSG(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device %s (error code %s)!\n", msg,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ void CudaCopyErrorMSG(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy %s (error code %s)!\n", msg,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__device__ color ray_color(const ray &r) {
  vec3f unit_direction = unit_vector(r.direction());
  auto t = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

__global__ void rand_init(curandState *rand_state, int width, int height,
                          long clock_for_rand) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  curand_init(212, x, 0, &rand_state[x]);
}

// __device__ int image_width = 1200;
// __device__ int image_height = 675;

__global__ void pixel_draw(uint32_t image_width, uint32_t image_height,
                           vec3f origin, vec3f vertical, vec3f horizontal,
                           vec3f lower_left_corner, uint8_t *image) {
  uint32_t now = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t i = now % image_width;
  uint32_t j = now / image_width;

  auto u = float(i) / (image_width - 1);
  auto v = float(j) / (image_height - 1);
  ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
  color pixel_color = ray_color(r);

  uint32_t idx = now * 3;
  image[idx + 0] = static_cast<uint8_t>(255.999 * pixel_color.b());
  image[idx + 1] = static_cast<uint8_t>(255.999 * pixel_color.g());
  image[idx + 2] = static_cast<uint8_t>(255.999 * pixel_color.r());
}

int main() {

  // Image
  const auto aspect_ratio = 16.0 / 9.0;
  const int image_width = 1200;
  const int image_height = static_cast<int>(image_width / aspect_ratio);

  // Camera

  auto viewport_height = 2.0;
  auto viewport_width = aspect_ratio * viewport_height;
  auto focal_length = 1.0;

  auto origin = point3(0, 0, 0);
  auto horizontal = vec3f(viewport_width, 0, 0);
  auto vertical = vec3f(0, viewport_height, 0);
  auto lower_left_corner =
      origin - horizontal / 2 - vertical / 2 - vec3f(0, 0, focal_length);

  uint32_t PhotoSize = image_width * image_height * 3 * sizeof(uint8_t);
  // grid block
  uint32_t threadsPerBlock = 256;
  uint32_t blocksPerGrid =
      (image_height * image_width + threadsPerBlock - 1) / threadsPerBlock;

  // Render
  bmp::bitmap photo(image_width, image_height); // photo
  std::cout << "Photo Size : " << image_width << ' ' << image_height << "\n";

  // CUDA
  cudaError_t err = cudaSuccess;

  // CUDA rand

  // long clock_for_rand = clock();
  // curandState* rand_state;
  // err = cudaMalloc((void**)&rand_state, PhotoSize * sizeof(curandState));
  // CudaAllocErrorMSG(err, "rand_state");
  // rand_init<<<blocksPerGrid, threadsPerBlock>>>(rand_state, image_width,
  // image_height, clock_for_rand);

  // CUDA alloc
  uint8_t *photoInGPU = NULL;
  err = cudaMalloc((void **)&photoInGPU, PhotoSize);
  CudaAllocErrorMSG(err, "photoInGPU");

  pixel_draw<<<blocksPerGrid, threadsPerBlock>>>(image_width, image_height,
                                                 origin, vertical, horizontal,
                                                 lower_left_corner, photoInGPU);

  cudaDeviceSynchronize();

  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(photo.image.data(), photoInGPU, PhotoSize,
                   cudaMemcpyDeviceToHost);

  CudaCopyErrorMSG(err, "Device to Host");

  cudaFree(photoInGPU);

  cudaDeviceSynchronize();
  cudaProfilerStop();

  std::cout << (int)photo.image[10000][0] << " " << (int)photo.image[10000][1]
            << " " << (int)photo.image[10000][2] << " " << std::endl;

  photo.generate("test.bmp");
  return 0;
}
