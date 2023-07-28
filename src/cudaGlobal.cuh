#ifndef CUDAGLOBAL_CUH
#define CUDAGLOBAL_CUH

#include <cstdio>
#include <cuda_runtime.h>

// __device__ need

__host__ __device__ void swap(float &a, float &b) {
  float tmp = a;
  a = b;
  b = tmp;
}

__host__ __device__ float min(float &a, float &b) { return a < b ? a : b; }

__host__ __device__ float max(float &a, float &b) { return a > b ? a : b; }

// CUDA ERROR Function

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

#endif