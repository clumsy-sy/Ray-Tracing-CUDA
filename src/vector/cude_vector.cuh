#ifndef CUDA_VECTOR_CUH
#define CUDA_VECTOR_CUH

class cuda_vector {
public:
  uint32_t size = 16;
  uint32_t idx = 0;
  void **begin = nullptr;

public:
  __device__ cuda_vector() {
    size = 16;
    idx = 0;
    begin = nullptr;
    cudaMalloc((void **)&begin, size * sizeof(void *));
  }

  __device__ bool vector_enlarge() {
    cudaError_t err = cudaSuccess;
    size *= 2;
    void **newbegin = nullptr;
    err = cudaMalloc((void **)&newbegin, size * sizeof(void *));
    if (err != cudaSuccess) {
      return false;
    }
    memcpy(newbegin, begin, idx * sizeof(void *));
    free(begin);
    begin = newbegin;
    return true;
  }

  __device__ bool push_back(void *p) {
    if (idx == size) {
      if (!vector_enlarge())
        return false;
    }
    begin[idx++] = p;
    return true;
  }

  __device__ void *operator[](uint32_t i) {
    if (i < idx)
      return begin[i];
    else
      return *begin;
  }

  __device__ void *get(uint32_t i) {
    if (i < idx)
      return begin[i];
    else
      return *begin;
  }

  __device__ ~cuda_vector() { cudaFree(*begin); }
};

#endif