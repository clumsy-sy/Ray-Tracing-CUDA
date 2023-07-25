#ifndef RAY_CUH
#define RAY_CUH

#include "../vector/vec3f.cuh"

class ray {
public:
    point3 orig;
    vec3f dir;

public:
//   __device__ ray() = default;
    __host__ __device__ ray(point3 origin, vec3f direction)
            : orig(std::move(origin)), dir(std::move(direction)) {}

    __host__ __device__ point3 origin() const { return orig; }

    __host__ __device__ vec3f direction() const { return dir; }

    __host__ __device__ point3 at(float t) const { return orig + t * dir; }
};

#endif