#ifndef RAY_CUH
#define RAY_CUH

#include "../vector/vec3f.cuh"

class ray {
public:
    point3 orig;
    vec3f dir;

public:
    __device__ ray() {};
    __device__ ray(const point3 &origin, const vec3f &direction)
            : orig(origin), dir(direction) {}

    __device__ point3 origin() const { return orig; }

    __device__ vec3f direction() const { return dir; }

    __device__ point3 at(float t) const { return orig + t * dir; }
};

#endif