#ifndef HITTABLE_LIST_HPP
#define HITTABLE_LIST_HPP

#include "../vector/cude_vector.cuh"
#include "hittable.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class hittable_list : public hittable {
public:
  uint32_t size = 256;
  uint32_t idx = 0;
  hittable **list = nullptr;

public:
  __device__ hittable_list() {
    size = 256;
    idx = 0;
    list = nullptr;
    list = (hittable **)malloc(size * sizeof(hittable *));
  }

  __device__ hittable_list(hittable_list &hlist) {
    size = hlist.size;
    idx = hlist.idx;
    list = hlist.list;
  }

  __device__ bool vector_enlarge() {
    cudaError_t err = cudaSuccess;
    size *= 2;
    hittable **newlist = nullptr;
    err = cudaMalloc((void **)&newlist, size * sizeof(hittable *));
    if (err != cudaSuccess) {
      return false;
    }
    memcpy(newlist, list, idx * sizeof(hittable *));
    free(list);
    list = newlist;
    return true;
  }

  __device__ bool push_back(hittable *p) {
    if (idx == size) {
      if (!vector_enlarge()) {
        printf("Vector Enlarge ERROR!");
        return false;
      }
    }
    list[idx++] = p;
    return true;
  }

  __device__ hittable *operator[](uint32_t i) {
    if (i < idx)
      return list[i];
    else
      return *list;
  }

  __device__ hittable *get(uint32_t i) {
    if (i < idx)
      return list[i];
    else
      return *list;
  }

  __device__ ~hittable_list() { cudaFree(list); }

  __device__ bool hit(const ray &r, float t_min, float t_max,
                      hit_record &rec) const override {
    //        printf("hittablelist!\n");
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;
    // 遍历列表，查找与光线交的物体，更新最近的交点和 tmax
    for (uint32_t i = 0; i < idx; i++) {
      if ((list[i])->hit(r, t_min, closest_so_far, temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }

    return hit_anything;
  }
  __device__ bool bounding_box(aabb &output_box) const override {
    // 遍历数组，求整体的 AABB（不断更新）
    if (idx == 0)
      return false;

    aabb temp_box;
    bool first_box = true;

    for (uint32_t i = 0; i < idx; i++) {
      if (!list[i]->bounding_box(temp_box))
        return false;
      output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
      first_box = false;
    }

    return true;
  }
};

#endif