#ifndef BVH_CUH
#define BVH_CUH

#include "../cudaGlobal.cuh"
#include "../global.cuh"
#include <curand_kernel.h>

#include "hittablelist.cuh"

__device__ inline bool box_compare(const hittable *a, const hittable *b,
                                   int axis) {
  aabb box_a, box_b;
  // 可优化，用空间换时间
  if (!a->bounding_box(box_a) || !b->bounding_box(box_b))
    printf("No bounding box in bvh_node constructor.\n");

  return box_a.min().e[axis] < box_b.min().e[axis];
}

__device__ inline bool box_x_compare(const hittable *a, const hittable *b) {
  return box_compare(a, b, 0);
}

__device__ inline bool box_y_compare(const hittable *a, const hittable *b) {
  return box_compare(a, b, 1);
}

__device__ inline bool box_z_compare(const hittable *a, const hittable *b) {
  return box_compare(a, b, 2);
}

__device__ void selection_sort(hittable **data, uint32_t left, uint32_t right,
                               uint32_t cmp) {
  for (int i = left; i < right; ++i) {
    hittable *min_val = data[i];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for (int j = i + 1; j < right; ++j) {
      hittable *val_j = data[j];

      if (box_compare(val_j, min_val, cmp)) {
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
    if (i != min_idx) {
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

/*
  BVH 的结点（也是可背光线击中的）
  左右儿子 + 自己的 AABB
*/
class bvh_node : public hittable {
public:
  hittable *left;
  hittable *right;
  aabb box;

public:
  __device__ bvh_node(hittable **src_objects, size_t start, size_t end,
                      aabb **lrbox, curandState &state) {
    hittable **objects =
        src_objects; // Create a modifiable array of the source scene objects
    // 随机一个轴，按这个轴排序
    float rand = curand_uniform(&state);
    uint32_t axis = 0;
    if (rand < 0.333f)
      axis = 0;
    else if (rand < 0.667f)
      axis = 1;
    else
      axis = 2;

    size_t object_span = end - start;
    // 根据当前结点大小分类处理，1 为 叶子结点， 2 单独处理
    if (object_span == 1) {
      left = right = objects[start];
    } else if (object_span == 2) {
      if (box_compare(objects[start], objects[start + 1], axis)) {
        left = objects[start];
        right = objects[start + 1];
      } else {
        left = objects[start + 1];
        right = objects[start];
      }
    } else {
      selection_sort(objects, start, end, axis);
      // 按随机的轴排序后，递归建树
      auto mid = start + object_span / 2;
      left = new bvh_node(objects, start, mid, lrbox, state);
      right = new bvh_node(objects, mid, end, lrbox, state);
    }
    //        aabb box_left, box_right;

    if (!left->bounding_box(*lrbox[0]) || !right->bounding_box(*lrbox[1]))
      printf("No bounding box in bvh_node constructor.\n");

    box = surrounding_box(*lrbox[0], *lrbox[1]);
  }

  __device__ bool hit(const ray &r, float t_min, float t_max,
                      hit_record &rec) const override {
    if (!box.hit(r, t_min, t_max))
      return false;
    // 递归查找光线与 AABB 的交
    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
  }

  __device__ bool bounding_box(aabb &output_box) const override {
    output_box = box;
    return true;
  }
};

#endif