#ifndef HITTABLE_LIST_HPP
#define HITTABLE_LIST_HPP


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../vector/cude_vector.cuh"
#include "hittable.cuh"


class hittable_list : public hittable {
public:
    uint32_t size = 256;
    uint32_t idx = 0;
    hittable **list = nullptr;

public:
    __device__ bool vector_enlarge() {
        cudaError_t err = cudaSuccess;
        size *= 2;
        hittable **newlist = nullptr;
        err = cudaMalloc((void **) &newlist, size * sizeof(hittable *));
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
            if (!vector_enlarge())
                return false;
        }
        list[idx++] = p;
        return true;
    }

    __device__ hittable *operator[](uint32_t i) {
        if (i < idx)
            return list[i];
        else return *list;
    }

    __device__ hittable *get(uint32_t i) {
        if (i < idx)
            return list[i];
        else return *list;
    }

    __device__ ~hittable_list() {
        cudaFree(list);
    }

    __device__ bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const override {
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

};

__device__ void hittable_init(hittable_list *hlist) {
    hlist->size = 256;
    hlist->idx = 0;
    hlist->list = nullptr;
    hlist->list = (hittable **) malloc(hlist->size * sizeof(hittable *));
}

//__global__ void hittable_list_init(cuda_vector *list) {
//    list = new cuda_vector();
//    list->push_back(nullptr);
//}
//
//class hittable_list : public hittable {
//public:
////    cuda_vector host_list;
//    cuda_vector *device_list = nullptr;
//
//public:
//    __host__ hittable_list() {
//        cuda_vector *list = nullptr;
//        cudaMalloc((void**)&list, sizeof(cuda_vector *));
//        std::cout << list << std::endl;
//        hittable_list_init<<<1,1>>>(list);
//        cudaDeviceSynchronize();
//        cudaMemcpy(device_list, list, sizeof(cuda_vector *), cudaMemcpyDeviceToHost);
//        std::cout << device_list << std::endl;
//    }
//
//    __device__ void add(hittable* &object) {
//        device_list->push_back(object);
//    }
//
//    __device__ auto hit(const ray &r, float t_min, float t_max, hit_record &rec) const -> bool override;
//
////    __device__ auto bounding_box(aabb &output_box) const -> bool override;
//};
//
//__device__ auto hittable_list::hit(const ray &r, float t_min, float t_max, hit_record &rec) const -> bool {
//    hit_record temp_rec;
//    bool hit_anything = false;
//    auto closest_so_far = t_max;
//    // 遍历列表，查找与光线交的物体，更新最近的交点和 tmax
//    for (uint32_t i = 0; i < device_list->size; i ++) {
//        if (   ((hittable*)device_list->get(i))->hit(r, t_min, closest_so_far, temp_rec)) {
//            hit_anything = true;
//            closest_so_far = temp_rec.t;
//            rec = temp_rec;
//        }
//    }
//
//    return hit_anything;
//}

//__device__ auto hittable_list::bounding_box(aabb &output_box) const -> bool {
//    // 遍历数组，求整体的 AABB（不断更新）
//    if (objects.empty())
//        return false;
//
//    aabb temp_box;
//    bool first_box = true;
//
//    for (uint32_t i = 0; i < (int) objects.size(); i++) {
//        if (!objects[i]->bounding_box(temp_box))
//            return false;
//        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
//        first_box = false;
//    }
//
//    return true;
//}

#endif