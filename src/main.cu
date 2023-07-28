#include "external/BMP.cuh"
#include "global.cuh"
#include "cudaGlobal.cuh"
#include "vector/vec3f.cuh"
#include "scene/scene.cuh"
#include "renderer/Renderer.cuh"
#include <iostream>

__global__ void world_init(hittable **world) {
    *world = new hittable_list;
    printf("addr : %llu\n", *world);
    hittable_init((hittable_list*)*world);
    random_balls_world((hittable_list*)*world);
    printf("addr : %llu\n", *world);
}

//__global__ void test(hittable **world){
//    random_balls_world((hittable_list*)*world);
//}


int main(int argc, const char **argv) {
    std::string photoname = "image.bmp";
    if (argc >= 2) {
        photoname = std::string(argv[1]);
    }
    // photo size
    float aspect_ratio = 3.0 / 2.0;
    uint32_t image_width = 1200;

    point3 lookfrom(13, 2, 3), lookat(0, 0, 0);
    color background(0.70, 0.80, 1.00);
    float vfov = 20.0;


    hittable **world = nullptr;
    cudaMalloc((void**)&world, sizeof(hittable**));
    world_init<<<1,1>>>(world);
    cudaDeviceSynchronize();

    // Camera
    vec3f vup(0, 1, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;

    camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus);
    // Renderer
    Renderer renderer(world, aspect_ratio, image_width, cam);
    renderer.set_photo_name(photoname);
    renderer.set_samples_per_pixel(64);
    renderer.set_max_depth(5);
    renderer.render();
    return 0;
}
