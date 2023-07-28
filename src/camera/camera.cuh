#ifndef CAMERA_HPP
#define CAMERA_HPP

class camera {
public:
  vec3f msg[4]; // 0:origin, 1 : horizontal, 2 : vertical, 3 : lower_left_corner
  vec3f u, v, w; // w:看向方向，u：镜头平面的 x，v：镜头屏幕的 y
  float lens_radius; // 镜头半径

public:
  __host__ __device__ camera(const point3 &lookfrom, const point3 &lookat,
                             const vec3f &vup, float vfov, float aspect_ratio,
                             float aperture, float focus_dist) {
    float theta = degrees_to_radians(vfov);
    float h = tan(theta / 2);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect_ratio * viewport_height;

    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    msg[0] = lookfrom;
    msg[1] = focus_dist * viewport_width * u;
    msg[2] = focus_dist * viewport_height * v;
    msg[3] = msg[0] - msg[1] / 2 - msg[2] / 2 - focus_dist * w;

    lens_radius = aperture / 2;
  }

  __device__ ray get_ray(float s, float t, curandState &state) {
    vec3f rd = lens_radius * random_in_unit_disk(state); // 镜头中随机一个点
    vec3f offset = u * rd.x() + v * rd.y();

    return {msg[0] + offset,
            msg[3] + s * msg[1] + t * msg[2] - msg[0] - offset};
  }
};

#endif