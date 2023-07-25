// #include "vec3f.cuh"

// __host__ __device__ inline vec3f vec3f::operator*(const float &r) const {
//   return {e[0] * r, e[1] * r, e[2] * r};
// }
// __host__ __device__ inline vec3f vec3f::operator/(const float &r) const {
//   return *this * (1 / r);
// }
// __host__ __device__ inline vec3f vec3f::operator+(const vec3f &v) const {
//   return {e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2]};
// }
// __host__ __device__ inline vec3f vec3f::operator-(const vec3f &v) const {
//   return {e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]};
// }
// __host__ __device__ inline vec3f vec3f::operator*(const vec3f &v) const{
//   return {e[0] * v.e[0], e[1] * v.e[1], e[2] * v.e[2]};
// }
// __host__ __device__ inline vec3f& vec3f::operator+=(const vec3f &v){
//   *this = *this + v;
//   return *this;
// }
// __host__ __device__ inline vec3f& vec3f::operator*=(const float &r){
//   *this = *this * r;
//   return *this;
// }
// __host__ __device__ inline vec3f& vec3f::operator/=(const float &r){
//   *this = *this / r;
//   return *this;
// }

// __host__ __device__ inline float vec3f::sum(){
//   return e[0] + e[1] + e[2];
// }
//  __host__ __device__ inline float vec3f::length_squared() const {
//   return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
// }
// __host__ __device__ inline float vec3f::length() const {
//     return std::sqrt(length_squared());
// }
// __host__ __device__ inline void vec3f::sqrt(){
//   e[0] = std::sqrt(e[0]);
//   e[1] = std::sqrt(e[1]);
//   e[2] = std::sqrt(e[2]);
// }

// __host__ __device__ inline bool vec3f::near_zero(){
//   return (std::fabs(e[0]) < esp) && (std::fabs(e[1]) < esp) &&
//   (std::fabs(e[2]) < esp);
// }

// inline std::istream& operator>>(std::istream& is, vec3f& v) {
//     is >> v.e[0] >> v.e[1] >> v.e[2];
//     return is;
// }

// inline std::ostream& operator<<(std::ostream& os, const vec3f& v) {
//     os << v.e[0] << " " << v.e[1] << " " << v.e[2];
//     return os;
// }
// // __host__ __device__ inline vec3f operator*(const float &r, const vec3f &v)
// {
// //   return v * r;
// // }

// __host__ __device__ inline float dot(const vec3f &a, const vec3f &b) {
//   return (a * b).sum();
// }

// __host__ __device__ inline vec3f cross(const vec3f &a, const vec3f &b) {
//   // clang-fomat off
//   return {
//     a.e[1] * b.e[2] - a.e[2] * b.e[1],
//     a.e[2] * b.e[0] - a.e[0] * b.e[2],
//     a.e[0] * b.e[1] - a.e[1] * b.e[0]
//   };
//   // clang-fomat on
// }

// __host__ __device__ inline vec3f unit_vector(vec3f v) {
//     return v / v.length();
// }

// __host__ __device__ inline vec3f reflect(const vec3f &v, const vec3f &n){
//   return v - 2 * dot(v, n) * n;
// }

// __host__ __device__  inline vec3f refract(const vec3f &uv, const vec3f &n,
// float etai_over_etat) {
//   auto cos_theta = std::min(dot(-uv, n), 1.0f);
//   vec3f r_out_perp = etai_over_etat * (uv + cos_theta * n);
//   vec3f r_out_parallel = -std::sqrt(std::abs(1.0 -
//   r_out_perp.length_squared())) * n; return r_out_perp + r_out_parallel;
// }

// // 随机 (x, y, z) \in (0, 1.0)
// inline static auto random_vec3f() -> vec3f {
//   return {random_float(), random_float(), random_float()};
// }
// // 随机 (x, y, z) \in (min, max)
// inline static auto random_vec3f(float min, float max) -> vec3f {
//   auto fun = random_float(min, max);
//   return {fun(), fun(), fun()};
// }

//   // 随机生成一个在单位圆内的点
// inline auto random_in_unit_sphere() -> vec3f {
//   while (true) {
//     auto p = random_vec3f(-1, 1);
//     if (p.length_squared() >= 1)
//       continue;
//     return p;
//   }
// }

// // 生成一个随机方向的单位向量
// inline auto random_unit_vector() -> vec3f {
//   return unit_vector(random_in_unit_sphere());
// }

// // 生成与法线方向同向的单位向量
// inline auto random_in_hemisphere(const vec3f &normal) -> vec3f {
//   // 判断与法线的位置关系
//   vec3f in_unit_sphere = random_in_unit_sphere();
//   if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the
//   normal
//     return in_unit_sphere;
//   else
//     return -in_unit_sphere;
// }

// // 平面单位圆内随机一点（z = 0）；
// auto random_in_unit_disk() -> vec3f {
//   auto fun = random_float(-1, 1);
//   while (true) {
//     auto p = vec3f(fun(), fun(), 0);
//     if (p.length_squared() >= 1)
//       continue;
//     return p;
//   }
// }

// // 线性插值
// inline auto lerp(const vec3f &a, const vec3f &b, const float &t) -> vec3f {
//   return a * (1 - t) + b * t;
// }
