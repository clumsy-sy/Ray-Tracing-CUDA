#ifndef GROBAL_HPP
#define GROBAL_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Constants
const float infinity = std::numeric_limits<float>::infinity();
const float PI = 3.1415926535897932385;
const float esp = 1e-8;
const float esp3 = esp * esp * esp;

// swap

__host__ __device__ void swap(float &a, float &b) {
  float tmp = a;
  a = b;
  b = tmp;
}

__host__ __device__ float min(float &a, float &b) { return a < b ? a : b; }

__host__ __device__ float max(float &a, float &b) { return a > b ? a : b; }

// Utility Functions
__host__ __device__ inline auto degrees_to_radians(float degrees) -> float {
  return degrees * PI / 180.0;
}

// 随机数生成
__host__ inline auto random_float() -> float {
  thread_local std::uniform_real_distribution<float> distribution(0.0, 1.0);
  thread_local std::mt19937 generator{std::random_device{}()};
  // thread_local std::mt19937 generator(10085);
  return distribution(generator);
}
__host__ inline auto random_float(float min, float max) -> auto{
  return [min, max]() -> float {
    thread_local std::uniform_real_distribution<float> distribution(min, max);
    thread_local std::mt19937 generator{std::random_device{}()};
    // thread_local std::mt19937 generator(10085);
    return distribution(generator);
  };
}
__host__ inline auto random_int(int min, int max) -> auto{
  return [min, max]() -> int {
    thread_local std::uniform_int_distribution<> distribution(min, max);
    thread_local std::mt19937 generator{std::random_device{}()};
    // thread_local std::mt19937 generator(10085);
    return distribution(generator);
  };
}
// 判断 x 的范围是否在 [min, max] 之间 否则现在边界
__host__ __device__ inline auto clamp(const float &x, const float &min,
                                      const float &max) -> float {
  if (x < min)
    return min;
  if (x > max)
    return max;
  return x;
}

// 求根公式
__host__ __device__ inline auto solveQuadratic(const float &a, const float &b,
                                               const float &c, float &x0,
                                               float &x1) -> bool {
  float discr = b * b - 4 * a * c;
  if (discr < 0)
    return false;
  else if (discr == 0)
    x0 = x1 = -0.5 * b / a;
  else {
    float q =
        (b > 0) ? -0.5 * (b + std::sqrt(discr)) : -0.5 * (b - std::sqrt(discr));
    x0 = q / a;
    x1 = c / q;
  }
  if (x0 > x1)
    swap(x0, x1);
  return true;
}
// b = 2 * h 情况下的求根公式
__host__ __device__ inline auto solveQuadratic_halfb(const float &a,
                                                     const float &half_b,
                                                     const float &c, float &x0,
                                                     float &x1) -> bool {
  float discr = half_b * half_b - a * c;
  if (discr < 0)
    return false;
  else if (discr == 0)
    x0 = x1 = -half_b / a;
  else {
    float q = -half_b + std::sqrt(discr);
    x0 = q / a;
    x1 = c / q;
  }
  if (x0 > x1)
    swap(x0, x1);
  return true;
}
// 进度条（需要加线程锁）
__host__ inline void UpdateProgress(float progress) {
  int barWidth = 100;

  std::cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %\r";
  std::cout.flush();
}
__host__ inline void UpdateProgress(std::int32_t now, std::int32_t total) {
  UpdateProgress(float(now) / total);
}

#endif