/*
 * Copyright 2014 RWTH Aachen University. All rights reserved.
 *
 * Licensed under the RWTH LM License (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cassert>
#include <cstring>
#include <acml.h>
#include <amdlibm.h>
#include <cblas.h>
#include <algorithm>

typedef double Real;

inline void FastZero(const int size, float x[]) {
  std::fill(x, x + size, 0.0f);
}

inline void FastZero(const int size, double x[]) {
  std::fill(x, x + size, 0.0);
}

inline void FastCopy(const float source[],
                     const int size,
                     float destination[]) {
  memcpy(destination, source, size * sizeof(float));
}

inline void FastCopy(const double source[], 
                     const int size,
                     double destination[]) {
  memcpy(destination, source, size * sizeof(double));
}

inline float FastMax(const float x[], const int size) {
  return *std::max_element(x, x + size);
}

inline double FastMax(const double x[], const int size) {
  return *std::max_element(x, x + size);
}

inline float FastComputeSum(const float x[], const int size) {
  return std::accumulate(x, x + size, 0.0f);
}

inline double FastComputeSum(const double x[], const int size) {
  return std::accumulate(x, x + size, 0.0);
}

inline void FastAddConstant(const float source[],
                            const int size,
                            const float value,
                            float destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [value](const float x) { return x + value; });
}

inline void FastAddConstant(const double source[],
                            const int size,
                            const double value,
                            double destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [value](const double x) { return x + value; });
}

inline void FastSubtractConstant(const float source[],
                                 const int size,
                                 const float value,
                                 float destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [value](const double x) { return x - value; });
}

inline void FastSubtractConstant(const double source[],
                                 const int size,
                                 const double value,
                                 double destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [value](const double x) { return x - value; });
}

inline void FastReverseSubtractConstant(
    const float source[],
    const int size,
    const float value,
    float destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [value](const float x) { return value - x; });
}

inline void FastReverseSubtractConstant(
    const double source[],
    const int size,
    const double value,
    double destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [value](const double x) { return value - x; });
}

inline void FastMultiplyByConstant(const float source[],
                                   const int size,
                                   const float value,
                                   float destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [value](const float x) { return value * x; });
}

inline void FastMultiplyByConstant(const double source[],
                                   const int size,
                                   const double value,
                                   double destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [value](const double x) { return value * x; });
}

inline void FastDivideByConstant(const float source[],
                                 const int size,
                                 const float value,
                                 float destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [value](const float x) { return x / value; });
}

inline void FastDivideByConstant(const double source[],
                                 const int size,
                                 const double value,
                                 double destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [value](const double x) { return x / value; });
}

inline void FastInvert(const float source[],
                       const int size,
                       float destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [](const double x) { return 1.0f / x; });
}

inline void FastInvert(const double source[],
                       const int size,
                       double destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [](const double x) { return 1.0 / x; });
}

inline void FastAdd(const float a[],
                    const int size,
                    const float b[],
                    float c[]) {
  std::transform(a,
                 a + size,
                 b,
                 c,
                 [](const float x, const float y) { return x + y; });
}

inline void FastAdd(const double a[],
                    const int size,
                    const double b[],
                    double c[]) {
  std::transform(a,
                 a + size,
                 b,
                 c,
                 [](const double x, const double y) { return x + y; });
}

inline void FastMultiplyByConstantAdd(const float alpha,
                                      const float a[],
                                      const int size,
                                      float b[]) {
  cblas_saxpy(size, alpha, a, 1, b, 1);
}

inline void FastMultiplyByConstantAdd(const double alpha,
                                      const double a[],
                                      const int size,
                                      double b[]) {
  cblas_daxpy(size, alpha, a, 1, b, 1);
}

inline void FastSub(const float a[],
                    const int size,
                    const float b[],
                    float c[]) {
  std::transform(a,
                 a + size,
                 b,
                 c,
                 [](const float x, const float y) { return x - y; });
}

inline void FastSub(const double a[],
                    const int size,
                    const double b[],
                    double c[]) {
  std::transform(a,
                 a + size,
                 b,
                 c,
                 [](const double x, const double y) { return x - y; });
}

inline void FastMultiply(const float a[],
                         const int size,
                         const float b[],
                         float c[]) {
  std::transform(a,
                 a + size,
                 b,
                 c,
                 [](const float x, const float y) { return x * y; });
}

inline void FastMultiply(const double a[],
                         const int size,
                         const double b[],
                         double c[]) {
  std::transform(a,
                 a + size,
                 b,
                 c,
                 [](const double x, const double y) { return x * y; });
}

inline void FastMultiplyAdd(const float a[],
                            const int size,
                            const float b[],
                            float c[]) {
  for (int i = 0; i < size; ++i)
    c[i] += a[i] * b[i];
}

inline void FastMultiplyAdd(const double a[],
                            const int size,
                            const double b[],
                            double c[]) {
  for (int i = 0; i < size; ++i)
    c[i] += a[i] * b[i];
}

inline void FastTanh(const float source[],
                     const int size,
                     float destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [](const float x) { return amd_tanhf(x); });
}

inline void FastTanh(const double source[],
                     const int size,
                     double destination[]) {
  std::transform(source,
                 source + size,
                 destination,
                 [](const double x) { return amd_tanh(x); });
}

inline void FastExponential(const float source[],
                            const int size,
                            float destination[]) {
  amd_vrsa_expf(size, const_cast<float *>(source), destination);
}

inline void FastExponential(const double source[],
                            const int size,
                            double destination[]) {
  amd_vrda_exp(size, const_cast<double *>(source), destination);
}

inline void FastMatrixVectorMultiply(const float a[],
                                     const bool transpose_a,
                                     const int rows_a,
                                     const int columns_a,
                                     const float x[],
                                     float y[]) {
  cblas_sgemv(CblasColMajor,
              transpose_a ? CblasTrans : CblasNoTrans,
              rows_a,
              columns_a,
              1.0f,
              a,
              rows_a,
              x,
              1,
              1.0f,
              y,
              1);
}

inline void FastMatrixVectorMultiply(const double a[],
                                     const bool transpose_a,
                                     const int rows_a,
                                     const int columns_a,
                                     const double x[],
                                     double y[]) {
  cblas_dgemv(CblasColMajor,
              transpose_a ? CblasTrans : CblasNoTrans,
              rows_a,
              columns_a,
              1.0,
              a,
              rows_a,
              x,
              1,
              1.0,
              y,
              1);
}

inline void FastOuterProduct(const float alpha,
                             const float x[],
                             const int size_x,
                             const float y[],
                             const int size_y,
                             float a[]) {
  cblas_sger(CblasColMajor,
             size_x,
             size_y,
             alpha,
             x,
             1,
             y,
             1,
             a,
             size_x);
}

inline void FastOuterProduct(const double alpha,
                             const double x[],
                             const int size_x,
                             const double y[],
                             const int size_y,
                             double a[]) {
  cblas_dger(CblasColMajor,
             size_x,
             size_y,
             alpha,
             x,
             1,
             y,
             1,
             a,
             size_x);
}

inline void FastMatrixMatrixMultiply(
      const float alpha,
      const float a[],
      const bool transpose_a,
      const int rows_a,
      const int columns_a,
      const float b[],
      const bool transpose_b,
      const int columns_b,
      float c[]) {
  cblas_sgemm(CblasColMajor,
              transpose_a ? CblasTrans : CblasNoTrans,
              transpose_b ? CblasTrans : CblasNoTrans,
              rows_a,
              columns_b,
              columns_a,
              alpha,
              a,
              transpose_a ? columns_a : rows_a,
              b,
              transpose_b ? columns_b : columns_a,
              1.0f,
              c,
              rows_a);
}

inline void FastMatrixMatrixMultiply(
      const double alpha,
      const double a[],
      const bool transpose_a,
      const int rows_a,  // m
      const int columns_a,  // k
      const double b[],
      const bool transpose_b,
      const int columns_b,  // n
      double c[]) {
  cblas_dgemm(CblasColMajor,
              transpose_a ? CblasTrans : CblasNoTrans,
              transpose_b ? CblasTrans : CblasNoTrans,
              rows_a,  // m
              columns_b,  // n
              columns_a,  // k
              alpha,
              a,
              transpose_a ? columns_a : rows_a,
              b,
              transpose_b ? columns_b : columns_a,
              1.0,
              c,
              rows_a);
}

inline Real *FastMalloc(const int size) {
  Real *result = new Real[size];
  assert(result != nullptr || size == 0);
  return result;
}

inline void FastFree(Real *x) {
  delete [] x;
}

inline double FastGetTime() {
  return dsecnd();
}
