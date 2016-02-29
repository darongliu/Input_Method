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
#include <ipps.h>
#include <mkl.h>

typedef double Real;

inline void FastZero(const int size, float x[]) {
  ippsZero_32f(x, size);
}

inline void FastZero(const int size, double x[]) {
  ippsZero_64f(x, size);
}

inline void FastCopy(const float source[],
                     const int size,
                     float destination[]) {
  ippsCopy_32f(source, destination, size);
}

inline void FastCopy(const double source[], 
                     const int size,
                     double destination[]) {
  ippsCopy_64f(source, destination, size);
}

inline float FastMax(const float x[], const int size) {
  float max;
  ippsMax_32f(x, size, &max);
  return max;
}

inline double FastMax(const double x[], const int size) {
  double max;
  ippsMax_64f(x, size, &max);
  return max;
}

inline float FastComputeSum(const float x[], const int size) {
  float sum;
  ippsSum_32f(x, size, &sum, ippAlgHintAccurate);
  return sum;
}

inline double FastComputeSum(const double x[], const int size) {
  double sum;
  ippsSum_64f(x, size, &sum);
  return sum;
}

inline void FastAddConstant(const float source[],
                            const int size,
                            const float value,
                            float destination[]) {
  ippsAddC_32f(source, value, destination, size);
}

inline void FastAddConstant(const double source[],
                            const int size,
                            const double value,
                            double destination[]) {
  ippsAddC_64f(source, value, destination, size);
}

inline void FastSubtractConstant(const float source[],
                                 const int size,
                                 const float value,
                                 float destination[]) {
  ippsSubC_32f(source, value, destination, size);
}

inline void FastSubtractConstant(const double source[],
                                 const int size,
                                 const double value,
                                 double destination[]) {
  ippsSubC_64f(source, value, destination, size);
}

inline void FastReverseSubtractConstant(
    const float source[],
    const int size,
    const float value,
    float destination[]) {
  ippsSubCRev_32f(source, value, destination, size);
}

inline void FastReverseSubtractConstant(
    const double source[],
    const int size,
    const double value,
    double destination[]) {
  ippsSubCRev_64f(source, value, destination, size);
}

inline void FastMultiplyByConstant(const float source[],
                                   const int size,
                                   const float value,
                                   float destination[]) {
  ippsMulC_32f(source, value, destination, size);
}

inline void FastMultiplyByConstant(const double source[],
                                   const int size,
                                   const double value,
                                   double destination[]) {
  ippsMulC_64f(source, value, destination, size);
}

inline void FastDivideByConstant(const float source[],
                                 const int size,
                                 const float value,
                                 float destination[]) {
  ippsDivC_32f(source, value, destination, size);
}

inline void FastDivideByConstant(const double source[],
                                 const int size,
                                 const double value,
                                 double destination[]) {
  ippsDivC_64f(source, value, destination, size);
}

inline void FastInvert(const float source[],
                       const int size,
                       float destination[]) {
  vsInv(size, source, destination);
}

inline void FastInvert(const double source[],
                       const int size,
                       double destination[]) {
  vdInv(size, source, destination);
}

inline void FastAdd(const float a[],
                    const int size,
                    const float b[],
                    float c[]) {
  vsAdd(size, a, b, c);
}

inline void FastAdd(const double a[],
                    const int size,
                    const double b[],
                    double c[]) {
  vdAdd(size, a, b, c);
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
  vsSub(size, a, b, c);
}

inline void FastSub(const double a[],
                    const int size,
                    const double b[],
                    double c[]) {
  vdSub(size, a, b, c);
}

inline void FastMultiply(const float a[],
                         const int size,
                         const float b[],
                         float c[]) {
  vsMul(size, a, b, c);
}

inline void FastMultiply(const double a[],
                         const int size,
                         const double b[],
                         double c[]) {
  vdMul(size, a, b, c);
}

inline void FastMultiplyAdd(const float a[],
                            const int size,
                            const float b[],
                            float c[]) {
  ippsAddProduct_32f(a, b, c, size);
}

inline void FastMultiplyAdd(const double a[],
                            const int size,
                            const double b[],
                            double c[]) {
  ippsAddProduct_64f(a, b, c, size);
}

inline void FastTanh(const float source[],
                     const int size,
                     float destination[]) {
  vsTanh(size, source, destination);
}

inline void FastTanh(const double source[],
                     const int size,
                     double destination[]) {
  vdTanh(size, source, destination);
}

inline void FastExponential(const float source[],
                            const int size,
                            float destination[]) {
  vsExp(size, source, destination);
}

inline void FastExponential(const double source[],
                            const int size,
                            double destination[]) {
  vdExp(size, source, destination);
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

inline Real *FastMalloc(const size_t size) {
  Real *result = static_cast<Real *>(mkl_malloc(size * sizeof(Real), 64));
  assert(result != nullptr || size == 0);
  return result;
}

inline void FastFree(Real *x) {
  mkl_free(x);
}

inline double FastGetTime() {
  return dsecnd();
}
