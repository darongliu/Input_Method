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
#include <cmath>
#include <algorithm>
#include "fast.h"
#include "sigmoid.h"

void Sigmoid::Evaluate(const int dimension, const int batch_size,
                       Real b_t[]) const {
#pragma omp parallel for
  for (int i = 0; i < batch_size; ++i) {
    std::transform(b_t + i * dimension,
                   b_t + (i + 1) * dimension,
                   b_t + i * dimension,
                   [](const Real x){ if (x > 0) {
                                       return 1. / (1. + exp(-x));
                                     } else {
                                       const Real expx = exp(x);
                                       return expx / (expx + 1.);
                                     } });
  }
}

void Sigmoid::MultiplyDerivative(const int dimension, const int batch_size,
                                 const Real b_t[], Real delta_t[]) const {
  const int size = dimension * batch_size;
  std::transform(b_t,
                 b_t + dimension * batch_size,
                 delta_t,
                 delta_t,
                 [](const Real x, const Real y) { return x * (1. - x) * y; });
}
