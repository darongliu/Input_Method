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
#include <cassert>
#include <algorithm>
#include "fast.h"
#include "tanh.h"

void Tanh::Evaluate(const int dimension, const int batch_size,
                    Real b_t[]) const {
#pragma omp parallel for
  for (int i = 0; i < batch_size; ++i)
    FastTanh(b_t + i * dimension, dimension, b_t + i * dimension);
}

void Tanh::MultiplyDerivative(const int dimension, const int batch_size,
                              const Real b_t[], Real delta_t[]) const {
  std::transform(
      b_t,
      b_t + dimension * batch_size,
      delta_t,
      delta_t,
      [](Real x, Real y){ return (1. - x * x) * y; });
}
