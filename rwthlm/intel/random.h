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
#include <fstream>
#include <mkl_vsl.h>

class Random {
public:
  Random(const uint32_t seed = 1) {
    vslNewStream(&stream_, VSL_BRNG_MT19937, seed);
  }

  virtual ~Random() {
    vslDeleteStream(&stream_);
  }

  void Reset(const uint32_t seed) {
    vslDeleteStream(&stream_);
    vslNewStream(&stream_, VSL_BRNG_MT19937, seed);
  }

  void ComputeGaussianRandomNumbers(const int num_values,
                                    const float mean,
                                    const float standard_deviation,
                                    float *result) {
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF,
                  stream_,
                  num_values,
                  result,
                  mean,
                  standard_deviation);
  }

  void ComputeGaussianRandomNumbers(const int num_values,
                                    const double mean,
                                    const double standard_deviation,
                                    double *result) {
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF,
                  stream_,
                  num_values,
                  result,
                  mean,
                  standard_deviation);
  }

  int operator()(const int n) {
    int result;
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream_, 1, &result, 0, n);
    assert(result >= 0 && result < n);
    return result;
  }

private:
  VSLStreamStatePtr stream_;
};
