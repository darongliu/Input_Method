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
#include <random>

class Random {
public:
  Random(const uint32_t seed = 1) : engine_(seed) {
  }

  virtual ~Random() {
  }

  void Reset(const uint32_t seed) {
    engine_.seed(seed);
  }

  void Read(std::ifstream *input_stream) {
    *input_stream >> engine_;
  }

  void Write(std::ofstream *output_stream) {
    *output_stream << engine_;
  }

  void Skip(const long long int num) {
    engine_.discard(num);
  }

  void ComputeGaussianRandomNumbers(const int num_values,
                                    const float mean,
                                    const float standard_deviation,
                                    float *result) {
    std::normal_distribution<float> distribution(mean, standard_deviation);
    for (int i = 0; i < num_values; ++i)
      result[i] = distribution(engine_);
  }

  void ComputeGaussianRandomNumbers(const int num_values,
                                    const double mean,
                                    const double standard_deviation,
                                    double *result) {
    std::normal_distribution<double> distribution(mean, standard_deviation);
    for (int i = 0; i < num_values; ++i)
      result[i] = distribution(engine_);
  }

  int operator()(const int n) {
    std::uniform_int_distribution<int> distribution(0, n - 1);
    const int result = distribution(engine_);
    assert(result >= 0 && result < n);
    return result;
  }

private:
  std::mt19937 engine_;
};
