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
#include "fast.h"
#include "recurrency.h"

Recurrency::Recurrency(const int output_dimension,
                       const int max_batch_size,
                       const int max_sequence_length,
                       Real *&b,
                       Real *&b_t,
                       Real *&delta,
                       Real *&delta_t)
    : Function(0, output_dimension, max_batch_size, max_sequence_length),
      b_(b),
      b_t_(b_t),
      delta_(delta),
      delta_t_(delta_t) {
  recurrent_weights_ = FastMalloc(output_dimension * output_dimension);
  momentum_recurrent_weights_ = FastMalloc(output_dimension *
                                           output_dimension);
}

const Real *Recurrency::Evaluate(const Slice &slice, const Real x[]) {
  // b_{-1} = 0
  if (b_t_ != b_) {
    FastMatrixMatrixMultiply(1.0,
                             recurrent_weights_,
                             false,
                             output_dimension(),
                             output_dimension(),
                             b_t_ - GetOffset(),
                             false,
                             slice.size(),
                             b_t_);
  }
  return b_t_;
}

void Recurrency::ComputeDelta(const Slice &slice, FunctionPointer f) {
  // delta_{T+1} = 0
  if (delta_t_ != delta_) {
    // batch_size_t >= batch_size_{t+1}, so delta_{t+1}_ must be filled with 0
    FastMatrixMatrixMultiply(1.0,
                             recurrent_weights_,
                             true,
                             output_dimension(),
                             output_dimension(),
                             delta_t_ - GetOffset(),
                             false,
                             slice.size(),  // smaller batch_size suffices
                             delta_t_);
  }
}

void Recurrency::AddDelta(const Slice &slice, Real delta_t[]) {
}

const Real *Recurrency::UpdateWeights(const Slice &slice,
                                      const Real learning_rate,
                                      const Real x[]) {
  // b_{-1} = 0 
  if (b_t_ != b_) {
    FastMatrixMatrixMultiply(-learning_rate,
                             delta_t_,
                             false,
                             output_dimension(),
                             slice.size(),
                             b_t_ - GetOffset(),
                             true,
                             output_dimension(),
                             momentum_recurrent_weights_);
  }
  return b_t_;
}

void Recurrency::Read(std::ifstream *input_stream) {
  input_stream->read(reinterpret_cast<char *>(recurrent_weights_),
                     output_dimension() * output_dimension() * sizeof(Real));
  input_stream->read(reinterpret_cast<char *>(momentum_recurrent_weights_),
                     output_dimension() * output_dimension() * sizeof(Real));
}

void Recurrency::Write(std::ofstream *output_stream) {
  output_stream->write(reinterpret_cast<const char *>(recurrent_weights_),
                       output_dimension() * output_dimension() * sizeof(Real));
  output_stream->write(
      reinterpret_cast<const char *>(momentum_recurrent_weights_),
      output_dimension() * output_dimension() * sizeof(Real));
}

void Recurrency::RandomizeWeights(Random *random) {
  random->ComputeGaussianRandomNumbers(output_dimension() * output_dimension(),
                                       0.,
                                       0.1,
//                                       1. / sqrt(output_dimension()),
                                       recurrent_weights_);
  FastZero(output_dimension() * output_dimension(), momentum_recurrent_weights_);
}
