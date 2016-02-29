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
#include <memory>
#include "fast.h"
#include "linear.h"
#include "recurrency.h"

Linear::Linear(const int input_dimension,
               const int output_dimension,
               const int max_batch_size,
               const int max_sequence_length,
               const bool is_recurrent,
               const bool use_bias,
               ActivationFunctionPointer activation_function)
    : Function(input_dimension,
               output_dimension,
               max_batch_size,
               max_sequence_length),
      activation_function_(std::move(activation_function)){
  b_ = FastMalloc(output_dimension * max_batch_size * max_sequence_length);
  delta_ = FastMalloc(output_dimension * max_batch_size * max_sequence_length);
  weights_ = FastMalloc(output_dimension * input_dimension);
  bias_ = use_bias ? FastMalloc(output_dimension) : nullptr;
  momentum_weights_ = FastMalloc(output_dimension * input_dimension);
  momentum_bias_ = use_bias ? FastMalloc(output_dimension) : nullptr;
  if (is_recurrent) {
    recurrency_ = RecurrencyPointer(new Recurrency(output_dimension,
                                                   max_batch_size,
                                                   max_sequence_length,
                                                   b_,
                                                   b_t_,
                                                   delta_,
                                                   delta_t_));
  }
}

const Real *Linear::Evaluate(const Slice &slice, const Real x[]) {
  if (bias_) {
    for (size_t i = 0; i < slice.size(); ++i)
        FastCopy(bias_, output_dimension(), b_t_ + i * output_dimension());
  }
  FastMatrixMatrixMultiply(1.0,
                           weights_,
                           false,
                           output_dimension(),
                           input_dimension(),
                           x,
                           false,
                           slice.size(),
                           b_t_);
  if (recurrency_)
    recurrency_->Evaluate(slice, x);
  activation_function_->Evaluate(output_dimension(), slice.size(), b_t_);
  const Real *result = b_t_;
  // let b_t_ point to next time step
  b_t_ += GetOffset();
  return result;
}

void Linear::ComputeDelta(const Slice &slice, FunctionPointer f) {
  // let b_t_ point to current time step
  b_t_ -= GetOffset();
  // delta_t_ points to current time step
  f->AddDelta(slice, delta_t_);
  if (recurrency_)
    recurrency_->ComputeDelta(slice, f);
  activation_function_->MultiplyDerivative(output_dimension(), slice.size(),
                                           b_t_, delta_t_);
}

void Linear::AddDelta(const Slice &slice, Real *delta_t) {
  // bias delta is zero
  FastMatrixMatrixMultiply(1.0,
                           weights_,
                           true,
                           input_dimension(),
                           output_dimension(),
                           delta_t_,
                           false,
                           slice.size(),
                           delta_t);
  // let delta_t_ point to previous time step
  delta_t_ += GetOffset();
}

const Real *Linear::UpdateWeights(const Slice &slice,
                                  const Real learning_rate,
                                  const Real x[]) {
  delta_t_ -= GetOffset();
  if (bias_) {
    for (size_t i = 0; i < slice.size(); ++i) {
      FastMultiplyByConstantAdd(-learning_rate,
                                delta_t_ + i * output_dimension(),
                                output_dimension(),
                                momentum_bias_);
    }
  }
  FastMatrixMatrixMultiply(-learning_rate,
                           delta_t_,
                           false,
                           output_dimension(),
                           slice.size(),
                           x,
                           true,
                           input_dimension(),
                           momentum_weights_);
  if (recurrency_)
    recurrency_->UpdateWeights(slice, learning_rate, x);
  const Real *result = b_t_;
  // let b_t_ point to next time step
  b_t_ += GetOffset();
  return result;
}

void Linear::UpdateMomentumWeights(const Real momentum) {
  FastAdd(momentum_weights_,
          output_dimension() * input_dimension(),
          weights_,
          weights_);
  FastMultiplyByConstant(momentum_weights_,
                         output_dimension() * input_dimension(),
                         momentum,
                         momentum_weights_);
  if (bias_) {
    FastAdd(momentum_bias_, output_dimension(), bias_, bias_);
    FastMultiplyByConstant(momentum_bias_,
                           output_dimension(),
                           momentum,
                           momentum_bias_);
  }
  if (recurrency_)
    recurrency_->UpdateMomentumWeights(momentum);
}

void Linear::ResetMomentum() {
  FastZero(output_dimension() * input_dimension(), momentum_weights_);
  if (momentum_bias_) {
    FastZero(output_dimension(), momentum_bias_);
  }
}

void Linear::Reset(const bool is_dependent) {
  if (is_dependent && recurrency_) {
    assert(max_batch_size() == 1);
    FastCopy(b_ + GetOffset(), GetOffset(), b_);
    b_t_ = b_ + GetOffset();
  } else {
    b_t_ = b_;
  }
  delta_t_ = delta_;
  const int size = GetOffset() * max_sequence_length();
  FastZero(size - GetOffset(), b_t_);
  FastZero(size, delta_);
}

void Linear::ExtractState(State *state) const {
  if (recurrency_) {
    assert(max_batch_size() == 1);
    std::vector<Real> hidden_layer;
    hidden_layer.insert(hidden_layer.end(), b_, b_ + GetOffset());
    state->states.push_back(hidden_layer);
  } else {
    state->states.push_back(std::vector<Real>());
  }
}

void Linear::SetState(const State &state, const int i) {
  if (recurrency_) {
    FastCopy(state.states[i].data(), GetOffset(), b_);
  } else {
    assert(state.states[i].empty());
  }
}

void Linear::Read(std::ifstream *input_stream) {
  input_stream->read(reinterpret_cast<char *>(weights_),
                     output_dimension() * input_dimension() * sizeof(Real));
  input_stream->read(reinterpret_cast<char *>(momentum_weights_),
                     output_dimension() * input_dimension() * sizeof(Real));
  if (bias_) {
    input_stream->read(reinterpret_cast<char *>(bias_),
                       output_dimension() * sizeof(Real));
    input_stream->read(reinterpret_cast<char *>(momentum_bias_),
                       output_dimension() * sizeof(Real));
  }
  if (recurrency_)
    recurrency_->Read(input_stream);
}

void Linear::Write(std::ofstream *output_stream) {
  output_stream->write(reinterpret_cast<const char *>(weights_),
                       output_dimension() * input_dimension() * sizeof(Real));
  output_stream->write(reinterpret_cast<const char *>(momentum_weights_),
                       output_dimension() * input_dimension() * sizeof(Real));
  if (bias_) {
    output_stream->write(reinterpret_cast<const char *>(bias_),
                         output_dimension() * sizeof(Real));
    output_stream->write(reinterpret_cast<const char *>(momentum_bias_),
                       output_dimension() * sizeof(Real));
  }
  if (recurrency_)
    recurrency_->Write(output_stream);
}

void Linear::RandomizeWeights(Random *random) {
//  const Real sigma = 1. / sqrt(input_dimension());
  const Real sigma = 0.1;
  random->ComputeGaussianRandomNumbers(output_dimension() * input_dimension(),
                                       0.,
                                       sigma,
                                       weights_);
  FastZero(output_dimension() * input_dimension(), momentum_weights_);
  if (bias_) {
    random->ComputeGaussianRandomNumbers(output_dimension(),
                                         0.,
                                         sigma,
                                         bias_);
    FastZero(output_dimension(), momentum_bias_);
  }
  if (recurrency_)
    recurrency_->RandomizeWeights(random);
}
