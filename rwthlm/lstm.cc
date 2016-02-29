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
#include <algorithm>
#include "lstm.h"
#include "fast.h"
#include "sigmoid.h"
#include "tanh.h"

LSTM::LSTM(const int input_dimension,
           const int output_dimension,
           const int max_batch_size,
           const int max_sequence_length,
           const bool use_bias)
    : Function(input_dimension,
               output_dimension,
               max_batch_size,
               max_sequence_length),
      sigmoid_(),
      tanh_() {
  int size = output_dimension * max_batch_size * max_sequence_length;
  b_ = FastMalloc(size);
  cec_b_ = FastMalloc(size);
  cec_input_b_ = FastMalloc(size);
  input_gate_b_ = FastMalloc(size);
  forget_gate_b_ = FastMalloc(size);
  output_gate_b_ = FastMalloc(size);

  b_t_ = b_;
  cec_input_b_t_ = cec_input_b_;
  cec_b_t_ = cec_b_;
  input_gate_b_t_ = input_gate_b_;
  forget_gate_b_t_ = forget_gate_b_;
  output_gate_b_t_ = output_gate_b_;

  size = output_dimension * max_batch_size * max_sequence_length;
  cec_epsilon_ = FastMalloc(size);
  delta_ = FastMalloc(size);
  input_gate_delta_ = FastMalloc(size);
  forget_gate_delta_ = FastMalloc(size);
  output_gate_delta_ = FastMalloc(size);

  cec_epsilon_t_ = cec_epsilon_;
  delta_t_ = delta_;
  input_gate_delta_t_ = input_gate_delta_;
  forget_gate_delta_t_ = forget_gate_delta_;
  output_gate_delta_t_ = output_gate_delta_;

  size = input_dimension * output_dimension;
  weights_ = FastMalloc(size);
  input_gate_weights_ = FastMalloc(size);
  forget_gate_weights_ = FastMalloc(size);
  output_gate_weights_ = FastMalloc(size);
  momentum_weights_ = FastMalloc(size);
  momentum_input_gate_weights_ = FastMalloc(size);
  momentum_forget_gate_weights_ = FastMalloc(size);
  momentum_output_gate_weights_ = FastMalloc(size);

  size = output_dimension * output_dimension;
  recurrent_weights_ = FastMalloc(size);
  input_gate_recurrent_weights_ = FastMalloc(size);
  forget_gate_recurrent_weights_ = FastMalloc(size);
  output_gate_recurrent_weights_ = FastMalloc(size);
  momentum_recurrent_weights_ = FastMalloc(size);
  momentum_input_gate_recurrent_weights_ = FastMalloc(size);
  momentum_forget_gate_recurrent_weights_ = FastMalloc(size);
  momentum_output_gate_recurrent_weights_ = FastMalloc(size);

  input_gate_peephole_weights_ = FastMalloc(output_dimension);
  forget_gate_peephole_weights_ = FastMalloc(output_dimension);
  output_gate_peephole_weights_ = FastMalloc(output_dimension);
  momentum_input_gate_peephole_weights_ = FastMalloc(output_dimension);
  momentum_forget_gate_peephole_weights_ = FastMalloc(output_dimension);
  momentum_output_gate_peephole_weights_ = FastMalloc(output_dimension);

  bias_ = use_bias ? FastMalloc(output_dimension) : nullptr;
  input_gate_bias_ = use_bias ? FastMalloc(output_dimension) : nullptr;
  forget_gate_bias_ = use_bias ? FastMalloc(output_dimension) : nullptr;
  output_gate_bias_ = use_bias ? FastMalloc(output_dimension) : nullptr;
  momentum_bias_ = use_bias ? FastMalloc(output_dimension) : nullptr;
  momentum_input_gate_bias_ = use_bias ?
      FastMalloc(output_dimension) : nullptr;
  momentum_forget_gate_bias_ = use_bias ?
      FastMalloc(output_dimension) : nullptr;
  momentum_output_gate_bias_ = use_bias ?
      FastMalloc(output_dimension) : nullptr;
}

LSTM::~LSTM() {
  FastFree(b_);
  FastFree(cec_input_b_);
  FastFree(cec_b_);
  FastFree(input_gate_b_);
  FastFree(forget_gate_b_);
  FastFree(output_gate_b_);
  FastFree(cec_epsilon_);
  FastFree(delta_);
  FastFree(input_gate_delta_);
  FastFree(forget_gate_delta_);
  FastFree(output_gate_delta_);
  FastFree(weights_);
  FastFree(recurrent_weights_);
  FastFree(input_gate_weights_);
  FastFree(input_gate_recurrent_weights_);
  FastFree(input_gate_peephole_weights_);
  FastFree(forget_gate_weights_);
  FastFree(forget_gate_recurrent_weights_);
  FastFree(forget_gate_peephole_weights_);
  FastFree(output_gate_weights_);
  FastFree(output_gate_recurrent_weights_);
  FastFree(output_gate_peephole_weights_);
  FastFree(momentum_weights_);
  FastFree(momentum_recurrent_weights_);
  FastFree(momentum_input_gate_weights_);
  FastFree(momentum_input_gate_recurrent_weights_);
  FastFree(momentum_input_gate_peephole_weights_);
  FastFree(momentum_forget_gate_weights_);
  FastFree(momentum_forget_gate_recurrent_weights_);
  FastFree(momentum_forget_gate_peephole_weights_);
  FastFree(momentum_output_gate_weights_);
  FastFree(momentum_output_gate_recurrent_weights_);
  FastFree(momentum_output_gate_peephole_weights_);
  FastFree(bias_);
  FastFree(input_gate_bias_);
  FastFree(forget_gate_bias_);
  FastFree(output_gate_bias_);
  FastFree(momentum_bias_);
  FastFree(momentum_input_gate_bias_);
  FastFree(momentum_forget_gate_bias_);
  FastFree(momentum_output_gate_bias_);
}

const Real *LSTM::Evaluate(const Slice &slice, const Real x[]) {
  const bool start = b_t_ == b_;
#pragma omp parallel sections
{
#pragma omp section
  EvaluateSubUnit(slice.size(),
                  input_gate_weights_,
                  input_gate_bias_,
                  start ? nullptr : input_gate_recurrent_weights_,
                  start ? nullptr : input_gate_peephole_weights_,
                  x,
                  b_t_ - GetOffset(),
                  cec_b_t_ - GetOffset(),
                  input_gate_b_t_,
                  &sigmoid_);
#pragma omp section
  EvaluateSubUnit(slice.size(),
                  forget_gate_weights_,
                  forget_gate_bias_,
                  start ? nullptr : forget_gate_recurrent_weights_,
                  start ? nullptr : forget_gate_peephole_weights_,
                  x,
                  b_t_ - GetOffset(),
                  cec_b_t_ - GetOffset(),
                  forget_gate_b_t_,
                  &sigmoid_);
}
  EvaluateSubUnit(slice.size(),
                  weights_,
                  bias_,
                  start ? nullptr : recurrent_weights_,
                  nullptr,
                  x,
                  b_t_ - GetOffset(),
                  nullptr,
                  cec_input_b_t_,
                  &tanh_);
  const int size = slice.size() * output_dimension();
  FastMultiply(input_gate_b_t_, size, cec_input_b_t_, cec_b_t_);
  if (!start) {
    FastMultiplyAdd(forget_gate_b_t_,
                    size,
                    cec_b_t_ - GetOffset(),
                    cec_b_t_);
  }
  EvaluateSubUnit(slice.size(),
                  output_gate_weights_,
                  output_gate_bias_,
                  start ? nullptr : output_gate_recurrent_weights_,
                  output_gate_peephole_weights_,
                  x,
                  b_t_ - GetOffset(),
                  cec_b_t_,
                  output_gate_b_t_,
                  &sigmoid_);
  FastCopy(cec_b_t_, size, b_t_);
  tanh_.Evaluate(output_dimension(), slice.size(), b_t_);
  FastMultiply(b_t_, size, output_gate_b_t_, b_t_);

  const Real *result = b_t_;
  b_t_ += GetOffset();
  cec_input_b_t_ += GetOffset();
  cec_b_t_ += GetOffset();
  input_gate_b_t_ += GetOffset();
  forget_gate_b_t_ += GetOffset();
  output_gate_b_t_ += GetOffset();
  return result;
}

void LSTM::EvaluateSubUnit(const int batch_size,
                           const Real weights[],
                           const Real bias[],
                           const Real recurrent_weights[],
                           const Real peephole_weights[],
                           const Real x[],
                           const Real recurrent_b_t[],
                           const Real cec_b_t[],
                           Real b_t[],
                           ActivationFunction *activation_function) {
  if (bias) {
    for (int i = 0; i < batch_size; ++i)
      FastCopy(bias, output_dimension(), b_t + i * output_dimension());
  }
  FastMatrixMatrixMultiply(1.0,
                           weights,
                           false,
                           output_dimension(),
                           input_dimension(),
                           x,
                           false,
                           batch_size,
                           b_t);
  if (recurrent_weights) {
    FastMatrixMatrixMultiply(1.0,
                             recurrent_weights,
                             false,
                             output_dimension(),
                             output_dimension(),
                             recurrent_b_t,
                             false,
                             batch_size,
                             b_t);
  }
  if (peephole_weights) {
#pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
      FastMultiplyAdd(peephole_weights,
                      output_dimension(),
                      cec_b_t + i * output_dimension(),
                      b_t + i * output_dimension());
    }
  }
  activation_function->Evaluate(output_dimension(), batch_size, b_t);
}

void LSTM::ComputeDelta(const Slice &slice, FunctionPointer f) {
  b_t_ -= GetOffset();
  cec_input_b_t_ -= GetOffset();
  cec_b_t_ -= GetOffset();
  input_gate_b_t_ -= GetOffset();
  forget_gate_b_t_ -= GetOffset();
  output_gate_b_t_ -= GetOffset();

  // cell outputs
  f->AddDelta(slice, delta_t_);
  if (delta_t_ != delta_) {
    FastMatrixMatrixMultiply(1.0,
                             recurrent_weights_,
                             true,
                             output_dimension(),
                             output_dimension(),
                             delta_t_ - GetOffset(),
                             false,
                             slice.size(),
                             delta_t_);
    FastMatrixMatrixMultiply(1.0,
                             input_gate_recurrent_weights_,
                             true,
                             output_dimension(),
                             output_dimension(),
                             input_gate_delta_t_ - GetOffset(),
                             false,
                             slice.size(),
                             delta_t_);
    FastMatrixMatrixMultiply(1.0,
                             forget_gate_recurrent_weights_,
                             true,
                             output_dimension(),
                             output_dimension(),
                             forget_gate_delta_t_ - GetOffset(),
                             false,
                             slice.size(),
                             delta_t_);
    FastMatrixMatrixMultiply(1.0,
                             output_gate_recurrent_weights_,
                             true,
                             output_dimension(),
                             output_dimension(),
                             output_gate_delta_t_ - GetOffset(),
                             false,
                             slice.size(),
                             delta_t_);
  }

  // output gates, part I
  const int size = slice.size() * output_dimension();
  FastCopy(cec_b_t_, size, output_gate_delta_t_);
  tanh_.Evaluate(output_dimension(), slice.size(), output_gate_delta_t_);

  // states, part I
  FastMultiply(output_gate_b_t_, size, delta_t_, cec_epsilon_t_);
  tanh_.MultiplyDerivative(output_dimension(),
                           slice.size(),
                           output_gate_delta_t_,
                           cec_epsilon_t_);

  // output gates, part II
  FastMultiply(output_gate_delta_t_,
               size,
               delta_t_,
               output_gate_delta_t_);
  sigmoid_.MultiplyDerivative(output_dimension(),
                              slice.size(),
                              output_gate_b_t_,
                              output_gate_delta_t_);

  // states, part II
#pragma omp parallel for
  for (int i = 0; i < (int) slice.size(); ++i) {
    FastMultiplyAdd(output_gate_peephole_weights_,
                    output_dimension(),
                    output_gate_delta_t_ + i * output_dimension(),
                    cec_epsilon_t_ + i * output_dimension());
  }
  if (delta_t_ != delta_) {
    FastMultiplyAdd(forget_gate_b_t_ + GetOffset(),
                    size,
                    cec_epsilon_t_ - GetOffset(),
                    cec_epsilon_t_);
#pragma omp parallel for
    for (int i = 0; i < (int) slice.size(); ++i) {
      FastMultiplyAdd(input_gate_peephole_weights_,
                      output_dimension(),
                      input_gate_delta_t_ - GetOffset() + i * output_dimension(),
                      cec_epsilon_t_ + i * output_dimension());
      FastMultiplyAdd(
          forget_gate_peephole_weights_,
          output_dimension(),
          forget_gate_delta_t_ - GetOffset() + i * output_dimension(),
          cec_epsilon_t_ + i * output_dimension());
    }
  }

  // cells
  FastMultiply(input_gate_b_t_, size, cec_epsilon_t_, delta_t_);
  tanh_.MultiplyDerivative(output_dimension(),
                           slice.size(),
                           cec_input_b_t_,
                           delta_t_);

#pragma omp parallel sections
{
#pragma omp section
{
  // forget gates
  if (b_t_ != b_) {
    FastMultiply(cec_b_t_ - GetOffset(),
                 size,
                 cec_epsilon_t_,
                 forget_gate_delta_t_);
    sigmoid_.MultiplyDerivative(output_dimension(),
                                slice.size(),
                                forget_gate_b_t_,
                                forget_gate_delta_t_);
  }
}
#pragma omp section
{
  // input gates
  FastMultiply(cec_epsilon_t_,
               size,
               cec_input_b_t_,
               input_gate_delta_t_);
  sigmoid_.MultiplyDerivative(output_dimension(),
                              slice.size(),
                              input_gate_b_t_,
                              input_gate_delta_t_);
}
}
}

void LSTM::AddDelta(const Slice &slice, Real delta_t[]) {
  FastMatrixMatrixMultiply(1.0,
                           weights_,
                           true,
                           input_dimension(),
                           output_dimension(),
                           delta_t_,
                           false,
                           slice.size(),
                           delta_t);
  FastMatrixMatrixMultiply(1.0,
                           input_gate_weights_,
                           true,
                           input_dimension(),
                           output_dimension(),
                           input_gate_delta_t_,
                           false,
                           slice.size(),
                           delta_t);
  FastMatrixMatrixMultiply(1.0,
                           forget_gate_weights_,
                           true,
                           input_dimension(),
                           output_dimension(),
                           forget_gate_delta_t_,
                           false,
                           slice.size(),
                           delta_t);
  FastMatrixMatrixMultiply(1.0,
                           output_gate_weights_,
                           true,
                           input_dimension(),
                           output_dimension(),
                           output_gate_delta_t_,
                           false,
                           slice.size(),
                           delta_t);
  cec_epsilon_t_ += GetOffset();
  delta_t_ += GetOffset();
  input_gate_delta_t_ += GetOffset();
  forget_gate_delta_t_ += GetOffset();
  output_gate_delta_t_ += GetOffset();
}

const Real *LSTM::UpdateWeights(const Slice &slice,
                                const Real learning_rate,
                                const Real x[]) {
  const int size = slice.size() * output_dimension();
  cec_epsilon_t_ -= GetOffset();
  delta_t_ -= GetOffset();
  input_gate_delta_t_ -= GetOffset();
  forget_gate_delta_t_ -= GetOffset();
  output_gate_delta_t_ -= GetOffset();
#pragma omp parallel sections
{
#pragma omp section
{
  if (bias_) {
    for (size_t i = 0; i < slice.size(); ++i) {
      FastMultiplyByConstantAdd(-learning_rate,
                                delta_t_ + i * output_dimension(),
                                output_dimension(),
                                momentum_bias_);
    }
  }
}
#pragma omp section
{
  if (input_gate_bias_) {
    for (size_t i = 0; i < slice.size(); ++i) {
      FastMultiplyByConstantAdd(-learning_rate,
                                input_gate_delta_t_ + i * output_dimension(),
                                output_dimension(),
                                momentum_input_gate_bias_);
    }
  }
}
#pragma omp section
{
  if (forget_gate_bias_) {
    for (size_t i = 0; i < slice.size(); ++i) {
      FastMultiplyByConstantAdd(-learning_rate,
                                forget_gate_delta_t_ + i * output_dimension(),
                                output_dimension(),
                                momentum_forget_gate_bias_);
    }
  }
}
#pragma omp section
{
  if (output_gate_bias_) {
    for (size_t i = 0; i < slice.size(); ++i) {
      FastMultiplyByConstantAdd(-learning_rate,
                                output_gate_delta_t_ + i * output_dimension(),
                                output_dimension(),
                                momentum_output_gate_bias_);
    }
  }
}
#pragma omp section
{
  FastMatrixMatrixMultiply(-learning_rate,
                           delta_t_,
                           false,
                           output_dimension(),
                           slice.size(),
                           x,
                           true,
                           input_dimension(),
                           momentum_weights_);
}
#pragma omp section
{
  FastMatrixMatrixMultiply(-learning_rate,
                           input_gate_delta_t_,
                           false,
                           output_dimension(),
                           slice.size(),
                           x,
                           true,
                           input_dimension(),
                           momentum_input_gate_weights_);
}
#pragma omp section
{
  FastMatrixMatrixMultiply(-learning_rate,
                           forget_gate_delta_t_,
                           false,
                           output_dimension(),
                           slice.size(),
                           x,
                           true,
                           input_dimension(),
                           momentum_forget_gate_weights_);
}
#pragma omp section
{
  FastMatrixMatrixMultiply(-learning_rate,
                           output_gate_delta_t_,
                           false,
                           output_dimension(),
                           slice.size(),
                           x,
                           true,
                           input_dimension(),
                           momentum_output_gate_weights_);
}

#pragma omp section
{
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
}
#pragma omp section
{
  if (b_t_ != b_) {
    FastMatrixMatrixMultiply(-learning_rate,
                             input_gate_delta_t_,
                             false,
                             output_dimension(),
                             slice.size(),
                             b_t_ - GetOffset(),
                             true,
                             output_dimension(),
                             momentum_input_gate_recurrent_weights_);
  }
}
#pragma omp section
{
  if (b_t_ != b_) {
    FastMatrixMatrixMultiply(-learning_rate,
                             forget_gate_delta_t_,
                             false,
                             output_dimension(),
                             slice.size(),
                             b_t_ - GetOffset(),
                             true,
                             output_dimension(),
                             momentum_forget_gate_recurrent_weights_);
  }
}
#pragma omp section
{
  if (b_t_ != b_) {
    FastMatrixMatrixMultiply(-learning_rate,
                             output_gate_delta_t_,
                             false,
                             output_dimension(),
                             slice.size(),
                             b_t_ - GetOffset(),
                             true,
                             output_dimension(),
                             momentum_output_gate_recurrent_weights_);
  }
}
}

#pragma omp parallel sections
{
#pragma omp section
{
  if (b_t_ != b_) {
    // destroys ..._gate_delta_t_, but this will not be used later anyway
    FastMultiplyByConstant(input_gate_delta_t_,
                           size,
                           -learning_rate,
                           input_gate_delta_t_);
    for (size_t i = 0; i < slice.size(); ++i) {
      FastMultiplyAdd(input_gate_delta_t_ + i * output_dimension(),
                      output_dimension(),
                      cec_b_t_ - GetOffset() + i * output_dimension(),
                      momentum_input_gate_peephole_weights_);
    }
  }
}
#pragma omp section
{
  if (b_t_ != b_) {
    FastMultiplyByConstant(forget_gate_delta_t_,
                           size,
                           -learning_rate,
                           forget_gate_delta_t_);
    for (size_t i = 0; i < slice.size(); ++i) {
      FastMultiplyAdd(forget_gate_delta_t_ + i * output_dimension(),
                      output_dimension(),
                      cec_b_t_ - GetOffset() + i * output_dimension(),
                      momentum_forget_gate_peephole_weights_);
    }
  }
}
#pragma omp section
{
  FastMultiplyByConstant(output_gate_delta_t_,
                         size,
                         -learning_rate,
                         output_gate_delta_t_);
  for (size_t i = 0; i < slice.size(); ++i) {
    FastMultiplyAdd(output_gate_delta_t_ + i * output_dimension(),
                    output_dimension(),
                    cec_b_t_ + i * output_dimension(),
                    momentum_output_gate_peephole_weights_);
  }
}
}

  const Real *result = b_t_;
  // let b_t_ point to next time step
  b_t_ += GetOffset();
  cec_input_b_t_ += GetOffset();
  cec_b_t_ += GetOffset();
  input_gate_b_t_ += GetOffset();
  forget_gate_b_t_ += GetOffset();
  output_gate_b_t_ += GetOffset();
  return result;
}

void LSTM::UpdateMomentumWeights(const Real momentum) {
  const int size1 = input_dimension() * output_dimension(),
            size2 = output_dimension() * output_dimension();
#pragma omp parallel sections
{
#pragma omp section
{
  if (bias_) {
    FastAdd(momentum_bias_, output_dimension(), bias_, bias_);
    FastMultiplyByConstant(momentum_bias_,
                           output_dimension(),
                           momentum,
                           momentum_bias_);
  }
}
#pragma omp section
{
  if (input_gate_bias_) {
    FastAdd(momentum_input_gate_bias_,
            output_dimension(),
            input_gate_bias_,
            input_gate_bias_);
    FastMultiplyByConstant(momentum_input_gate_bias_,
                           output_dimension(),
                           momentum,
                           momentum_input_gate_bias_);
  }
}
#pragma omp section
{
  if (forget_gate_bias_) {
    FastAdd(momentum_forget_gate_bias_,
            output_dimension(),
            forget_gate_bias_,
            forget_gate_bias_);
    FastMultiplyByConstant(momentum_forget_gate_bias_,
                           output_dimension(),
                           momentum,
                           momentum_forget_gate_bias_);
  }
}
#pragma omp section
{
  if (output_gate_bias_) {
    FastAdd(momentum_output_gate_bias_,
            output_dimension(),
            output_gate_bias_,
            output_gate_bias_);
    FastMultiplyByConstant(momentum_output_gate_bias_,
                           output_dimension(),
                           momentum,
                           momentum_output_gate_bias_);
  }
}
#pragma omp section
{
  FastAdd(momentum_weights_, size1, weights_, weights_);
  FastMultiplyByConstant(momentum_weights_,
                         size1,
                         momentum,
                         momentum_weights_);
}
#pragma omp section
{
  FastAdd(momentum_input_gate_weights_,
          size1,
          input_gate_weights_,
          input_gate_weights_);
  FastMultiplyByConstant(momentum_input_gate_weights_,
                         size1,
                         momentum,
                         momentum_input_gate_weights_);
}
#pragma omp section
{
  FastAdd(momentum_forget_gate_weights_,
          size1,
          forget_gate_weights_,
          forget_gate_weights_);
  FastMultiplyByConstant(momentum_forget_gate_weights_,
                         size1,
                         momentum,
                         momentum_forget_gate_weights_);
}
#pragma omp section
{
  FastAdd(momentum_output_gate_weights_,
          size1,
          output_gate_weights_,
          output_gate_weights_);
  FastMultiplyByConstant(momentum_output_gate_weights_,
                         size1,
                         momentum,
                         momentum_output_gate_weights_);
}
#pragma omp section
{
  FastAdd(momentum_recurrent_weights_,
          size2,
          recurrent_weights_,
          recurrent_weights_);
  FastMultiplyByConstant(momentum_recurrent_weights_,
                         size2,
                         momentum,
                         momentum_recurrent_weights_);
}
#pragma omp section
{
  FastAdd(momentum_input_gate_recurrent_weights_,
          size2,
          input_gate_recurrent_weights_,
          input_gate_recurrent_weights_);
  FastMultiplyByConstant(momentum_input_gate_recurrent_weights_,
                         size2,
                         momentum,
                         momentum_input_gate_recurrent_weights_);
}
#pragma omp section
{
  FastAdd(momentum_forget_gate_recurrent_weights_,
          size2,
          forget_gate_recurrent_weights_,
          forget_gate_recurrent_weights_);
  FastMultiplyByConstant(momentum_forget_gate_recurrent_weights_,
                         size2,
                         momentum,
                         momentum_forget_gate_recurrent_weights_);
}
#pragma omp section
{
  FastAdd(momentum_output_gate_recurrent_weights_,
          size2,
          output_gate_recurrent_weights_,
          output_gate_recurrent_weights_);
  FastMultiplyByConstant(momentum_output_gate_recurrent_weights_,
                         size2,
                         momentum,
                         momentum_output_gate_recurrent_weights_);
}
#pragma omp section
{
  FastAdd(momentum_input_gate_peephole_weights_,
          output_dimension(),
          input_gate_peephole_weights_,
          input_gate_peephole_weights_);
  FastMultiplyByConstant(momentum_input_gate_peephole_weights_,
                         output_dimension(),
                         momentum,
                         momentum_input_gate_peephole_weights_);
}
#pragma omp section
{
  FastAdd(momentum_forget_gate_peephole_weights_,
          output_dimension(),
          forget_gate_peephole_weights_,
          forget_gate_peephole_weights_);
  FastMultiplyByConstant(momentum_forget_gate_peephole_weights_,
                         output_dimension(),
                         momentum,
                         momentum_forget_gate_peephole_weights_);
}
#pragma omp section
{
  FastAdd(momentum_output_gate_peephole_weights_,
          output_dimension(),
          output_gate_peephole_weights_,
          output_gate_peephole_weights_);
  FastMultiplyByConstant(momentum_output_gate_peephole_weights_,
                         output_dimension(),
                         momentum,
                         momentum_output_gate_peephole_weights_);
}
}
}

void LSTM::ResetMomentum() {
  int size = input_dimension() * output_dimension();
  FastZero(size, momentum_weights_);
  FastZero(size, momentum_input_gate_weights_);
  FastZero(size, momentum_forget_gate_weights_);
  FastZero(size, momentum_output_gate_weights_);

  size = output_dimension() * output_dimension();
  FastZero(size, momentum_recurrent_weights_);
  FastZero(size, momentum_input_gate_recurrent_weights_);
  FastZero(size, momentum_forget_gate_recurrent_weights_);
  FastZero(size, momentum_output_gate_recurrent_weights_);

  FastZero(output_dimension(), momentum_input_gate_peephole_weights_);
  FastZero(output_dimension(), momentum_forget_gate_peephole_weights_);
  FastZero(output_dimension(), momentum_output_gate_peephole_weights_);

  if (momentum_bias_) {
    FastZero(output_dimension(), momentum_bias_);
    FastZero(output_dimension(), momentum_input_gate_bias_);
    FastZero(output_dimension(), momentum_forget_gate_bias_);
    FastZero(output_dimension(), momentum_output_gate_bias_);
  }
}

void LSTM::Reset(const bool is_dependent) {
  if (is_dependent) {
    assert(max_batch_size() == 1);
    FastCopy(b_ + GetOffset(), GetOffset(), b_);
    FastCopy(cec_b_ + GetOffset(), GetOffset(), cec_b_);
    b_t_ = b_ + GetOffset();
    cec_b_t_ = cec_b_ + GetOffset();
  } else {
    b_t_ = b_;
    cec_b_t_ = cec_b_;
  }

  cec_input_b_t_ = cec_input_b_;
  input_gate_b_t_ = input_gate_b_;
  forget_gate_b_t_ = forget_gate_b_;
  output_gate_b_t_ = output_gate_b_;

  cec_epsilon_t_ = cec_epsilon_;
  delta_t_ = delta_;
  input_gate_delta_t_ = input_gate_delta_;
  forget_gate_delta_t_ = forget_gate_delta_;
  output_gate_delta_t_ = output_gate_delta_;

  const int size = GetOffset() * max_sequence_length();
  FastZero(size - GetOffset(), b_t_);
  FastZero(size - GetOffset(), cec_b_t_);

  FastZero(size, cec_input_b_);
  FastZero(size, input_gate_b_);
  FastZero(size, forget_gate_b_);
  FastZero(size, output_gate_b_);

  FastZero(size, cec_epsilon_);
  FastZero(size, delta_);
  FastZero(size, input_gate_delta_);
  FastZero(size, forget_gate_delta_);
  FastZero(size, output_gate_delta_);
}

void LSTM::ExtractState(State *state) const {
  std::vector<Real> hidden_layers;
  hidden_layers.insert(hidden_layers.end(), b_, b_ + GetOffset());
  hidden_layers.insert(hidden_layers.end(), cec_b_, cec_b_ + GetOffset());
  state->states.push_back(hidden_layers);
}

void LSTM::SetState(const State &state, const int i) {
  FastCopy(state.states[i].data(), GetOffset(), b_);
  FastCopy(state.states[i].data() + GetOffset(), GetOffset(), cec_b_);
}

void LSTM::RandomizeWeights(Random *random) {
//  const Real sigma = 1. / sqrt(input_dimension());
  const Real sigma = 0.1;
  int size = output_dimension() * input_dimension();
  random->ComputeGaussianRandomNumbers(size, 0., sigma, weights_);
  random->ComputeGaussianRandomNumbers(size, 0., sigma, input_gate_weights_);
  random->ComputeGaussianRandomNumbers(size, 0., sigma, forget_gate_weights_);
  random->ComputeGaussianRandomNumbers(size, 0., sigma, output_gate_weights_);
  FastZero(size, momentum_weights_);
  FastZero(size, momentum_input_gate_weights_);
  FastZero(size, momentum_forget_gate_weights_);
  FastZero(size, momentum_output_gate_weights_);

  size = output_dimension() * output_dimension();
  random->ComputeGaussianRandomNumbers(size, 0., sigma, recurrent_weights_);
  random->ComputeGaussianRandomNumbers(size,
                                       0.,
                                       sigma,
                                       input_gate_recurrent_weights_);
  random->ComputeGaussianRandomNumbers(size,
                                       0.,
                                       sigma,
                                       forget_gate_recurrent_weights_);
  random->ComputeGaussianRandomNumbers(size,
                                       0.,
                                       sigma,
                                       output_gate_recurrent_weights_);
  FastZero(size, momentum_recurrent_weights_);
  FastZero(size, momentum_input_gate_recurrent_weights_);
  FastZero(size, momentum_forget_gate_recurrent_weights_);
  FastZero(size, momentum_output_gate_recurrent_weights_);

  size = output_dimension();
  random->ComputeGaussianRandomNumbers(size,
                                       0.,
                                       sigma,
                                       input_gate_peephole_weights_);
  random->ComputeGaussianRandomNumbers(size,
                                       0.,
                                       sigma,
                                       forget_gate_peephole_weights_);
  random->ComputeGaussianRandomNumbers(size,
                                       0.,
                                       sigma,
                                       output_gate_peephole_weights_);
  FastZero(size, momentum_input_gate_peephole_weights_);
  FastZero(size, momentum_forget_gate_peephole_weights_);
  FastZero(size, momentum_output_gate_peephole_weights_);
  if (bias_) {
    random->ComputeGaussianRandomNumbers(size,
                                         0.,
                                         sigma,
                                         bias_);
    random->ComputeGaussianRandomNumbers(size,
                                         0.,
                                         sigma,
                                         input_gate_bias_);
    random->ComputeGaussianRandomNumbers(size,
                                         0.,
                                         sigma,
                                         forget_gate_bias_);
    random->ComputeGaussianRandomNumbers(size,
                                         0.,
                                         sigma,
                                         output_gate_bias_);
    FastZero(size, momentum_bias_);
    FastZero(size, momentum_input_gate_bias_);
    FastZero(size, momentum_forget_gate_bias_);
    FastZero(size, momentum_output_gate_bias_);
  }
}

void LSTM::Read(std::ifstream *input_stream) {
  int size = output_dimension() * input_dimension() * sizeof(Real);
  input_stream->read(reinterpret_cast<char *>(weights_), size);
  input_stream->read(reinterpret_cast<char *>(input_gate_weights_), size);
  input_stream->read(reinterpret_cast<char *>(forget_gate_weights_), size);
  input_stream->read(reinterpret_cast<char *>(output_gate_weights_), size);
  input_stream->read(reinterpret_cast<char *>(momentum_weights_), size);
  input_stream->read(reinterpret_cast<char *>(momentum_input_gate_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(momentum_forget_gate_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(momentum_output_gate_weights_),
                     size);

  size = output_dimension() * output_dimension() * sizeof(Real);
  input_stream->read(reinterpret_cast<char *>(recurrent_weights_), size);
  input_stream->read(reinterpret_cast<char *>(input_gate_recurrent_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(forget_gate_recurrent_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(output_gate_recurrent_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(momentum_recurrent_weights_), size);
  input_stream->read(reinterpret_cast<char *>(momentum_input_gate_recurrent_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(momentum_forget_gate_recurrent_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(momentum_output_gate_recurrent_weights_),
                     size);

  size = output_dimension() * sizeof(Real);
  input_stream->read(reinterpret_cast<char *>(input_gate_peephole_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(forget_gate_peephole_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(output_gate_peephole_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(momentum_input_gate_peephole_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(momentum_forget_gate_peephole_weights_),
                     size);
  input_stream->read(reinterpret_cast<char *>(momentum_output_gate_peephole_weights_),
                     size);
  if (bias_) {
    input_stream->read(reinterpret_cast<char *>(bias_),
                       size);
    input_stream->read(reinterpret_cast<char *>(input_gate_bias_),
                       size);
    input_stream->read(reinterpret_cast<char *>(forget_gate_bias_),
                       size);
    input_stream->read(reinterpret_cast<char *>(output_gate_bias_),
                       size);
    input_stream->read(reinterpret_cast<char *>(momentum_bias_),
                       size);
    input_stream->read(reinterpret_cast<char *>(momentum_input_gate_bias_),
                       size);
    input_stream->read(reinterpret_cast<char *>(momentum_forget_gate_bias_),
                       size);
    input_stream->read(reinterpret_cast<char *>(momentum_output_gate_bias_),
                       size);
  }
}

void LSTM::Write(std::ofstream *output_stream) {
  int size = output_dimension() * input_dimension() * sizeof(Real);
  output_stream->write(reinterpret_cast<char *>(weights_), size);
  output_stream->write(reinterpret_cast<char *>(input_gate_weights_), size);
  output_stream->write(reinterpret_cast<char *>(forget_gate_weights_), size);
  output_stream->write(reinterpret_cast<char *>(output_gate_weights_), size);
  output_stream->write(reinterpret_cast<char *>(momentum_weights_), size);
  output_stream->write(reinterpret_cast<char *>(momentum_input_gate_weights_),
                                                size);
  output_stream->write(reinterpret_cast<char *>(momentum_forget_gate_weights_),
                                                size);
  output_stream->write(reinterpret_cast<char *>(momentum_output_gate_weights_),
                                                size);
  size = output_dimension() * output_dimension() * sizeof(Real);
  output_stream->write(reinterpret_cast<char *>(recurrent_weights_), size);
  output_stream->write(reinterpret_cast<char *>(input_gate_recurrent_weights_),
                       size);
  output_stream->write(reinterpret_cast<char *>(forget_gate_recurrent_weights_),
                       size);
  output_stream->write(reinterpret_cast<char *>(output_gate_recurrent_weights_),
                       size);
  output_stream->write(reinterpret_cast<char *>(momentum_recurrent_weights_),
                       size);
  output_stream->write(
      reinterpret_cast<char *>(momentum_input_gate_recurrent_weights_),
      size);
  output_stream->write(
      reinterpret_cast<char *>(momentum_forget_gate_recurrent_weights_),
      size);
  output_stream->write(
      reinterpret_cast<char *>(momentum_output_gate_recurrent_weights_),
      size);
  size = output_dimension() * sizeof(Real);
  output_stream->write(reinterpret_cast<char *>(input_gate_peephole_weights_),
                       size);
  output_stream->write(reinterpret_cast<char *>(forget_gate_peephole_weights_),
                       size);
  output_stream->write(reinterpret_cast<char *>(output_gate_peephole_weights_),
                       size);
  output_stream->write(
      reinterpret_cast<char *>(momentum_input_gate_peephole_weights_),
      size);
  output_stream->write(
      reinterpret_cast<char *>(momentum_forget_gate_peephole_weights_),
      size);
  output_stream->write(
      reinterpret_cast<char *>(momentum_output_gate_peephole_weights_),
      size);
  if (bias_) {
    output_stream->write(reinterpret_cast<char *>(bias_), size);
    output_stream->write(reinterpret_cast<char *>(input_gate_bias_), size);
    output_stream->write(reinterpret_cast<char *>(forget_gate_bias_), size);
    output_stream->write(reinterpret_cast<char *>(output_gate_bias_), size);
    output_stream->write(reinterpret_cast<char *>(momentum_bias_), size);
    output_stream->write(reinterpret_cast<char *>(momentum_input_gate_bias_),
                         size);
    output_stream->write(reinterpret_cast<char *>(momentum_forget_gate_bias_),
                         size);
    output_stream->write(reinterpret_cast<char *>(momentum_output_gate_bias_),
                         size);
  }
}
