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
#include "fast.h"
#include "tablelookup.h"
#include <algorithm>

TableLookup::TableLookup(const int input_dimension,
                         const int output_dimension,
                         const int max_batch_size,
                         const int max_sequence_length,
                         const int order,
                         const bool is_recurrent,
                         const bool use_bias,
                         const bool is_feedforward,
                         ActivationFunctionPointer activation_function)
    : Function(input_dimension,
               output_dimension,
               max_batch_size,
               max_sequence_length),
      order_(order),
      is_feedforward_(is_feedforward),
      word_dimension_(output_dimension / order),
      activation_function_(std::move(activation_function)) {
  assert(order == 0 || output_dimension == word_dimension_ * order);
  b_ = FastMalloc(output_dimension * max_batch_size * max_sequence_length);
  delta_ = FastMalloc(output_dimension * max_batch_size * max_sequence_length);
  weights_ = FastMalloc(word_dimension_ * input_dimension);
  bias_ = use_bias ? FastMalloc(word_dimension_) : nullptr;
  if (is_recurrent) {
    recurrency_ = RecurrencyPointer(new Recurrency(output_dimension,
                                                   max_batch_size,
                                                   max_sequence_length,
                                                   b_,
                                                   b_t_,
                                                   delta_,
                                                   delta_t_));
  }
  ResetHistories();
}

const Real *TableLookup::Evaluate(const Slice &slice, const Real x[]) {
  Real *result = b_t_;
  UpdateHistories(slice.size(), x);
  if (bias_) {
    for (size_t i = 0; i < slice.size() * order_; ++i)
      FastCopy(bias_, word_dimension_, b_t_ + i * word_dimension_);
  }
  
#pragma omp parallel for
  for (int i = 0; i < slice.size() * order_; ++i) {
    const int word = histories_[i / order_][i % order_];
    FastAdd(&weights_[word * word_dimension_],
            word_dimension_,
            b_t_ + i * word_dimension_,
            b_t_ + i * word_dimension_);
  }
  if (recurrency_)
    recurrency_->Evaluate(slice, x);
  activation_function_->Evaluate(output_dimension(), slice.size(), result);
  b_t_ = result + GetOffset();
  return result;
}

void TableLookup::ComputeDelta(const Slice &slice, FunctionPointer f) {
  b_t_ -= GetOffset();
  f->AddDelta(slice, delta_t_);
  if (recurrency_)
    recurrency_->ComputeDelta(slice, f);
  activation_function_->MultiplyDerivative(output_dimension(), slice.size(),
                                           b_t_, delta_t_);
  delta_t_ += GetOffset();
}

void TableLookup::AddDelta(const Slice &slice, Real delta_t[]) {
  // there is no layer prior to the table lookup layer
}

const Real *TableLookup::UpdateWeights(const Slice &slice,
                                       const Real learning_rate,
                                       const Real x[]) {
  delta_t_ -= GetOffset();
  if (!is_feedforward_)
    UpdateHistories(slice.size(), x);
  if (bias_) {
    for (size_t i = 0; i < slice.size() * order_; ++i) {
      FastMultiplyByConstantAdd(-learning_rate,
                                delta_t_ + i * word_dimension_,
                                word_dimension_,
                                bias_);
    }
  }
  for (size_t i = 0; i < slice.size() * order_; ++i) {
    const int word = histories_[i / order_][i % order_];
    FastMultiplyByConstantAdd(
        -learning_rate,
        delta_t_ + i * word_dimension_,
        word_dimension_,
        weights_ + word_dimension_ * word);
  }
  if (recurrency_)
    recurrency_->UpdateWeights(slice, learning_rate, x);
  const Real *result = b_t_;
  b_t_ += GetOffset();
  return result;
}

void TableLookup::UpdateMomentumWeights(const Real momentum) {
  if (recurrency_)
    recurrency_->UpdateMomentumWeights(momentum);
}

void TableLookup::ResetMomentum() {
  if (recurrency_)
    recurrency_->ResetMomentum();
}

void TableLookup::UpdateHistories(const size_t size, const Real x[]) {
  if (histories_.empty()) {
    for (size_t i = 0; i < size; ++i)
      histories_.push_back(std::vector<int>(order_, static_cast<int>(x[i])));
  } else {
    for (size_t i = 0; i < size; ++i) {
      std::vector<int> &history(histories_[i]);
      history.insert(history.begin(), static_cast<int>(x[i]));
      history.pop_back();
    }
  }
}

void TableLookup::RandomizeWeights(Random *random) {
  random->ComputeGaussianRandomNumbers(word_dimension_ * input_dimension(),
                                       0.,
                                       0.1,
//                                       1.,
                                       weights_);
  if (bias_) {
    random->ComputeGaussianRandomNumbers(word_dimension_,
                                         0.,
                                         0.1,
//                                         1.,
                                         bias_);
  }
  if (recurrency_)
    recurrency_->RandomizeWeights(random);
}

void TableLookup::Reset(const bool is_dependent) {
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

void TableLookup::ExtractState(State *state) const {
  std::vector<Real> hidden_layer;
  if (recurrency_)
    hidden_layer.insert(hidden_layer.end(), b_, b_ + GetOffset());
  for (const auto &history : histories_)
    hidden_layer.insert(hidden_layer.end(), history.begin(), history.end());
  state->states.push_back(hidden_layer);
}

void TableLookup::SetState(const State &state, const int i) {
  const auto &s = state.states[i];
  if (recurrency_)
    FastCopy(s.data(), GetOffset(), b_);
  ResetHistories();
  for (int j = recurrency_ ? GetOffset() : 0; j < s.size(); j += order_)
    histories_.push_back(std::vector<int>(s.begin() + j, s.begin() + j + order_));
}

void TableLookup::Read(std::ifstream *input_stream) {
  input_stream->read(
      reinterpret_cast<char *>(weights_),
      word_dimension_ * input_dimension() * sizeof(Real));
  if (bias_) {
    input_stream->read(reinterpret_cast<char *>(bias_),
                       word_dimension_ * sizeof(Real));
  }
  if (recurrency_)
    recurrency_->Read(input_stream);
}

void TableLookup::Write(std::ofstream *output_stream) {
  output_stream->write(
      reinterpret_cast<const char *>(weights_),
      word_dimension_ * input_dimension() * sizeof(Real));
  if (bias_) {
    output_stream->write(
        reinterpret_cast<const char *>(bias_),
        word_dimension_ * sizeof(Real));
  }
  if (recurrency_)
    recurrency_->Write(output_stream);
}
