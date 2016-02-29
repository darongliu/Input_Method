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
#include "fast.h"
#include "function.h"
#include "random.h"
#include "recurrency.h"

class TableLookup : public Function {
public:
  TableLookup(const int input_dimension,
              const int output_dimension,
              const int max_batch_size,
              const int max_sequence_length,
              const int order,
              const bool is_recurrent,
              const bool use_bias,
              const bool is_feedforward,
              ActivationFunctionPointer activation_function);

  virtual ~TableLookup() {
    FastFree(b_);
    FastFree(delta_);
    FastFree(weights_);
    FastFree(bias_);
  }

  virtual const Real *Evaluate(const Slice &slice, const Real b_t[]);

  virtual void ComputeDelta(const Slice &slice, FunctionPointer f);

  virtual void AddDelta(const Slice &slice, Real delta_t[]);

  virtual const Real *UpdateWeights(const Slice &slice,
                                    const Real learning_rate,
                                    const Real x[]);

  virtual void UpdateMomentumWeights(const Real momentum);

  virtual void ResetMomentum();

  void UpdateHistories(const size_t size, const Real x[]);

  virtual void ResetHistories() {
    histories_.clear();
  }

  virtual void Reset(const bool is_dependent);

  virtual void ExtractState(State *state) const;

  virtual void SetState(const State &state, const int i = 0);

  virtual void Read(std::ifstream *input_stream);

  virtual void Write(std::ofstream *output_stream);

  virtual void RandomizeWeights(Random *random);

private:
  friend class GradientTest;

  const bool is_feedforward_;
  const size_t order_, word_dimension_;
  std::vector<std::vector<int>> histories_;
  Real *b_, *b_t_, *delta_, *delta_t_, *weights_, *bias_;
  RecurrencyPointer recurrency_;
  const ActivationFunctionPointer activation_function_;
};
