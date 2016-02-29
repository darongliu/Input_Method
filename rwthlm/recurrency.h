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
#include <memory>
#include "fast.h"
#include "function.h"
#include "random.h"

class Recurrency : public Function {
public:
  Recurrency(const int output_dimension,
             const int max_batch_size,
             const int max_sequence_length,
             Real *&b,
             Real *&b_t,
             Real *&delta,
             Real *&delta_t);

  virtual ~Recurrency() {
    FastFree(recurrent_weights_);
    FastFree(momentum_recurrent_weights_);
  }

  virtual const Real *Evaluate(const Slice &slice, const Real b_t[]);

  virtual void ComputeDelta(const Slice &slice, FunctionPointer f);

  virtual void AddDelta(const Slice &slice, Real delta_t[]);

  virtual const Real *UpdateWeights(const Slice &slice,
                                    const Real learning_rate,
                                    const Real x[]);

  virtual void UpdateMomentumWeights(const Real momentum) {
    FastAdd(momentum_recurrent_weights_,
            output_dimension() * output_dimension(),
            recurrent_weights_,
            recurrent_weights_);
    FastMultiplyByConstant(momentum_recurrent_weights_,
                           output_dimension() * output_dimension(),
                           momentum,
                           momentum_recurrent_weights_);
  }

  virtual void ResetMomentum() {
    FastZero(output_dimension() * output_dimension(),
             momentum_recurrent_weights_);
  }

  virtual void Reset(const bool is_dependent) {
  }

  virtual void ExtractState(State *state) const {
  }

  virtual void SetState(const State &state, const int i = 0) {
  }

  virtual void Read(std::ifstream *input_stream);

  virtual void Write(std::ofstream *output_stream);

  virtual void RandomizeWeights(Random *random);

private:
  friend class GradientTest;

  Real *&b_,
       *&b_t_,
       *&delta_,
       *&delta_t_,
       *recurrent_weights_,
       *momentum_recurrent_weights_;
};

typedef std::unique_ptr<Recurrency> RecurrencyPointer;
