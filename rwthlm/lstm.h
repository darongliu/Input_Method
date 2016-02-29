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
#include "function.h"
#include "sigmoid.h"
#include "tanh.h"

class LSTM : public Function {
public:
  LSTM(const int input_dimension,
       const int output_dimension,
       const int max_batch_size,
       const int max_sequence_length,
       const bool use_bias);

  virtual ~LSTM();

  virtual const Real *Evaluate(const Slice &slice, const Real x[]);

  virtual void ComputeDelta(const Slice &slice, FunctionPointer f);

  virtual const Real *UpdateWeights(const Slice &slice,
                                    const Real learning_rate,
                                    const Real x[]);

  virtual void UpdateMomentumWeights(const Real momentum);

  virtual void ResetMomentum();

  virtual void AddDelta(const Slice &slice, Real delta_t[]);

  virtual void Reset(const bool is_dependent);

  virtual void ExtractState(State *state) const;

  virtual void SetState(const State &state, const int i = 0);

  virtual void RandomizeWeights(Random *random);

  virtual void Read(std::ifstream *input_stream);

  virtual void Write(std::ofstream *output_stream);

private:
  friend class GradientTest;

  void EvaluateSubUnit(const int batch_size,
                       const Real weights[],
                       const Real bias[],
                       const Real recurrent_weights[],
                       const Real peephole_weights[],
                       const Real x[],
                       const Real recurrent_b_t[],
                       const Real cec_b_t[],
                       Real b_t[],
                       ActivationFunction *activation_function);

  Real *b_,  // activations
       *b_t_,
       *cec_b_,  // s_c^t
       *cec_b_t_,
       *cec_input_b_,  // g(a_c^t)
       *cec_input_b_t_,
       *input_gate_b_,
       *input_gate_b_t_,
       *forget_gate_b_,
       *forget_gate_b_t_,
       *output_gate_b_,
       *output_gate_b_t_,
       *cec_epsilon_,  // deltas + epsilons
       *cec_epsilon_t_,
       *delta_,
       *delta_t_,
       *input_gate_delta_,
       *input_gate_delta_t_,
       *forget_gate_delta_,
       *forget_gate_delta_t_,
       *output_gate_delta_,
       *output_gate_delta_t_,
       *weights_,  // weights
       *recurrent_weights_,
       *input_gate_weights_,
       *input_gate_recurrent_weights_,
       *input_gate_peephole_weights_,
       *forget_gate_weights_,
       *forget_gate_recurrent_weights_,
       *forget_gate_peephole_weights_,
       *output_gate_weights_,
       *output_gate_recurrent_weights_,
       *output_gate_peephole_weights_,
       *momentum_weights_,  // momentum weights
       *momentum_recurrent_weights_,
       *momentum_input_gate_weights_,
       *momentum_input_gate_recurrent_weights_,
       *momentum_input_gate_peephole_weights_,
       *momentum_forget_gate_weights_,
       *momentum_forget_gate_recurrent_weights_,
       *momentum_forget_gate_peephole_weights_,
       *momentum_output_gate_weights_,
       *momentum_output_gate_recurrent_weights_,
       *momentum_output_gate_peephole_weights_,
       *bias_,  // bias
       *input_gate_bias_,
       *forget_gate_bias_,
       *output_gate_bias_,
       *momentum_bias_,  // momentum bias
       *momentum_input_gate_bias_,
       *momentum_forget_gate_bias_,
       *momentum_output_gate_bias_;
  Tanh tanh_;
  Sigmoid sigmoid_;
};
