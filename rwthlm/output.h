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
#include "vocabulary.h"

class Output : public Function {
public:
  Output(const int input_dimension,
         const int max_batch_size,
         const int max_sequence_length,
         const int num_oovs,
         const bool use_bias,
         ConstVocabularyPointer vocabulary,
         ActivationFunctionPointer activation_function);

  virtual ~Output();

  virtual const Real *Evaluate(const Slice &slice, const Real x[]);

  virtual void ComputeDelta(const Slice &slice, FunctionPointer f);

  virtual void AddDelta(const Slice &slice, Real delta_t[]);

  virtual const Real *UpdateWeights(const Slice &slice,
                                    const Real learning_rate,
                                    const Real x[]);

  virtual void UpdateMomentumWeights(const Real momentum);

  virtual void ResetMomentum();

  virtual void Reset(const bool is_dependent);

  virtual void ExtractState(State *state) const {
  }

  virtual void SetState(const State &state, const int i = 0) {
  }

  virtual void RandomizeWeights(Random *random);

  virtual void Read(std::ifstream *input_stream);

  virtual void Write(std::ofstream *output_stream);

  virtual int GetOffset() {
    return (num_classes_ + max_class_size_) * max_batch_size();
  }

  Real ComputeLogProbability(const Slice &slice,
                             const Real x[],
                             const bool verbose,
                             ProbabilitySequenceVector *probabilities);

private:
  friend class GradientTest;

  const int num_classes_,
            num_out_of_shortlist_words_,
            shortlist_size_,
            max_class_size_,
            num_oovs_;

  Real *class_b_,
       *class_b_t_,
       *class_delta_,
       *class_delta_t_,
       *class_weights_,
       *class_bias_,
       *word_b_,
       *word_b_t_,
       *word_delta_,
       *word_delta_t_,
       *word_weights_,
       *word_bias_,
       *momentum_class_weights_,
       *momentum_class_bias_;

  std::vector<int> word_offset_;
  ConstVocabularyPointer vocabulary_;
  const ActivationFunctionPointer activation_function_;
};
