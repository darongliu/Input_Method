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
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "fast.h"
#include "function.h"
#include "output.h"

class Net : public Function {
public:
  Net(const ConstVocabularyPointer &vocabulary,
      const int max_batch_size,
      const int max_sequence_length,
      const int num_oovs,
      const bool is_feedforward,
      const Real learning_rate,
      const Real momentum,
      Random *random);

  virtual ~Net() {
  }

  virtual const Real *Evaluate(const Slice &slice, const Real x[]);

  virtual void ComputeDelta(const Slice &slice, FunctionPointer f);

  virtual const Real *UpdateWeights(const Slice &slice, const Real x[]);

  virtual const Real *UpdateWeights(const Slice &slice,
                                    const Real learning_rate,
                                    const Real x[]);

  void UpdateMomentumWeights() {
    UpdateMomentumWeights(momentum());
  }

  virtual void UpdateMomentumWeights(const Real momentum) {
    for (auto &f : functions_)
      f->UpdateMomentumWeights(momentum);
  }

  virtual void ResetMomentum() {
    for (auto &f : functions_)
      f->ResetMomentum();
  }

  virtual void AddDelta(const Slice &slice, Real delta_t[]);

  virtual void Reset(const bool is_dependent);

  virtual void ResetHistories() {
    for (auto &f : functions_)
      f->ResetHistories();
  }

  virtual void ExtractState(State *state) const;

  virtual void SetState(const State &state, const int i = 0);

  virtual void RandomizeWeights(Random *random);

  virtual Real ComputeLogProbability(
      const Slice &slice,
      const Real x[],
      const bool verbose,
      ProbabilitySequenceVector *probabilities = nullptr);

  virtual void Read(std::ifstream *input_stream);

  virtual void Write(std::ofstream *output_stream);

  void Read(const std::string &file_name);

  void Write(const std::string &file_name);

  void Compose(FunctionPointer f) {
    functions_.push_back(f);
    set_output_dimension(f->output_dimension());
  }

  void BuildNetworkAndRandomize(const std::string &net_config,
                                const bool use_bias) {
    BuildNetworkLayers(net_config, use_bias);
    std::cout << "Randomly initializing neural network weights ..." <<
                 std::endl;
    RandomizeWeights(random_);
  }

  void BuildNetworkAndLoad(const std::string &nnlm_file,
                           const bool use_bias) {
    BuildNetworkLayers(nnlm_file, use_bias);
    std::cout << "Reading neural network from file '" << nnlm_file <<
                 "' ..." << std::endl;
    Read(nnlm_file);
    std::cout << "Best development perplexity after " << epoch() <<
                 " epochs: " << std::fixed << std::setprecision(15) <<
                 best_perplexity() << std::endl;
  }

  int epoch() const {
    return epoch_;
  }

  void set_epoch(int epoch) {
    epoch_ = epoch;
  }

  Real learning_rate() const {
    return learning_rate_;
  }

  void set_learning_rate(Real learning_rate) {
    learning_rate_ = learning_rate;
  }

  Real momentum() const {
    return momentum_;
  }

  void set_momentum(const Real momentum) {
    momentum_ = momentum;
  }

  Real best_perplexity() const {
    return best_perplexity_;
  }

  void set_best_perplexity(Real best_perplexity) {
    best_perplexity_ = best_perplexity;
  }

private:
  friend class GradientTest;

  ActivationFunctionPointer SetUpActivationFunction(const char type) const;

  FunctionPointer SetUpFunction(const char type,
                                const int dimension,
                                const bool firstFunction,
                                const bool use_bias,
                                ActivationFunctionPointer g) const;

  void BuildNetworkLayers(const std::string &nnlm_file,
                          const bool use_bias);

  const bool is_feedforward_;
  const int num_oovs_;
  int epoch_;
  Real learning_rate_, momentum_, best_perplexity_;
  std::vector<FunctionPointer> functions_;
  Random *random_;
  const ConstVocabularyPointer &vocabulary_;
};

typedef std::unique_ptr<Net> NetPointer;
