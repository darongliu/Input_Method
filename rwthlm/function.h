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
#include <memory>
#include <vector>
#include "fast.h"
#include "random.h"

struct State {
  std::vector<std::vector<Real>> states;
};

class ActivationFunction {
public:
  ActivationFunction() {
  }

  ~ActivationFunction() {
  }

  virtual void Evaluate(const int dimension,
                        const int batch_size,
                        Real b_t[]) const = 0;

  virtual void MultiplyDerivative(const int dimension,
                                  const int batch_size,
                                  const Real b_t[],
                                  Real delta_t[]) const = 0;
};

class Function;
typedef std::shared_ptr<Function> FunctionPointer;
typedef std::unique_ptr<ActivationFunction> ActivationFunctionPointer;
typedef std::vector<int> Slice;
typedef std::vector<Real> ProbabilitySequence;
typedef std::vector<ProbabilitySequence> ProbabilitySequenceVector;

class Function {
public:
  Function(const int input_dimension,
           const int output_dimension,
           const int max_batch_size,
           const int max_sequence_length) 
      : input_dimension_(input_dimension),
        output_dimension_(output_dimension),
        max_batch_size_(max_batch_size),
        max_sequence_length_(max_sequence_length) {
  }

  virtual ~Function() {
  }

  virtual const Real *Evaluate(const Slice &slice, const Real x[]) = 0;
  
  virtual void ComputeDelta(const Slice &slice, FunctionPointer f) = 0;

  virtual const Real *UpdateWeights(const Slice &slice,
                                    const Real learning_rate,
                                    const Real x[]) = 0;

  virtual void UpdateMomentumWeights(const Real momentum) = 0;

  virtual void ResetMomentum() = 0;

  virtual void AddDelta(const Slice &slice, Real delta_t[]) = 0;

  virtual void Reset(const bool is_dependent) = 0;

  virtual void ExtractState(State *state) const = 0;

  virtual void SetState(const State &state, const int i = 0) = 0;

  virtual void RandomizeWeights(Random *random) = 0;

  virtual void Read(std::ifstream *input_stream) = 0;

  virtual void Write(std::ofstream *output_stream) = 0;

  virtual Real ComputeLogProbability(
      const Slice &slice,
      const Real x[],
      const bool verbose,
      ProbabilitySequenceVector *probabilities = nullptr) {
    assert(false);
    return -1.;
  }

  virtual void ResetHistories() {
  }

  int input_dimension() const {
    return input_dimension_;
  }

  int output_dimension() const {
    return output_dimension_;
  }

  int max_batch_size() const {
    return max_batch_size_;
  }

  int max_sequence_length() const {
    return max_sequence_length_;
  }

protected:
  virtual int GetOffset() const {
    return output_dimension() * max_batch_size();
  }

  void set_output_dimension(const int output_dimension) {
    output_dimension_ = output_dimension;
  }

private:
  int output_dimension_;
  const int input_dimension_, max_batch_size_, max_sequence_length_;
};
