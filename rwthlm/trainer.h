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
#include <fstream>
#include <memory>
#include <boost/functional/hash.hpp>
#include "data.h"
#include "function.h"
#include "net.h"
#include "random.h"
#include "vocabulary.h"

class Caster {
public:
  Caster(const Slice &slice) : real_slice_(slice.begin(), slice.end()) {
  }

  const Real *Cast() {
    return real_slice_.data();
  }

private:
  const std::vector<Real> real_slice_;
};

class Trainer {
public:
  Trainer(const int max_epoch,
          const bool shuffle,
          const bool verbose,
          const bool is_feedforward,
          const std::string &net_config,
          const NetPointer &net,
          ConstVocabularyPointer vocabulary,
          DataPointer training_data,
          DataPointer dev_data,
          Random *random);

  virtual ~Trainer() {
  }

  void Train(const uint32_t seed);

  void TrainEpoch();

  void AutoInitializeLearningRate(const int seed);

  Real ComputePerplexity(DataPointer data);

private:
  friend class GradientTest;

  static const int kMaxNumBatches = 30, kMinNumDecreases = 25;
  static const Real kAutoInitialLearningRate, kMaxRelativeIncrease;

  static bool IsFiniteNumber(const Real x) {
    // x is not infinity and x is not NaN
    return (x <= std::numeric_limits<Real>::max() &&
            x >= std::numeric_limits<Real>::min()); 
  }

  void Shuffle(const uint32_t seed) {
    if (shuffle_) {
      std::cout << "Shuffling ...";
      size_t epoch_seed = seed;
      boost::hash_combine(epoch_seed, net_->epoch());
      random_->Reset(epoch_seed);
      training_data_->Shuffle(random_);
      std::cout << " done" << std::endl;
    }
  }

  void TrainBatch(const Batch &batch,
                  Real *log_probability,
                  int64_t *num_running_words);

  void TrainBatchFeedforward(const Batch &batch,
                             Real *log_probability,
                             int64_t *num_running_words);

  Real AutoAdjustLearningRate(const Real factor,
                              const int seed,
                              Real learning_rate);

  const bool shuffle_, verbose_, is_feedforward_;
  const int max_epoch_;
  const std::string net_config_;
  const NetPointer &net_;
  const DataPointer training_data_, dev_data_;
  ConstVocabularyPointer vocabulary_;
  Random *random_;
};
