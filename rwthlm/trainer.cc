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
#include <cmath>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem/operations.hpp>
#include "fast.h"
#include "identity.h"
#include "linear.h"
#include "output.h"
#include "sigmoid.h"
#include "softmax.h"
#include "tablelookup.h"
#include "tanh.h"
#include "trainer.h"

const Real Trainer::kAutoInitialLearningRate = 0.05;
const Real Trainer::kMaxRelativeIncrease = 2.0;

namespace bp = boost::posix_time;

Trainer::Trainer(const int max_epoch,
                 const bool shuffle,
                 const bool verbose,
                 const bool is_feedforward,
                 const std::string &net_config,
                 const NetPointer &net,
                 ConstVocabularyPointer vocabulary,
                 DataPointer training_data,
                 DataPointer dev_data,
                 Random *random)
    : net_(std::move(net)),
      training_data_(training_data),
      dev_data_(dev_data),
      net_config_(net_config),
      vocabulary_(vocabulary),
      verbose_(verbose),
      is_feedforward_(is_feedforward),
      random_(random),
      shuffle_(shuffle),
      max_epoch_(max_epoch) {
}

void Trainer::Train(const uint32_t seed) {
  // Note: rwthlm will train forever unless you stop it ...
  std::cout << "Training ..." << std::endl;
  while (max_epoch_ == 0 || net_->epoch() < max_epoch_) {
    Shuffle(seed);
    const bp::ptime time(bp::second_clock::local_time());
    TrainEpoch();
    net_->set_epoch(net_->epoch() + 1);
    std::cout << "epoch " << net_->epoch() << " took " << std::fixed <<
                 std::setprecision(2) << (bp::second_clock::local_time() -
				 time).total_seconds() / 60. << " minutes" << std::endl;
    const Real perplexity = ComputePerplexity(dev_data_);
    std::cout << "development perplexity = " << std::setw(20) << std::fixed <<
                 std::setprecision(15) << perplexity << std::scientific <<
                 ", learning rate = " << net_->learning_rate();
    if (net_->momentum() > 0.0)
      std::cout << ", momentum = " << std::fixed << net_->momentum();
    std::cout << std::endl;
    if (net_->best_perplexity() > perplexity) {
      net_->set_best_perplexity(perplexity);
      net_->Write(net_config_);
    } else {
      const int new_epoch = net_->epoch();
      const Real new_learning_rate = 0.5 * net_->learning_rate();
      net_->Read(net_config_);
      net_->set_learning_rate(new_learning_rate);
      net_->set_epoch(new_epoch);
      net_->Write(net_config_);
    }
  }
}

void Trainer::TrainEpoch() {
  Real log_probability = 0.;
  int64_t num_running_words = 0;
  for (const Batch &batch : *training_data_) {
    net_->Reset(false);
    net_->ResetHistories();
    bp::ptime time;
    if (verbose_)
      time = bp::microsec_clock::local_time();
    if (is_feedforward_)
      TrainBatchFeedforward(batch, &log_probability, &num_running_words);
    else
      TrainBatch(batch, &log_probability, &num_running_words);
    if (verbose_) {
      std::cout << "training perplexity = " << std::fixed <<
                   std::setprecision(2) << exp(-log_probability /
                   num_running_words) << std::endl;
      std::cout << "time = " << std::fixed << std::setprecision(3) <<
                   (bp::microsec_clock::local_time() - time).
                   total_milliseconds() / 1000. << " seconds" << std::endl;
    }
  }
}

void Trainer::TrainBatch(const Batch &batch,
                         Real *log_probability,
                         int64_t *num_running_words) {
  // forward pass
  auto previous_slice(*batch.Begin(0));
  for (auto next_slice : batch) {
    const Real *x = net_->Evaluate(next_slice, Caster(previous_slice).Cast());
    *log_probability += net_->ComputeLogProbability(next_slice, x, false);
    *num_running_words += next_slice.size();
    previous_slice = next_slice;
  }

  // backward pass
  auto slice = batch.End(1);
  do {
    --slice;
    net_->ComputeDelta(*slice, FunctionPointer());
  } while (slice != batch.Begin(1));
  net_->ResetHistories();

  // weight update
  previous_slice = *batch.Begin(0);
  for (auto next_slice : batch) {
    net_->UpdateWeights(next_slice, Caster(previous_slice).Cast());
    previous_slice = next_slice;
  }
  net_->UpdateMomentumWeights();
}

void Trainer::TrainBatchFeedforward(const Batch &batch,
                                    Real *log_probability,
                                    int64_t *num_running_words) {
  // forward pass
  auto previous_slice(*batch.Begin(0));
  for (auto next_slice : batch) {
    net_->Reset(false);
    const Real *x = net_->Evaluate(next_slice, Caster(previous_slice).Cast());
    *log_probability += net_->ComputeLogProbability(next_slice, x, false);
    *num_running_words += next_slice.size();
    net_->ComputeDelta(next_slice, FunctionPointer());
    net_->UpdateWeights(next_slice, Caster(previous_slice).Cast());
    previous_slice = next_slice;
    net_->UpdateMomentumWeights();
  }
}

Real Trainer::AutoAdjustLearningRate(const Real factor,
                                     const int seed,
                                     Real learning_rate) {
  assert(factor != 1.);
  assert(kMinNumDecreases < kMaxNumBatches);
  Real ppl, candidate_learning_rate = -1.;
  while (true) {
    ppl = std::numeric_limits<Real>::max();
    int num_batches = 0, num_decreases = 0;
    int64_t num_running_words = 0;
    Real log_probability = 0.;
    learning_rate *= factor;
    net_->set_learning_rate(learning_rate);
    std::cout << std::scientific << std::setprecision(2) << learning_rate <<
                 ':' << std::endl;
    for (const Batch &batch : *training_data_) {
      net_->Reset(false);
      net_->ResetHistories();
      if (is_feedforward_)
        TrainBatchFeedforward(batch, &log_probability, &num_running_words);
      else
        TrainBatch(batch, &log_probability, &num_running_words);
      const Real new_ppl = exp(-log_probability / num_running_words);
      // keep track of whether error falls consistently
      if (new_ppl < ppl)
        ++num_decreases;
      // reject learning rate in case of wild oscillations
      if (new_ppl / ppl > kMaxRelativeIncrease)
        num_decreases = -kMaxNumBatches;
      ppl = new_ppl;
      std::cout << "  " << std::fixed << std::setw(10) <<
                   std::setprecision(2) << ppl << std::endl;
      std::cout.flush();
      ++num_batches;
      if (!IsFiniteNumber(ppl))
        break;
      if (kMaxNumBatches - num_batches < kMinNumDecreases - num_decreases)
        break;
      // no infinity and enough decreases? -> candidate found!
      if (num_batches == kMaxNumBatches) {
        assert(num_decreases >= kMinNumDecreases);
        candidate_learning_rate = learning_rate;
        break;
      }
    }
    net_->ResetMomentum();
    random_->Reset(seed);
    net_->RandomizeWeights(random_);
    if (factor > 1. && candidate_learning_rate != learning_rate)
      break;
    if (factor < 1. && candidate_learning_rate > 0.)
      break;
  }
  return candidate_learning_rate;
}

void Trainer::AutoInitializeLearningRate(const int seed) {
  std::cout << "Determining initial learning rate ..." << std::endl;
  assert(training_data_->GetNumBatches() >= kMaxNumBatches);
  Shuffle(seed);

  // increase learning rate until strong perplexity increase/fluctuation
  std::cout << "Increasing ..." << std::endl;
  Real learning_rate = AutoAdjustLearningRate(2.,
                                              seed,
                                              kAutoInitialLearningRate);
  if (learning_rate < 0.) {
    // decrease learning rate until perplexity drops satisfactorily
    std::cout << "Decreasing ..." << std::endl;
    learning_rate = AutoAdjustLearningRate(0.5,
                                           seed,
                                           2. * kAutoInitialLearningRate);
  }
  net_->set_learning_rate(learning_rate);
  std::cout << "initial learning rate: " << std::scientific <<
               std::setprecision(2) << learning_rate << std::endl;
  random_->Reset(seed);
  net_->RandomizeWeights(random_);
}

Real Trainer::ComputePerplexity(DataPointer data) {
  int num_running_words = 0;
  Real log_probability = 0.;
  for (auto &batch : *data) {
    net_->Reset(false);
    net_->ResetHistories();
    Sequence slice(*batch.Begin(0));
    for (auto next_slice : batch) {
      if (is_feedforward_)
        net_->Reset(false);
      const Real *x = net_->Evaluate(next_slice, Caster(slice).Cast());
      log_probability += net_->ComputeLogProbability(next_slice, x, verbose_);
      slice = next_slice;
      num_running_words += next_slice.size();
    }
  }
  return exp(-log_probability / num_running_words);
}
