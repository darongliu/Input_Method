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
#include <cassert>
#include <cmath>
#include <boost/filesystem/operations.hpp>
#include "gradienttest.h"

GradientTest::GradientTest(const uint32_t seed,
                           Trainer *const trainer,
                           const bool is_feedforward)
    : seed_(seed),
      significant_digits_(6),
      epsilon_(1e-5),
      trainer_(trainer) {
  if (is_feedforward) {
    for (auto &indices : trainer_->training_data_->data_) {
      assert(indices.size() >= 2);
      indices.resize(2);
    }
  }
  num_running_words_ = trainer_->training_data_->CountNumRunningWords();
}

// N.b. This test will fail for float (which has only six digits of precision)!
void GradientTest::Test() {
  std::cout << "Testing gradient implementation ...\n";
  for (auto f : trainer_->net_->functions_) {
    TableLookup *table_lookup = dynamic_cast<TableLookup *>(f.get());
    if (table_lookup != nullptr) {
      Test(table_lookup);
      continue;
    }
    Linear *linear = dynamic_cast<Linear *>(f.get());
    if (linear != nullptr) {
      Test(linear);
      continue;
    }
    LSTM *lstm = dynamic_cast<LSTM *>(f.get());
    if (lstm != nullptr) {
      Test(lstm);
      continue;
    }
    Output *output = dynamic_cast<Output *>(f.get());
    if (output != nullptr) {
      Test(output);
    }
  }
  std::cout << "\nGradient test SUCCEEDED!\n";
}

void GradientTest::Test(TableLookup *table_lookup) {
  std::cout << "\nTesting \"TableLookup\" ...\n";
  const int word_dimension = table_lookup->word_dimension_;
  TestWeights("weights",
              word_dimension * table_lookup->input_dimension(),
              table_lookup->weights_);
  if (table_lookup->bias_)
    TestWeights("bias", word_dimension, table_lookup->bias_);
  if (table_lookup->recurrency_) {
    const int output_dimension = table_lookup->output_dimension();
    TestWeights("recurrent weights",
                output_dimension * output_dimension,
                table_lookup->recurrency_->recurrent_weights_);
  }
}

void GradientTest::Test(Linear *linear) {
  std::cout << "\nTesting \"Linear\" ...\n";
  const int output_dimension = linear->output_dimension();
  TestWeights("weights",
              output_dimension * linear->input_dimension(),
              linear->weights_);
  if (linear->bias_) {
    TestWeights("bias",
                output_dimension,
                linear->bias_);
  }
  if (linear->recurrency_) {
    TestWeights("recurrent weights",
                output_dimension * output_dimension,
                linear->recurrency_->recurrent_weights_);
  }
}

void GradientTest::Test(LSTM *lstm) {
  std::cout << "\nTesting \"LSTM\" ...\n";
  int size = lstm->output_dimension() * lstm->input_dimension();
  TestWeights("weights", size, lstm->weights_);
  TestWeights("input gate weights", size, lstm->input_gate_weights_);
  TestWeights("forget gate weights", size, lstm->forget_gate_weights_);
  TestWeights("output gate weights", size, lstm->output_gate_weights_);
  size = lstm->output_dimension() * lstm->output_dimension();
  TestWeights("recurrent weights", size, lstm->recurrent_weights_);
  TestWeights("input gate recurrent weights",
              size,
              lstm->input_gate_recurrent_weights_);
  TestWeights("forget gate recurrent weights",
              size,
              lstm->forget_gate_recurrent_weights_);
  TestWeights("output gate recurrent weights",
              size,
              lstm->output_gate_recurrent_weights_);
  size = lstm->output_dimension();
  TestWeights("input gate peephole weights",
              size,
              lstm->input_gate_peephole_weights_);
  TestWeights("forget gate peephole weights",
              size,
              lstm->forget_gate_peephole_weights_);
  TestWeights("output gate peephole weights",
              size,
              lstm->output_gate_peephole_weights_);
  if (lstm->bias_) {
    TestWeights("bias",
                lstm->output_dimension(),
                lstm->bias_);
    TestWeights("input gate bias",
                lstm->output_dimension(),
                lstm->input_gate_bias_);
    TestWeights("forget gate bias",
                lstm->output_dimension(),
                lstm->forget_gate_bias_);
    TestWeights("output gate bias",
                lstm->output_dimension(),
                lstm->output_gate_bias_);
  }
}

void GradientTest::Test(Output *output) {
  std::cout << "\nTesting \"Output\" ...\n";
  const int num_classes = output->num_classes_,
            num_out_of_shortlist_words = output->num_out_of_shortlist_words_,
            input_dimension = output->input_dimension();
  TestWeights("class weights",
              num_classes * input_dimension,
              output->class_weights_);
  TestWeights("word weights",
              num_out_of_shortlist_words * input_dimension,
              output->word_weights_);
  if (output->class_bias_) {
    TestWeights("class bias",
                num_classes,
                output->class_bias_);
    TestWeights("word bias", num_out_of_shortlist_words, output->word_bias_);
  }
}

void GradientTest::ComputeDifferenceQuotient(const int size,
                                             Real weights[],
                                             Real quotient[]) {
  for (int i = 0; i < size; ++i) {
    weights[i] += epsilon_;
    const Real f1 = num_running_words_ * log(trainer_->ComputePerplexity(
        trainer_->training_data_));
    weights[i] -= 2. * epsilon_;
    const Real f0 = num_running_words_ * log(trainer_->ComputePerplexity(
        trainer_->training_data_));
    quotient[i] = (f1 - f0) / (2. * epsilon_);
    weights[i] += epsilon_;
  }
}

void GradientTest::ComputeDerivative(const int size,
                                     const Real weights[],
                                     Real derivative[]) {
  FastCopy(weights, size, derivative);
  trainer_->TrainEpoch();
  FastSub(derivative, size, weights, derivative);
  FastDivideByConstant(derivative,
                       size,
                       trainer_->net_->learning_rate(),
                       derivative);
  trainer_->random_->Reset(seed_);
  if (boost::filesystem::exists(trainer_->net_config_))
    trainer_->net_->Read(trainer_->net_config_);
  else
    trainer_->net_->RandomizeWeights(trainer_->random_);
}

void GradientTest::Compare(const std::string &name,
                           const int size,
                           const Real derivative[],
                           const Real quotient[]) const {
  std::cout << name << ":\n";
  for (int i = 0; i < size; ++i) {
    std::cout << "dF/dw:           " << std::setw(11) << std::fixed <<
                 std::setprecision(7) << derivative[i] <<
                 "\nDelta F/Delta w: " << std::setw(11) << quotient[i] <<
                 '\n';
    const Real threshold = pow(
        10.0,
        std::max(static_cast<Real>(0.0), 
            static_cast<Real>(ceil(log10(std::min(fabs(derivative[i]),
            fabs(quotient[i])))))) - significant_digits_);
    assert(fabs(derivative[i] - quotient[i]) < threshold);
  }
}

void GradientTest::TestWeights(const std::string &name,
                               const int size,
                               Real weights[]) {
  // anything to check?
  if (size == 0)
    return;
  Real *quotient = FastMalloc(size);
  ComputeDifferenceQuotient(size, weights, quotient);
  Real *derivative = FastMalloc(size);
  ComputeDerivative(size, weights, derivative);
  Compare(name, size, derivative, quotient);
  FastFree(quotient);
  FastFree(derivative);
}
