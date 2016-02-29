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
#include <boost/functional/hash.hpp>
#include <iomanip>
#include <iostream>
//#include <numeric>  // only for checking normalization
#include "output.h"

Output::Output(const int input_dimension,
               const int max_batch_size,
               const int max_sequence_length,
               const int num_oovs,
               const bool use_bias,
               ConstVocabularyPointer vocabulary,
               ActivationFunctionPointer activation_function)
    : Function(
          input_dimension,
          vocabulary->GetVocabularySize() + vocabulary->GetNumClasses() -
              vocabulary->ComputeShortlistSize(),
          max_batch_size,
          max_sequence_length),
      activation_function_(std::move(activation_function)),
      num_classes_(vocabulary->GetNumClasses()),
      max_class_size_(vocabulary->GetMaxClassSize()),
      num_oovs_(num_oovs),
      num_out_of_shortlist_words_(vocabulary->GetVocabularySize() -
                                  vocabulary->ComputeShortlistSize()),
      shortlist_size_(vocabulary->ComputeShortlistSize()),
      vocabulary_(vocabulary) {
  class_b_ = FastMalloc((num_classes_ + max_class_size_) * max_batch_size *
                        max_sequence_length);
  class_delta_ = FastMalloc((num_classes_ + max_class_size_) * max_batch_size *
                            max_sequence_length);
  class_weights_ = FastMalloc(num_classes_ * input_dimension);
  class_bias_ = use_bias ? FastMalloc(num_classes_) : nullptr;
  momentum_class_weights_ = FastMalloc(num_classes_ * input_dimension);
  momentum_class_bias_ = use_bias ? FastMalloc(num_classes_) : nullptr;

  word_b_ = class_b_ + num_classes_ * max_batch_size;
  word_delta_ = class_delta_ + num_classes_ * max_batch_size;
  word_weights_ = FastMalloc(num_out_of_shortlist_words_ * input_dimension);
  word_bias_ = use_bias ? FastMalloc(num_out_of_shortlist_words_) : nullptr;

  int sum = 0;
  for (int i = 0; i < num_classes_; ++i) {
    word_offset_.push_back(sum);
    const int class_size = vocabulary_->GetClassSize(i);
    // we do not need weights for shortlist words
    if (class_size != 1)
      sum += class_size;
  }
}

Output::~Output() {
  FastFree(class_b_);
  FastFree(class_delta_);
  FastFree(class_weights_);
  FastFree(class_bias_);
  FastFree(word_weights_);
  FastFree(word_bias_);
  FastFree(momentum_class_weights_);
  FastFree(momentum_class_bias_);
}

const Real *Output::Evaluate(const Slice &slice, const Real x[]) {
  const Real *result = class_b_t_;
  // class part
  if (class_bias_) {
    for (size_t i = 0; i < slice.size(); ++i)
      FastCopy(class_bias_, num_classes_, class_b_t_ + i * num_classes_);
  }
  FastMatrixMatrixMultiply(1.0,
                           class_weights_,
                           false,
                           num_classes_,
                           input_dimension(),
                           x,
                           false,
                           slice.size(),
                           class_b_t_);
  activation_function_->Evaluate(num_classes_, slice.size(), class_b_t_);
  class_b_t_ += GetOffset();

  // word part
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(slice.size()); ++i) {
    const int clazz = vocabulary_->GetClass(slice[i]),
              class_size = vocabulary_->GetClassSize(clazz);
    // shortlist class?
    if (class_size > 1) {
      if (class_bias_) {
        FastCopy(word_bias_ + word_offset_[clazz],
                 class_size,
                 word_b_t_ + i * max_class_size_);
      }
      FastMatrixVectorMultiply(
          word_weights_ + word_offset_[clazz] * input_dimension(),
          false,
          class_size,
          input_dimension(),
          x + i * input_dimension(),
          word_b_t_ + i * max_class_size_);
      activation_function_->Evaluate(class_size, 1,
                                     word_b_t_ + i * max_class_size_);
    }
  }
  word_b_t_ += GetOffset();
  return result;
}

void Output::ComputeDelta(const Slice &slice, FunctionPointer f) {
  class_b_t_ -= GetOffset();
  word_b_t_ -= GetOffset();

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(slice.size()); ++i) {
    const int clazz = vocabulary_->GetClass(slice[i]),
              class_size = vocabulary_->GetClassSize(clazz);
    class_delta_t_[i * num_classes_ + clazz] = 1.;
    if (class_size == 1)
      continue;
    word_delta_t_[i * max_class_size_ + slice[i] - word_offset_[clazz] -
                  shortlist_size_] = 1.;
    activation_function_->MultiplyDerivative(
        class_size,
        1,
        word_b_t_ + i * max_class_size_,
        word_delta_t_ + i * max_class_size_);
  }
  activation_function_->MultiplyDerivative(num_classes_, slice.size(),
                                           class_b_t_, class_delta_t_);
}

void Output::AddDelta(const Slice &slice, Real delta_t[]) {
  // class part
  FastMatrixMatrixMultiply(1.0,
                           class_weights_,
                           true,
                           input_dimension(),
                           num_classes_,
                           class_delta_t_,
                           false,
                           slice.size(),
                           delta_t);
  class_delta_t_ += GetOffset();

  // word part
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(slice.size()); ++i) {
    const int clazz = vocabulary_->GetClass(slice[i]),
              class_size = vocabulary_->GetClassSize(clazz);
    // shortlist class?
    if (class_size == 1)
      continue;
    FastMatrixVectorMultiply(
        word_weights_ + word_offset_[clazz] * input_dimension(),
        true,
        class_size,  // rows of A, not op(A)!
        input_dimension(),
        word_delta_t_ + i * max_class_size_,
        delta_t + i * input_dimension());
  }
  word_delta_t_ += GetOffset();
}

const Real *Output::UpdateWeights(const Slice &slice,
                                  const Real learning_rate,
                                  const Real x[]) {
  const Real *result = class_b_t_;
  // class part
  class_delta_t_ -= GetOffset();
  if (class_bias_) {
    for (size_t i = 0; i < slice.size(); ++i) {
      FastMultiplyByConstantAdd(-learning_rate,
                                class_delta_t_ + i * num_classes_,
                                num_classes_,
                                momentum_class_bias_);
    }
  }
  FastMatrixMatrixMultiply(-learning_rate,
                           class_delta_t_,
                           false,
                           num_classes_,
                           slice.size(),
                           x,
                           true,
                           input_dimension(),
                           momentum_class_weights_);
  class_b_t_ += GetOffset();

  // word part
  word_delta_t_ -= GetOffset();
  for (size_t i = 0; i < slice.size(); ++i) {
    const int clazz = vocabulary_->GetClass(slice[i]),
              class_size = vocabulary_->GetClassSize(clazz);
    if (class_size == 1)
      continue;
    if (class_bias_) {
      FastMultiplyByConstantAdd(-learning_rate,
                                word_delta_t_ + i * max_class_size_,
                                class_size,
                                word_bias_ + word_offset_[clazz]);
    }
    FastOuterProduct(
        -learning_rate,
        word_delta_t_ + i * max_class_size_,
        class_size,
        x + i * input_dimension(),
        input_dimension(),
        word_weights_ + word_offset_[clazz] * input_dimension());
  }
  word_b_t_ += GetOffset();
  return result;
}

void Output::UpdateMomentumWeights(const Real momentum) {
  FastAdd(momentum_class_weights_,
          num_classes_ * input_dimension(),
          class_weights_,
          class_weights_);
  FastMultiplyByConstant(momentum_class_weights_,
                         num_classes_ * input_dimension(),
                         momentum,
                         momentum_class_weights_);
  if (class_bias_) {
    FastAdd(momentum_class_bias_,
            num_classes_,
            class_bias_,
            class_bias_);
    FastMultiplyByConstant(momentum_class_bias_,
                           num_classes_,
                           momentum,
                           momentum_class_bias_);
  }
}

void Output::ResetMomentum() {
  FastZero(num_classes_ * input_dimension(), momentum_class_weights_);
  FastZero(num_classes_, momentum_class_bias_);
}

void Output::Reset(const bool is_dependent) {
  class_b_t_ = class_b_;
  class_delta_t_ = class_delta_;
  word_b_t_ = word_b_;
  word_delta_t_ = word_delta_;

  FastZero((num_classes_ + max_class_size_) * max_batch_size() *
           max_sequence_length(), class_b_);
  FastZero((num_classes_ + max_class_size_) * max_batch_size() *
           max_sequence_length(), class_delta_);
}

void Output::RandomizeWeights(Random *random) {
//  const Real sigma = 1. / sqrt(input_dimension());
  const Real sigma = 0.1;
  random->ComputeGaussianRandomNumbers(num_classes_ * input_dimension(),
                                       0.,
                                       sigma,
                                       class_weights_);
  random->ComputeGaussianRandomNumbers(
      num_out_of_shortlist_words_ * input_dimension(),
      0.,
      sigma,
      word_weights_);
  FastZero(num_classes_ * input_dimension(), momentum_class_weights_);
  if (class_bias_) {
    random->ComputeGaussianRandomNumbers(num_classes_,
                                         0.,
                                         sigma,
                                         class_bias_);
    random->ComputeGaussianRandomNumbers(num_out_of_shortlist_words_,
                                         0.,
                                         sigma,
                                         word_bias_);
    FastZero(num_classes_, momentum_class_bias_);
  }
}

void Output::Read(std::ifstream *input_stream) {
  input_stream->read(reinterpret_cast<char *>(class_weights_),
                     num_classes_ * input_dimension() * sizeof(Real));
  input_stream->read(reinterpret_cast<char *>(momentum_class_weights_),
                     num_classes_ * input_dimension() * sizeof(Real));
  if (num_out_of_shortlist_words_ > 0) {
    input_stream->read(
        reinterpret_cast<char *>(word_weights_),
        num_out_of_shortlist_words_ * input_dimension() * sizeof(Real));
  }
  if (class_bias_) {
    input_stream->read(reinterpret_cast<char *>(class_bias_),
                       num_classes_ * sizeof(Real));
    input_stream->read(reinterpret_cast<char *>(momentum_class_bias_),
                       num_classes_ * sizeof(Real));
    if (num_out_of_shortlist_words_ > 0) {
      input_stream->read(reinterpret_cast<char *>(word_bias_),
                         num_out_of_shortlist_words_ * sizeof(Real));
    }
  }
}

void Output::Write(std::ofstream *output_stream) {
  output_stream->write(reinterpret_cast<char *>(class_weights_), num_classes_ *
                       input_dimension() * sizeof(Real));
  output_stream->write(reinterpret_cast<char *>(momentum_class_weights_),
                       num_classes_ * input_dimension() * sizeof(Real));
  if (num_out_of_shortlist_words_ > 0) {
    output_stream->write(reinterpret_cast<char *>(
        word_weights_),
        num_out_of_shortlist_words_ * input_dimension() * sizeof(Real));
  }
  if (class_bias_) {
    output_stream->write(reinterpret_cast<char *>(class_bias_),
                         num_classes_ * sizeof(Real));
    output_stream->write(reinterpret_cast<char *>(momentum_class_bias_),
                         num_classes_ * sizeof(Real));
    if (num_out_of_shortlist_words_ > 0) {
      output_stream->write(reinterpret_cast<char *>(word_bias_),
                           num_out_of_shortlist_words_ * sizeof(Real));
    }
  }
}

Real Output::ComputeLogProbability(const Slice &slice,
                                   const Real x[],
                                   const bool verbose,
                                   ProbabilitySequenceVector *probabilities) {
  Real log_probability = 0.;
  if (!slice.empty()) {
    const int offset = num_classes_ * max_batch_size();
    for (size_t i = 0; i < slice.size(); ++i) {
      const int clazz = vocabulary_->GetClass(slice[i]),
                class_size = vocabulary_->GetClassSize(clazz);
      Real probability = x[num_classes_ * i + clazz];
/*
      assert(clazz < num_classes_);
      assert(clazz >= 0);
      const double z = std::accumulate(x + num_classes_ * i, x + num_classes_ * (i + 1), 0.);
      assert(z > 0.99999 && z < 1.00001);
*/
      if (vocabulary_->HasUnk() &&
          slice[i] == vocabulary_->GetIndex(vocabulary_->unk()))
        probability /= num_oovs_ + 1.;  // <unk> may represent multiple words
      log_probability += log(probability);
      if (class_size > 1) {
        const Real word_probability = x[offset + max_class_size_ * i +
                                        slice[i] - word_offset_[clazz] -
                                        shortlist_size_];
/*
        const int start = offset + max_class_size_ * i,
                  word = slice[i] - word_offset_[clazz] - shortlist_size_;
        assert(word >= 0);
        assert(word < max_class_size_);
        const double z = std::accumulate(x + start, x + start + max_class_size_, 0.);
        assert(z > 0.99999 && z < 1.00001);
*/
        probability *= word_probability;
        log_probability += log(word_probability);
      }
      if (probabilities) {
        probabilities->at(i).push_back(probability);
      }
      if (verbose) {
        std::cout << "\tp( " << vocabulary_->GetWord(slice[i]) <<
                     " | ... ) \t = [1gram] " << std::setprecision(8) <<
                     probability << " [ " << std::setprecision(5) <<
                     log(probability) / log(10.) << " ]\n";
      }
    }
  }
  return log_probability;
}
