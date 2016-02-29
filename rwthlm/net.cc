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
#include <limits>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem/operations.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/range/adaptor/reversed.hpp>
#include "identity.h"
#include "linear.h"
#include "lstm.h"
#include "net.h"
#include "output.h"
#include "sigmoid.h"
#include "softmax.h"
#include "tablelookup.h"
#include "tanh.h"

Net::Net(const ConstVocabularyPointer &vocabulary,
         const int max_batch_size,
         const int max_sequence_length,
         const int num_oovs,
         const bool is_feedforward,
         const Real learning_rate,
         const Real momentum,
         Random *random)
    : Function(1, 0, max_batch_size, max_sequence_length),
      num_oovs_(num_oovs),
      is_feedforward_(is_feedforward),
      vocabulary_(vocabulary),
      epoch_(0),
      learning_rate_(learning_rate),
      momentum_(momentum),
      best_perplexity_(std::numeric_limits<Real>::max()),
      random_(random) {
}

const Real *Net::Evaluate(const Slice &slice, const Real x[]) {
  for (FunctionPointer f : functions_)
    x = f->Evaluate(slice, x);
  return x;
}

void Net::ComputeDelta(const Slice &slice, FunctionPointer f) {
  for (FunctionPointer g : boost::adaptors::reverse(functions_)) {
    g->ComputeDelta(slice, f);
    f = g;
  }
}

const Real *Net::UpdateWeights(const Slice &slice, const Real x[]) {
  return UpdateWeights(slice, learning_rate(), x);
}

const Real *Net::UpdateWeights(const Slice &slice,
                               const Real learning_rate,
                               const Real x[]) {
  for (FunctionPointer f : functions_)
    x = f->UpdateWeights(slice, learning_rate, x);
  return x;
}

void Net::AddDelta(const Slice &slice, Real delta_t[]) {
  // not needed
}

void Net::Reset(const bool is_dependent) {
  assert(!is_dependent || max_batch_size() == 1);
  for (FunctionPointer f : functions_)
    f->Reset(is_dependent);
}

void Net::ExtractState(State *state) const {
  for (FunctionPointer f : functions_)
    f->ExtractState(state);
}

void Net::SetState(const State &state, const int i) {
  int j = 0;
  for (FunctionPointer f : functions_)
    f->SetState(state, j++);
}

void Net::RandomizeWeights(Random *random) {
  for (FunctionPointer f : functions_)
    f->RandomizeWeights(random);
}

void Net::Read(std::ifstream *input_stream) {
  input_stream->read(reinterpret_cast<char *>(&epoch_),
                     sizeof(int));
  input_stream->read(reinterpret_cast<char *>(&learning_rate_),
                     sizeof(Real));
  input_stream->read(reinterpret_cast<char *>(&best_perplexity_),
                     sizeof(Real));
  for (FunctionPointer f : functions_)
    f->Read(input_stream);
}

void Net::Write(std::ofstream *output_stream) {
  output_stream->write(reinterpret_cast<const char *>(&epoch_),
                       sizeof(int));
  output_stream->write(reinterpret_cast<const char *>(&learning_rate_),
                       sizeof(Real));
  output_stream->write(reinterpret_cast<const char *>(&best_perplexity_),
                       sizeof(Real));
  for (FunctionPointer f : functions_)
    f->Write(output_stream);
}

Real Net::ComputeLogProbability(const std::vector<int> &slice,
                                const Real x[],
                                const bool verbose,
                                ProbabilitySequenceVector *probabilities) {
  return functions_.back()->ComputeLogProbability(slice,
                                                  x,
                                                  verbose,
                                                  probabilities);
}

ActivationFunctionPointer Net::SetUpActivationFunction(const char type) const {
  ActivationFunctionPointer f;
  switch (type) {
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
  case 'i': f = ActivationFunctionPointer(new Identity());
    break;
  case 'l':
  case 'r': f = ActivationFunctionPointer(new Tanh());
    break;
  case 'L':
  case 'R': f = ActivationFunctionPointer(new Sigmoid());
    break;
  case 'm':
  case 'M':
    break;
  case 'x': f = ActivationFunctionPointer(new Softmax());
    break;
  default:
    assert(false);
  }
  return f;
}

FunctionPointer Net::SetUpFunction(const char type,
                                   const int dimension,
                                   const bool firstFunction,
                                   const bool use_bias,
                                   ActivationFunctionPointer g) const {
  FunctionPointer f;
  switch (type) {
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
  case 'i':
  case 'l':
  case 'L':
  case 'r':
  case 'R':
    if (firstFunction) {
        f = std::make_shared<TableLookup>(vocabulary_->GetVocabularySize(),
                                          dimension,
                                          max_batch_size(),
                                          max_sequence_length(),
                                          type >= '2' && type <= '9' ? type - '0' : 1,
                                          type == 'r' || type == 'R',
                                          use_bias,
                                          is_feedforward_,
                                          std::move(g));
    } else {
        f = std::make_shared<Linear>(output_dimension(),
                                     dimension,
                                     max_batch_size(),
                                     max_sequence_length(),
                                     type == 'r' || type == 'R',
                                     use_bias,
                                     std::move(g));
    }
    break;
  case 'm':
  case 'M':
    assert(!firstFunction);
    f = std::make_shared<LSTM>(output_dimension(),
                               dimension,
                               max_batch_size(),
                               max_sequence_length(),
                               use_bias);
    break;
  case 'x':
    f = std::make_shared<Output>(output_dimension(),
                                 max_batch_size(),
                                 max_sequence_length(),
                                 num_oovs_,
                                 use_bias,
                                 vocabulary_,
                                 std::move(g));
    break;
  default:
    assert(false);
  }
  return f;
}

void Net::BuildNetworkLayers(const std::string &net_config,
                             const bool use_bias) {
  std::vector<std::string> tokens;
  boost::split(tokens, net_config, boost::algorithm::is_any_of("-"));
  // first token is arbitrary name
  for (size_t i = 1; i < tokens.size(); ++i) {
    const char type = tokens[i][0];
    assert(type != 'x');
    std::stringstream converter(tokens[i].substr(1));
    int dimension;
    converter >> dimension;
    ActivationFunctionPointer g = SetUpActivationFunction(type);
    FunctionPointer f = SetUpFunction(type,
                                      dimension,
                                      i == 1,
                                      use_bias,
                                      std::move(g));
    Compose(f);
  }
  // add final softmax layer
  ActivationFunctionPointer g = SetUpActivationFunction('x');
  FunctionPointer f = SetUpFunction('x',
                                    vocabulary_->GetVocabularySize(),
                                    false,
                                    use_bias,
                                    std::move(g));
  Compose(f);
}

void Net::Read(const std::string &file_name) {
  assert(boost::filesystem::exists(file_name));
  std::ifstream file(file_name.c_str(), std::ios::in | std::ios::binary);
  assert(file.good());
  Read(&file);
  file.close();
}

void Net::Write(const std::string &file_name) {
  if (boost::filesystem::exists(file_name)) {
    boost::filesystem::copy_file(
        file_name, file_name + ".bk",
        boost::filesystem::copy_option::overwrite_if_exists);
  }
  std::ofstream file(file_name.c_str(), std::ios::out | std::ios::binary);
  Write(&file);
  file.close();
}
