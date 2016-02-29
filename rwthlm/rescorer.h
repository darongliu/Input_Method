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
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include "net.h"

class Rescorer {
public:
  Rescorer(const ConstVocabularyPointer &vocabulary,
           const NetPointer &net,
           const int num_oov_words,
           const Real nn_lambda)
      : vocabulary_(vocabulary),
        net_(net),
        nn_lambda_(nn_lambda),
        num_oov_words_(num_oov_words) {
  }

  virtual ~Rescorer() {
  }

  void Rescore(const std::vector<std::string> &file_names) {
    std::cout << "Rescoring ..." << std::endl;
    for (auto &file_name : file_names) {
      std::cout << "lattice '" << file_name << "' ..." << std::endl;
      Reset();
      ReadLattice(file_name);
      RescoreLattice();
      WriteLattice(file_name);
    }
  }

protected:
  virtual void Reset() = 0;
  virtual void ReadLattice(const std::string &file_name) = 0;
  virtual void RescoreLattice() = 0;
  virtual void WriteLattice(const std::string &file_name) = 0;

  static std::string ExtendedFileName(const std::string &file_name,
                                      const std::string &extension) {
    const bool ends_with_gz = boost::algorithm::ends_with(file_name, ".gz");
    return (ends_with_gz ? file_name.substr(0, file_name.size() - 3) :
            file_name) + extension + (ends_with_gz ? ".gz" : "");
  }

  static std::string FileNameWithoutExtension(const std::string &file_name) {
    const bool ends_with_gz = boost::algorithm::ends_with(file_name, ".gz");
    return file_name.substr(
        0,
        file_name.substr(0, 
                         file_name.size() - (ends_with_gz ? 3 : 0)).rfind('.'));
  }

  const Real nn_lambda_;
  const int num_oov_words_;
  const ConstVocabularyPointer &vocabulary_;
  const NetPointer &net_;
};

typedef std::unique_ptr<Rescorer> RescorerPointer;
