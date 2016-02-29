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
#include <cstdint>  // for int64_t
#include <sstream>
#include <boost/algorithm/string/trim.hpp>
#include "data.h"
#include "file.h"

Data::Data(const std::string &data_file_name,
           const int max_batch_size,
           const int max_sequence_length,
           const WordWrappingType word_wrapping_type,
           const bool debug_no_sb,
           ConstVocabularyPointer vocabulary)
    : max_batch_size_(max_batch_size),
      max_sequence_length_(max_sequence_length),
      debug_no_sb_(debug_no_sb),
      vocabulary_(vocabulary) {
  switch (word_wrapping_type) {
  case kConcatenated:
    PrepareDataSequenceWise(data_file_name, true);
    break;
  case kVerbatim:
    PrepareDataSequenceWise(data_file_name, false);
    break;
  case kFixed:
    PrepareDataWithFixedLength(data_file_name);
    break;
  }
}

Data::Data(const SequenceVector data,
           const int max_batch_size,
           const int max_sequence_length,
           ConstVocabularyPointer vocabulary)
    : data_(data),
      max_batch_size_(max_batch_size),
      max_sequence_length_(max_sequence_length),
      debug_no_sb_(false),
      vocabulary_(vocabulary) {
}

int64_t Data::ReadIndices(const std::string &data_file_name,
                          const ConstVocabularyPointer &vocabulary,
                          SequenceVector *data) {
  int64_t num_running_words = 0;
  assert(data->empty());
  // read text from file
  ReadableFile file(data_file_name);
  std::string line, word;
  while (file.GetLine(&line)) {
    // istringstream does not work with trailing whitespace (duplicate words)
    boost::trim(line);
    assert(!boost::algorithm::starts_with(line, "<s>"));
    assert(!boost::algorithm::ends_with(line, "</s>"));
    std::istringstream iss(line);
    std::vector<int> indices;
    while (!iss.eof()) {
      iss >> word;
      indices.push_back(vocabulary->GetIndex(word));
    }
    assert(!vocabulary->IsSentenceBoundary(word));
    if (!debug_no_sb_)
      indices.push_back(vocabulary->sb_index());
    num_running_words += indices.size();
    data->push_back(indices);
  }
  return num_running_words;
}

void Data::PrepareDataSequenceWise(const std::string &data_file_name,
                                   const bool concatenate) {
  SequenceVector indices;
  ReadIndices(data_file_name, vocabulary_, &indices);
  for (Sequence line : indices) {
    Append(max_sequence_length_ - 1, concatenate, &line);
  }

  // prepend last token of previous sequence
  if (!debug_no_sb_) {
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i].insert(data_[i].begin(),
                      i == 0 ? vocabulary_->sb_index() : data_[i - 1].back());
    }
  }
  SortBatches();
  for (auto &sentence : data_) {
    assert(sentence.size() <= static_cast<size_t>(max_sequence_length_));
  }
}

void Data::PrepareDataWithFixedLength(const std::string &data_file_name) {
  SequenceVector indices;
  int64_t num_running_words = ReadIndices(data_file_name, vocabulary_,
                                          &indices);
  assert(data_.empty());
  data_.resize(max_batch_size_, Sequence());
  int j = 0,
      batch = 0,
      // number of words per batch (incl. final <sb>, excl. overlapping words)
      n = static_cast<int>(num_running_words / max_batch_size_) +
          (static_cast<int>(num_running_words % max_batch_size_) > 0),
      last_word = vocabulary_->sb_index();
  for (Sequence line : indices) {
    // split current line into BPTT sequences
    while (line.size() > 0) {
      // last word used for probability computation, but not yet for training
      if (data_[j].empty())
        data_[j].push_back(last_word);
      const int num_move = std::min(std::min(static_cast<int>(line.size()),
          max_sequence_length_ - static_cast<int>(data_[j].size())), n);
      data_[j].insert(data_[j].end(), line.begin(), line.begin() + num_move);
      line.erase(line.begin(), line.begin() + num_move);
      n -= num_move;
      last_word = data_[j].back();
      // next BPTT sequence?
      if (data_[j].size() == max_sequence_length_) {
        if (batch == 0)
          data_.resize(data_.size() + max_batch_size_, Sequence());
        j += max_batch_size_;
      }
      // next batch?
      if (n == 0 && batch != max_batch_size_ - 1) {
        n = static_cast<int>(num_running_words / max_batch_size_) +
            (++batch < static_cast<int>(num_running_words % max_batch_size_));
        j = batch;
      }
    }
  }
  // we may have added too many empty lines in advance
  while (data_.back().empty())
    data_.pop_back();
}

void Data::Append(const size_t max_length,
                  const bool concatenate,
                  Sequence *current) {
  if (concatenate) {
    if (data_.empty())
      data_.push_back(Sequence());
    Sequence *last = &data_.back();
    if (last->size() + current->size() <= max_length) {
      // sentence can be appended
      last->insert(last->end(), current->begin(), current->end());
    } else if (current->size() <= max_length) {
      // sentence alone is not too long
      data_.push_back(*current);
    } else {
      // break at maximum lengths + append
      while (current->size() > 0) {
        const int size = std::min(max_length - last->size(), current->size());
        last->insert(last->end(), current->begin(), current->begin() + size);
        current->erase(current->begin(), current->begin() + size);
        if (current->size() > 0) {
          data_.push_back(std::vector<int>());
          last = &data_.back();
        }
      }
    }
  } else {
    data_.push_back(*current);
  }
}

void Data::SortBatches() {
  SequenceVector::iterator begin = data_.begin(), end;
  while (begin != data_.end()) {
    end = data_.end() - begin >= max_batch_size_ ?
          begin + max_batch_size_ : data_.end();
    std::sort(begin, end, [](const Sequence &s, const Sequence &t)
              { return s.size() > t.size(); });
    begin = end;
  }
}
