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
#include <iostream>
#include <numeric>
#include <vector>
#include <boost/iterator/iterator_facade.hpp>
#include "random.h"
#include "vocabulary.h"

typedef std::vector<int> Sequence;
typedef std::vector<Sequence> SequenceVector;

enum WordWrappingType {
  kConcatenated, kFixed, kVerbatim
};

class BatchIterator : public boost::iterator_facade<BatchIterator,
    const Sequence, boost::bidirectional_traversal_tag> {
public:
  BatchIterator() : position_(0), offset_(0) {
    // forward traversal iterators require default constructor (§17.6.3.1)
  }

  BatchIterator(const SequenceVector::const_iterator &begin,
                const SequenceVector::const_iterator &end,
                const int position, const int offset)
      : begin_(begin), end_(end), position_(position), offset_(offset) {
    assert(offset == 0 || offset == 1);
    for (auto it = begin; it != end && !it->empty(); ++it) {
      if (position >= static_cast<int>(it->size() - 1) + offset)
        break;
      slice_.push_back(it->at(position));
    }
  }

private:
  friend class boost::iterator_core_access;

  void increment() {
    ++position_;
    slice_.clear();
    for (auto it = begin_; it != end_; ++it) {
      if (position_ >= static_cast<int>(it->size()) -1 + offset_)
        break;
      slice_.push_back((*it)[position_]);
    }
  }

  void decrement() {
    slice_.clear();
    --position_;
    for (auto it = begin_; it != end_; ++it) {
      if (position_ < offset_ || position_ >= 
          static_cast<int>(it->size()) - 1 + offset_)
        break;
      slice_.push_back((*it)[position_]);
    }
  }

  bool equal(const BatchIterator &other) const {
    return position_ == other.position_;
  }

  const Sequence &dereference() const {
    return slice_;
  }

  int position_;
  Sequence slice_;
  const int offset_;
  const SequenceVector::const_iterator begin_, end_;
};

class Batch {
public:
  Batch(const SequenceVector::const_iterator &begin_sequence,
        const SequenceVector::const_iterator &end_sequence)
      : begin_sequence_(begin_sequence), end_sequence_(end_sequence) {
  }

  BatchIterator Begin(const int offset) const {
    return BatchIterator(begin_sequence(), end_sequence(), offset, offset);
  }

  BatchIterator End(const int offset) const {
    return BatchIterator(begin_sequence(), end_sequence(),
                         begin_sequence()->size() - 1 + offset, offset);
  }

  BatchIterator begin() const {
    return Begin(1);
  }

  BatchIterator end() const {
    return End(1);
  }

private:
  friend class DataIterator;

  SequenceVector::const_iterator begin_sequence() const {
    return begin_sequence_;
  }

  SequenceVector::const_iterator end_sequence() const {
    return end_sequence_;
  }

  void set_begin_sequence(const SequenceVector::const_iterator &begin_sequence) {
    begin_sequence_ = begin_sequence;
  }

  void set_end_sequence(const SequenceVector::const_iterator &end_sequence) {
    end_sequence_ = end_sequence;
  }

  SequenceVector::const_iterator begin_sequence_, end_sequence_;
};

class DataIterator : public boost::iterator_facade<DataIterator, const Batch,
    boost::incrementable_traversal_tag> {
public:
  DataIterator(const SequenceVector::const_iterator &begin,
               const SequenceVector::const_iterator &end,
               const int max_batch_size)
      : batch_(begin, end - begin >= max_batch_size ?
               begin + max_batch_size : end),
        data_end_(end),
        max_batch_size_(max_batch_size) {
  }

private:
  friend class boost::iterator_core_access;

  void increment() {
    batch_.set_begin_sequence(data_end_ - batch_.begin_sequence() >
                              max_batch_size_ ? batch_.begin_sequence() +
                              max_batch_size_ : data_end_);
    batch_.set_end_sequence(data_end_ - batch_.end_sequence() >
                            max_batch_size_ ? batch_.end_sequence() +
                            max_batch_size_ : data_end_);
  }

  bool equal(const DataIterator &other) const {
    return batch_.begin_sequence() == other.batch_.begin_sequence();
  }

  const Batch &dereference() const {
    return batch_;
  }

  const int max_batch_size_;
  const SequenceVector::const_iterator data_end_;
  Batch batch_;
};

class Data {
public:
  Data(const std::string &data_file_name,
       const int max_batch_size,
       const int max_sequence_length,
       const WordWrappingType word_wrapping_type,
       const bool debug_no_sb,
       ConstVocabularyPointer vocabulary);

  Data(const SequenceVector data,
       const int max_batch_size,
       const int max_sequence_length,
       ConstVocabularyPointer vocabulary);

  void Shuffle(Random *random) {
    // sort: current shuffling result shall not depend on previous shuffling
    std::sort(data_.begin(), data_.end());
    std::random_shuffle(data_.begin(), data_.end(), *random);
    SortBatches();
  }

  int64_t CountNumRunningWords() const {
    // subtract one for sentence begin token
    return std::accumulate(data_.begin(), data_.end(), 0LL,
                           [](const int64_t sum, const Sequence &s)
                           { return sum + s.size() - 1; });
  }

  int GetNumBatches() const {
    return (data_.size() + max_batch_size() - 1) / max_batch_size();
  }

  int GetVocabularySize() const {
    return vocabulary_->GetVocabularySize();
  }

  DataIterator begin() const {
    return DataIterator(data_.begin(), data_.end(), max_batch_size_);
  }

  DataIterator end() const {
    return DataIterator(data_.end(), data_.end(), max_batch_size_);
  }

  int max_batch_size() const {
    return max_batch_size_;
  }

private:
  friend class GradientTest;

  int64_t ReadIndices(const std::string &data_file_name,
                      const ConstVocabularyPointer &vocabulary,
                      SequenceVector *data);

  void PrepareDataSequenceWise(const std::string &data_file_name,
                               const bool concatenate);

  void PrepareDataWithFixedLength(const std::string &data_file_name);

  void Append(const size_t max_length,
              const bool concatenate,
              Sequence *current);

  void SortBatches();

  const int max_batch_size_, max_sequence_length_;
  const bool debug_no_sb_;
  const ConstVocabularyPointer vocabulary_;
  SequenceVector data_;
};

typedef std::shared_ptr<Data> DataPointer;
