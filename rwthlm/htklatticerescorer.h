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
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/functional/hash.hpp>
#include "fast.h"
#include "function.h"
#include "rescorer.h"

class HtkLatticeRescorer : public Rescorer {
public:
  enum LookAheadSemiring {
    kTropical, kLog, kNone
  };

  enum OutputFormat {
    kCtm, kLattice, kExpandedLattice
  };

  HtkLatticeRescorer(const ConstVocabularyPointer &vocabulary,
                     const NetPointer &net,
                     const OutputFormat output_format,
                     const int num_oov_words,
                     const Real nn_lambda,
                     const LookAheadSemiring semiring,
                     const Real look_ahead_lm_scale,
					 const Real lm_scale,
                     const Real pruning_threshold,
                     const size_t pruning_limit,
                     const int dp_order,
                     const bool is_dependent,
                     const bool clear_initial_links,
                     const bool set_sb_next_to_last_links,
                     const bool set_sb_last_links)
      : Rescorer(vocabulary, net, num_oov_words, nn_lambda),
        unk_index_(vocabulary_->HasUnk() ?
                   vocabulary_->GetIndex(vocabulary_->unk()) : -1),
        output_format_(output_format),
        pruning_limit_(pruning_limit),
        semiring_(semiring),
        look_ahead_lm_scale_(look_ahead_lm_scale),
		lm_scale_(lm_scale),
        pruning_threshold_(pruning_threshold),
        epsilon_(1e-8),
        dp_order_(dp_order),
        is_dependent_(is_dependent),
        clear_initial_links_(clear_initial_links),
        set_sb_next_to_last_links_(set_sb_next_to_last_links),
        set_sb_last_links_(set_sb_last_links) {
  }

  ~HtkLatticeRescorer() {
  }

  virtual void ReadLattice(const std::string &file_name);
  virtual void RescoreLattice();
  virtual void WriteLattice(const std::string &file_name);

private:
  struct Node {
    int id, time;
    Real look_ahead_score;
    bool operator<(const Node &other) const {
      return time < other.time;
    };
  };

  struct Link {
    int from, to, word, pronunciation;
    Real lm_score, am_score;
  };

  struct Hypothesis {
    Hypothesis() {
      traceback_id = 0;
      score = 0.;
    }
    bool operator<(const Hypothesis &other) const {
      return score < other.score;
    }
    State state;
    size_t traceback_id;
    Real score;
  };

  struct Trace {
    Trace(const int link_id,
          const int history_word,
          const int predecessor_traceback_id,
          const Real score)
        : link_id(link_id),
          history_word(history_word),
          predecessor_traceback_id(predecessor_traceback_id),
          score(score) {
    }
    int link_id, predecessor_traceback_id, history_word;
    Real score;
  };

  std::pair<size_t, size_t> ParseField(
      const std::string &line,
      const std::string &field,
      int *const result) const;
  std::pair<size_t, size_t> ParseField(
      const std::string &line,
      const std::string &field,
      Real *const result) const;
  std::pair<size_t, size_t> ParseField(
      const std::string &line,
      const std::string &field,
      std::string *const result = nullptr) const;
  void ParseLine(const std::string &line, int *num_links);

  int AddTraceback(const int link_id,
                   const int history_word,
                   const int predecessor_traceback_id,
                   const Real score) {
    traceback_.push_back(Trace(link_id,
                               history_word,
                               predecessor_traceback_id,
                               score));
    return traceback_.size() - 1;
  }

  int GetHistoryWord(const int traceback_id) const {
    return traceback_[traceback_id].history_word;
  }

  int GetTime(const int traceback_id) const {
    return nodes_[GetToNodeID(traceback_id)].time;
  }

  int GetToNodeID(const int traceback_id) const {
    const int link_id = traceback_[traceback_id].link_id;
    if (link_id < 0)
      return 0;
    return links_[link_id].to;
  }

  int GetLinkID(const int traceback_id) const {
    return traceback_[traceback_id].link_id;
  }

  Real GetScore(const int traceback_id) const {
    return traceback_[traceback_id].score;
  }

  Real GetLinkLmScore(const int traceback_id) const {
    const Link &link = links_[traceback_[traceback_id].link_id];
    const Real from_look_ahead_score = nodes_[link.from].look_ahead_score,
               to_look_ahead_score = nodes_[link.to].look_ahead_score;
	return (GetScore(traceback_id) - GetScore(GetPredecessorID(traceback_id)) -
		   link.am_score - to_look_ahead_score + from_look_ahead_score) / lm_scale_;
  }

  int GetPredecessorID(const int traceback_id) const {
    return traceback_[traceback_id].predecessor_traceback_id;
  }

  size_t Hash(int traceback_id) const {
    size_t hash = 0;
    for (int i = 0; i < dp_order_; ++i) {
      // stop at beginning-of-sentence or "real" word
      while (GetLinkID(traceback_id) >= 0 &&
             links_[GetLinkID(traceback_id)].lm_score == 0.)
        traceback_id = GetPredecessorID(traceback_id);
      boost::hash_combine(hash,
                          static_cast<size_t>(GetHistoryWord(traceback_id)));
      traceback_id = GetPredecessorID(traceback_id);
    }
    return hash;
  }

  Real ScaledLogAdd(const Real scale, Real x, Real y) {
    if (y >= std::numeric_limits<Real>::max())
      return x;
    if (x >= std::numeric_limits<Real>::max())
      return y;
    const Real inverted_scale = 1. / scale;
    x *= inverted_scale;
    y *= inverted_scale;
    const Real min = std::min(x, y);
    return scale * (min - LogOnePlusX(exp(min - std::max(x, y))));
  }

  Real LogOnePlusX(double x) {
    if (x <= -1.) {
      assert(false);
      return -1.;
    }
    if (fabs(x) > 1e-4)
      return log(1.0 + x);
    return (1. - 0.5 * x) * x;
  }

  void SortTopologically();
  void SortTopologicallyHelper(const int node_id,
                               std::unordered_set<int> *visited);
  void ComputeLookAheadScores();
  void Reset();
  void Prune(const int time);
  void TraceBack();
  void TraceBackCtm();
  void TraceBackLattice();
  void TraceBackExpandedLattice();
  void WriteCtm(const std::string &file_name);
  void WriteHtkLattice(const std::string &file_name);
  void WriteExpandedHtkLattice(const std::string &file_name);

  const int unk_index_, dp_order_;
  const size_t pruning_limit_;
  const LookAheadSemiring semiring_;
  const OutputFormat output_format_;
  const Real look_ahead_lm_scale_, lm_scale_, pruning_threshold_, epsilon_;
  const bool is_dependent_,
             clear_initial_links_,
             set_sb_next_to_last_links_,
             set_sb_last_links_;
  std::vector<int> single_best_, topological_order_;
  std::unordered_map<int, std::string> oov_by_link_;
  std::vector<Node> nodes_;
  std::vector<Node *> sorted_nodes_;
  std::vector<Link> links_;
  std::vector<std::vector<int>> successor_links_;
  std::vector<std::priority_queue<Hypothesis>> hypotheses_;
  std::unordered_map<int, std::vector<int>> nodes_by_time_;
  std::vector<Trace> traceback_;
  std::unordered_map<int, Real> best_score_by_time_;
};
