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
#include <cstdint>
#include "linear.h"
#include "lstm.h"
#include "output.h"
#include "tablelookup.h"
#include "trainer.h"

class GradientTest {
public:
  GradientTest(const uint32_t seed, Trainer *const trainer, const bool is_feedforward);

  virtual ~GradientTest() {
  }

  void Test();

private:
  void Test(TableLookup *table_lookup);
  void Test(Linear *linear);
  void Test(LSTM *lstm);
  void Test(Output *output);

  void ComputeDifferenceQuotient(const int size,
                                 Real weights[],
                                 Real quotient[]);

  void ComputeDerivative(const int size,
                         const Real weights[],
                         Real derivative[]);

  void Compare(const std::string &name,
               const int size,
               const Real derivative[],
               const Real quotient[]) const;

  void TestWeights(const std::string &name, const int size, Real weights[]);

  const int significant_digits_;
  const uint32_t seed_;
  const Real epsilon_;
  int64_t num_running_words_;
  Trainer *const trainer_;
};
