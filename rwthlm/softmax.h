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
#include "function.h"

class Softmax : public ActivationFunction {
public:
  Softmax() : ActivationFunction() {
  }

  virtual ~Softmax() {
  }

  virtual void Evaluate(const int dimension,
                        const int batch_size,
                        Real b_t[]) const;

  virtual void MultiplyDerivative(const int dimension,
                                  const int batch_size,
                                  const Real b_t[],
                                  Real delta_t[]) const;
};
