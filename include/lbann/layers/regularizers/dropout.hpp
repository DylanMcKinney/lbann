////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED

#include "lbann/layers/regularizers/regularizer.hpp"

namespace lbann {

/** Dropout layer.
 *  Probabilistically drop layer outputs. See:
 *    Srivastava, Nitish, et al. "Dropout: a simple way to prevent
 *    neural networks from overfitting." Journal of Machine Learning
 *    Research 15.1 (2014).
 *  The weights are multiplied by 1/(keep probability) at training
 *  time, as discussed in section 10 of the paper. Keep probabilities
 *  of 0.5 for fully-connected layers and 0.8 for input layers are
 *  good starting points.
 */
template <data_layout T_layout>
class dropout : public regularizer_layer {
 public:
  /** Keep units with probabiliy keep_prob. */
  dropout(lbann_comm *comm,
          float keep_prob=0.5f) :
    regularizer_layer(comm),
    m_keep_prob(keep_prob) {}

  dropout(const dropout& other) :
    regularizer_layer(other),
    m_keep_prob(other.m_keep_prob),
    m_mask(other.m_mask) {
    if (m_mask != nullptr) { m_mask = m_mask->Copy(); }
  }

  dropout& operator=(const dropout& other) {
    regularizer_layer::operator=(other);
    m_keep_prob = other.m_keep_prob;
    if (m_mask != nullptr) { delete m_mask; }
    m_mask = other.m_mask;
    if (m_mask != nullptr) { m_mask = m_mask->Copy(); }
    return *this;
  }

  ~dropout() override {
    if (m_mask != nullptr) { delete m_mask; }
  }

  dropout* copy() const override { return new dropout(*this); }

  std::string get_type() const override { return "dropout"; }

  std::string get_description() const override {
    return " dropout keep_prob: " + std::to_string(m_keep_prob) 
           + " dataLayout: " + get_data_layout_string(get_data_layout());
  }

  void setup_matrices(const El::Grid& grid) override {
    regularizer_layer::setup_matrices(grid);
    if (m_mask != nullptr) { delete m_mask; }
    m_mask = get_activations().Copy();    
  }
  data_layout get_data_layout() const override { return T_layout; }

 protected:
  /** Drop out units in forward propagation. */
  void fp_compute() override {

    // Matrices
    const auto& input = get_prev_activations();
    auto& output = get_activations();

    if (this->m_model->get_execution_mode() != execution_mode::training
        || m_keep_prob < 0.0f) {
      // Do nothing if dropout is disabled
      El::LockedView(output, input);
    } else {

      // Construct mask matrix
      const DataType scale = DataType(1) / m_keep_prob;
      const int height = input.Height();
      const int width = input.Width();
      m_mask->Resize(height, width);
#ifdef LBANN_SEQUENTIAL_CONSISTENCY
      bernoulli_fill_procdet(*m_mask, height, width, m_keep_prob);
      *m_mask *= scale;
#else
      El::EntrywiseMap(*m_mask,
                       (std::function<DataType(const DataType&)>)
                       ([this,scale](const DataType& z)->DataType {
                         auto& gen = get_fast_generator();
                         std::bernoulli_distribution dist(m_keep_prob);
                         return dist(gen) ? scale : DataType(0);
                       }));
#endif // LBANN_SEQUENTIAL_CONSISTENCY

      // Apply mask matrix to get activations
      El::Hadamard(input, *m_mask, output);

    }
  }

  /** Adjust gradients for dropout in backprop. */
  void bp_compute() override {
    const auto& gradient_wrt_output = get_prev_error_signals();
    auto& gradient_wrt_input = get_error_signals();
    if (this->m_model->get_execution_mode() != execution_mode::training
        || m_keep_prob < 0.0f) {
      El::Axpy(DataType(1), gradient_wrt_output, gradient_wrt_input);
    } else {
      El::Hadamard(gradient_wrt_output, *m_mask, *m_mask);
      El::Axpy(DataType(1), *m_mask, gradient_wrt_input);
    }
  }

  /** Probability of keeping each unit. */
  float m_keep_prob;
  /** Current dropout mask (a scaled Bernoulli random matrix). */
  AbsDistMat *m_mask = nullptr;

};

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED
