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

#ifndef RESHAPE_HPP_INCLUDED
#define RESHAPE_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** Reshape layer */
template <data_layout T_layout>
class reshape_layer : public transform_layer {
 public:
  reshape_layer(lbann_comm *comm,
                int num_dims,
                const int *dims) :
    transform_layer(comm) {
    this->m_num_neuron_dims = num_dims;
    this->m_neuron_dims.assign(dims, dims+num_dims);
  }
  reshape_layer* copy() const override { return new reshape_layer(*this); }
  std::string get_type() const override { return "reshape"; }
  data_layout get_data_layout() const override { return T_layout; }

  void setup_dims() override {
    // Store neuron tensor dimensions
    const int num_neuron_dims = this->m_num_neuron_dims;
    const std::vector<int> neuron_dims = this->m_neuron_dims;

    // Initialize previous neuron tensor dimensions
    transform_layer::setup_dims();

    // Initialize neuron tensor dimensions
    this->m_num_neuron_dims = num_neuron_dims;
    this->m_neuron_dims = neuron_dims;

    // Determine any unspecified dimensions
    int unspecified_dim = -1;
    for(int dim = 0; dim < this->m_num_neuron_dims; ++dim) {
      if(this->m_neuron_dims[dim] <= 0) {
        unspecified_dim = dim;
        this->m_neuron_dims[dim] = 1;
      }
    }
    if(unspecified_dim >= 0) {
      const int specified_size = std::accumulate(this->m_neuron_dims.begin(),
                                                 this->m_neuron_dims.end(),
                                                 1,
                                                 std::multiplies<int>());
      this->m_neuron_dims[unspecified_dim] = this->m_num_neurons / specified_size;
    }

    // Check that reshape is valid
    if(this->m_num_neurons != std::accumulate(this->m_neuron_dims.begin(),
                                              this->m_neuron_dims.end(),
                                              1,
                                              std::multiplies<int>())) {
      std::string num_neurons = " {";
      for(int n: m_neuron_dims) num_neurons += " " + std::to_string(n);
      num_neurons += " }";
      throw lbann_exception("reshape_layer: invalid neuron dimensions, " +
                             std::to_string(m_num_neurons) + " != " + num_neurons);
    }

  }

  void fp_compute() override {
    El::LockedView(get_activations(), get_prev_activations());
  }

  void bp_compute() override {
    El::Axpy(DataType(1), get_prev_error_signals(), get_error_signals());
  }

};

} // namespace lbann

#endif // RESHAPE_HPP_INCLUDED
