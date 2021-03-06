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

#ifndef LBANN_LAYER_NOISE_HPP_INCLUDED
#define LBANN_LAYER_NOISE_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** Layer draws outputs from a Gaussian distribution.
 *  During validation and testing, the layer outputs zeros.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class noise_layer : public transform_layer {
 private:
  /** Noise factor */
  DataType m_noise_factor;

 public:
  noise_layer(lbann_comm *comm,
              const std::vector<int>& neuron_dims,
              DataType noise_factor=DataType(1.0),
              cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm),
      m_noise_factor(noise_factor) {

    // Record neuron dimensions
    this->m_neuron_dims = neuron_dims;
    this->m_num_neuron_dims = neuron_dims.size();
    this->m_num_neurons = std::accumulate(neuron_dims.begin(),
                                          neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

    // Constant layer has no parents
    m_expected_num_parent_layers = 0;
  }
  noise_layer* copy() const override { return new noise_layer(*this); }
  std::string get_type() const override { return "noise"; }
  data_layout get_data_layout() const override { return T_layout; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
     s << "noise_layer  noise_factor: " << m_noise_factor
       << " dataLayout: " << this->get_data_layout_string(get_data_layout());
     return s.str();
  }

 protected:

  void setup_dims() override {
    const auto neuron_dims = this->m_neuron_dims;
    transform_layer::setup_dims();
    this->m_neuron_dims = neuron_dims;
    this->m_num_neuron_dims = neuron_dims.size();
    this->m_num_neurons = std::accumulate(neuron_dims.begin(),
                                          neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());
  }

  void fp_compute() override {
    auto& output = get_activations();
    if (this->m_model->get_execution_mode() == execution_mode::training) {
      gaussian_fill(output,
                    output.Height(), output.Width(),
                    DataType(0), m_noise_factor);
    } else {
      El::Zero(output);
    }
  }

  void bp_compute() override {
  }

};

} // namespace lbann

#endif // LBANN_LAYER_NOISE_HPP_INCLUDED
