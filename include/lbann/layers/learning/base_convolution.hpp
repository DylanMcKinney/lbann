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
//
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_BASE_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYER_BASE_CONVOLUTION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/learning/learning.hpp"
#include "lbann/base.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/im2col.hpp"

namespace lbann {

// Forward declaration.
class lbann_callback_imcomm;

/// Base Convolution layer
class base_convolution_layer : public learning {
 protected:

  friend class lbann_callback_imcomm;

  /// Weight initialization scheme
  weight_initialization m_weight_initialization;
  /// Convolutional filter dimensions
  std::vector<int> m_conv_dims;
  /// Size of convolutional filters
  int m_conv_size;
  /// Convolution padding
  std::vector<int> m_conv_pads;
  /// Convolution strides
  std::vector<int> m_conv_strides;

  /// View into convolutional filter weights
  ElMat *m_filter_weights_v;
  /// View into convolutional filter weights gradient
  ElMat *m_filter_weights_gradient_v;
  /// View into bias weights
  ElMat *m_bias_weights_v;
  /// View into bias weights gradient
  ElMat *m_bias_weights_gradient_v;

  /// Scaling factor for bias term
  DataType m_bias_scaling_factor;

#ifdef __LIB_CUDNN

  /// Bias tensor descriptor
  cudnnTensorDescriptor_t m_bias_desc;
  /// Filter descriptor
  cudnnFilterDescriptor_t m_filter_desc;
  /// Convolution descriptor
  cudnnConvolutionDescriptor_t m_convolution_desc;

  /// Forward pass algorithm
  cudnnConvolutionFwdAlgo_t m_forward_algo;
  /// Backward pass filter algorithm
  /** Compute gradient w.r.t. filter. */
  cudnnConvolutionBwdFilterAlgo_t m_backward_filter_algo;
  /// Backward pass data algorithm
  /** Compute gradient w.r.t. data, which is passed to previous layer. */
  cudnnConvolutionBwdDataAlgo_t m_backward_data_algo;

  /// GPU memory for convolution filters
  std::vector<DataType *> m_filter_weights_d;
  /// GPU memory for convolution filters gradient
  std::vector<DataType *> m_filter_weights_gradient_d;
  /// GPU memory for convolution bias
  std::vector<DataType *> m_bias_weights_d;
  /// GPU memory for convolution bias gradient
  std::vector<DataType *> m_bias_weights_gradient_d;

  /// Filter and bias gradients computed on each GPU
  StarMat m_weights_gradient_per_gpu;

#endif // __LIB_CUDNN

  public:

  base_convolution_layer(int index,
                    lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    int conv_dim,
                    int conv_pad,
                    int conv_stride,
                    weight_initialization init,
                    optimizer *opt,
                    bool has_bias = true,
                    cudnn::cudnn_manager *cudnn = NULL)
    : base_convolution_layer(index,
                        comm,
                        num_data_dims,
                        num_output_channels,
                        std::vector<int>(num_data_dims, conv_dim).data(),
                        std::vector<int>(num_data_dims, conv_pad).data(),
                        std::vector<int>(num_data_dims, conv_stride).data(),
                        init,
                        opt,
                        has_bias,
                        cudnn) {}

  base_convolution_layer(int index,
                    lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    const int *conv_dims,
                    const int *conv_pads,
                    const int *conv_strides,
                    weight_initialization init,
                    optimizer *opt,
                    bool has_bias = true,
                    cudnn::cudnn_manager *cudnn = NULL)
    : learning(index, comm, opt),
      m_weight_initialization(init) {

    // Initialize convolution parameters
    m_conv_dims.assign(conv_dims, conv_dims+num_data_dims);
    m_conv_pads.assign(conv_pads, conv_pads+num_data_dims);
    m_conv_strides.assign(conv_strides, conv_strides+num_data_dims);

    // Initialize neuron tensor dimensions
    this->m_num_neuron_dims = num_data_dims + 1;
    this->m_neuron_dims.resize(this->m_num_neuron_dims);
    this->m_neuron_dims[0] = num_output_channels;///???deconv

    // Activate or disable bias
    m_bias_scaling_factor = has_bias ? DataType(1) : DataType(0);

  #ifdef __LIB_CUDNN
    m_weights_gradient_per_gpu = StarMat(this->m_comm->get_model_grid());
  #endif // #ifdef __LIB_CUDNN

  #ifdef __LIB_CUDNN

    // Initialize cuDNN objects
    m_bias_desc = NULL;
    m_filter_desc = NULL;
    m_convolution_desc = NULL;
    m_forward_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    m_backward_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    m_backward_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

    // Set parameters for GPU implementation
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }

  #endif // #ifdef __LIB_CUDNN

  }

  base_convolution_layer(const base_convolution_layer& other) :
    learning(other),
    m_weight_initialization(other.m_weight_initialization),
    m_conv_dims(other.m_conv_dims),
    m_conv_size(other.m_conv_size),
    m_conv_pads(other.m_conv_pads),
    m_conv_strides(other.m_conv_strides) {
    m_filter_weights_v = other.m_filter_weights_v->Copy();
    m_filter_weights_gradient_v = other.m_filter_weights_gradient_v->Copy();
    m_bias_weights_v = other.m_bias_weights_v->Copy();
    m_bias_weights_gradient_v = other.m_bias_weights_gradient_v->Copy();
    m_bias_scaling_factor = other.m_bias_scaling_factor;
    setup_views();  // Update views.
    // Update optimizer parameters if needed.
    if (this->m_optimizer->get_parameters()) {
      this->m_optimizer->set_parameters(this->m_weights);
    }
  }

  base_convolution_layer& operator=(const base_convolution_layer& other) {
    learning::operator=(other);
    m_weight_initialization = other.m_weight_initialization;
    m_conv_dims = other.m_conv_dims;
    m_conv_size = other.m_conv_size;
    m_conv_pads = other.m_conv_pads;
    m_conv_strides = other.m_conv_strides;
    if (m_filter_weights_v) {
      delete m_filter_weights_v;
      delete m_filter_weights_gradient_v;
      delete m_bias_weights_v;
      delete m_bias_weights_gradient_v;
    }
    m_filter_weights_v = other.m_filter_weights_v->Copy();
    m_filter_weights_gradient_v = other.m_filter_weights_gradient_v->Copy();
    m_bias_weights_v = other.m_bias_weights_v->Copy();
    m_bias_weights_gradient_v = other.m_bias_weights_gradient_v->Copy();
    m_bias_scaling_factor = other.m_bias_scaling_factor;
    setup_views();  // Update views.
    // Update optimizer parameters if needed.
    if (this->m_optimizer->get_parameters()) {
      this->m_optimizer->set_parameters(this->m_weights);
    }
    return *this;
  }

  ~base_convolution_layer() {
  #ifdef __LIB_CUDNN
    // Destroy cuDNN objects
    if(m_bias_desc) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_bias_desc));
    }
    if(m_filter_desc) {
      CHECK_CUDNN(cudnnDestroyFilterDescriptor(m_filter_desc));
    }
    if(m_convolution_desc) {
      CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(m_convolution_desc));
    }

    // Deallocate GPU memory
    this->m_cudnn->deallocate_on_gpus(m_filter_weights_d);
    this->m_cudnn->deallocate_on_gpus(m_filter_weights_gradient_d);
    this->m_cudnn->deallocate_on_gpus(m_bias_weights_d);
    this->m_cudnn->deallocate_on_gpus(m_bias_weights_gradient_d);

    // Unpin host memory
    if(this->m_using_gpus) {
      this->m_cudnn->unpin_matrix(*(this->m_weights));
      this->m_cudnn->unpin_matrix(m_weights_gradient_per_gpu);
    }
  #endif // #ifdef __LIB_CUDNN

    delete m_filter_weights_v;
    delete m_filter_weights_gradient_v;
    delete m_bias_weights_v;
    delete m_bias_weights_gradient_v;
  }


  template<data_layout T_layout> void initialize_distributed_matrices() {
    learning::initialize_distributed_matrices<T_layout>();

    /// Instantiate these view objects but do not allocate data for them
    /// TODO: model parallel implementation
    m_filter_weights_v           = new StarMat(this->m_comm->get_model_grid());
    m_filter_weights_gradient_v  = new StarMat(this->m_comm->get_model_grid());
    m_bias_weights_v             = new StarMat(this->m_comm->get_model_grid());
    m_bias_weights_gradient_v    = new StarMat(this->m_comm->get_model_grid());
  }
  
  // reverse_dimensions() needed for implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;

  void setup_dims() {
    // Initialize previous neuron tensor dimensions
    learning::setup_dims();

  }

  void setup_data() {
    learning::setup_data();

    // Initialize matrices
    this->m_weights->Resize(m_conv_size / this->m_neuron_dims[0] + 1,
                            this->m_neuron_dims[0]);
    this->m_weights_gradient->Resize(this->m_weights->Height(),
                                     this->m_weights->Width());
  #ifdef __LIB_CUDNN
    if(this->m_using_gpus) {
      m_weights_gradient_per_gpu.Resize(this->m_weights_gradient->Height(),
                                        this->m_weights_gradient->Width()
                                        * this->m_cudnn->get_num_gpus());
    }

    // Pin host memory
    if(this->m_using_gpus) {
      this->m_cudnn->pin_matrix(*this->m_weights);
      this->m_cudnn->pin_matrix(m_weights_gradient_per_gpu);
    }
  #endif // #ifdef __LIB_CUDNN

    // Initialize filters
    const int fan_in = m_conv_size / this->m_neuron_dims[0];
    const int fan_out = m_conv_size / this->m_prev_neuron_dims[0];
    El::View(*m_filter_weights_v, *this->m_weights,
             El::IR(0,this->m_weights->Height()-1), El::ALL);
    initialize_matrix(*m_filter_weights_v, this->m_weight_initialization, fan_in, fan_out);

    // Initialize bias
    El::View(*m_bias_weights_v, *this->m_weights,
             El::IR(this->m_weights->Height()-1), El::ALL);
    El::Zero(*m_bias_weights_v);

    // Initialize optimizer
    if(this->m_optimizer != NULL) {
      this->m_optimizer->setup(this->m_weights);
    }

  }

  void setup_views() {
    learning::setup_views();
    // Set up views into weights and weights gradient
    El::View(*m_filter_weights_v, *this->m_weights,
             El::IR(0,this->m_weights->Height()-1), El::ALL);
    El::View(*m_filter_weights_gradient_v, *this->m_weights_gradient,
             El::IR(0,this->m_weights_gradient->Height()-1), El::ALL);
    El::View(*m_bias_weights_v, *this->m_weights,
             El::IR(this->m_weights->Height()-1), El::ALL);
    El::View(*m_bias_weights_gradient_v, *this->m_weights_gradient,
             El::IR(this->m_weights_gradient->Height()-1), El::ALL);
  }

  /// Initialize GPU objects
  void setup_gpu() {
    learning::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else

    // Get device properties
    cudaDeviceProp device_props;
    CHECK_CUDA(cudaGetDeviceProperties(&device_props, 0));

    // Maximum workspace size
    const size_t work_space_limit = device_props.totalGlobalMem/2;

    // Initialize descriptors
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&m_filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&m_convolution_desc));

    // Set filter descriptor
    std::vector<int> conv_dims = m_conv_dims;
    conv_dims.insert(conv_dims.begin(), this->m_prev_neuron_dims[0]);
    conv_dims.insert(conv_dims.begin(), this->m_neuron_dims[0]);
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(m_filter_desc,
                                           this->m_cudnn->get_cudnn_data_type(),
                                           CUDNN_TENSOR_NCHW,
                                           conv_dims.size(),
                                           conv_dims.data()));

    // Set convolution descriptor
    // Note: upscales are not supported as of cuDNN v5.1
    std::vector<int> conv_upscales(this->m_num_neuron_dims-1, 1);
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(m_convolution_desc,
                                                m_conv_pads.size(),
                                                m_conv_pads.data(),
                                                m_conv_strides.data(),
                                                conv_upscales.data(),
                                                CUDNN_CONVOLUTION,
                                                this->m_cudnn->get_cudnn_data_type()));

    // Set bias tensor descriptor
    std::vector<int> bias_dims(this->m_num_prev_neuron_dims+1, 1);
    bias_dims[1] = this->m_bias_weights_v->Width();
    std::vector<int> bias_strides(this->m_num_prev_neuron_dims+1, 1);
    bias_strides[0] = bias_dims[1];
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_bias_desc,
                                           this->m_cudnn->get_cudnn_data_type(),
                                           bias_dims.size(),
                                           bias_dims.data(),
                                           bias_strides.data()));

    // Choose algorithms
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(this->m_cudnn->get_handle(),
                                                    this->m_prev_neurons_cudnn_desc,
                                                    m_filter_desc,
                                                    m_convolution_desc,
                                                    this->m_neurons_cudnn_desc,
                                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                    work_space_limit,
                                                    &m_forward_algo));
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->m_cudnn->get_handle(),
                                                           this->m_prev_neurons_cudnn_desc,
                                                           this->m_neurons_cudnn_desc,
                                                           m_convolution_desc,
                                                           m_filter_desc,
                                                           CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                           work_space_limit,
                                                           &m_backward_filter_algo));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(this->m_cudnn->get_handle(),
                                                         m_filter_desc,
                                                         this->m_neurons_cudnn_desc,
                                                         m_convolution_desc,
                                                         this->m_prev_neurons_cudnn_desc,
                                                         CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                                                         work_space_limit,
                                                         &m_backward_data_algo));

    // Initialize work space
    size_t max_work_space = 0;
    size_t required_work_space;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(this->m_cudnn->get_handle(),
                                                        this->m_prev_neurons_cudnn_desc,
                                                        m_filter_desc,
                                                        m_convolution_desc,
                                                        this->m_neurons_cudnn_desc,
                                                        m_forward_algo,
                                                        &required_work_space));
    max_work_space = std::max(max_work_space, required_work_space);
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(this->m_cudnn->get_handle(),
                                                               this->m_prev_neurons_cudnn_desc,
                                                               this->m_neurons_cudnn_desc,
                                                               m_convolution_desc,
                                                               m_filter_desc,
                                                               m_backward_filter_algo,
                                                               &required_work_space));
    max_work_space = std::max(max_work_space, required_work_space);
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(this->m_cudnn->get_handle(),
                                                             m_filter_desc,
                                                             this->m_neurons_cudnn_desc,
                                                             m_convolution_desc,
                                                             this->m_prev_neurons_cudnn_desc,
                                                             m_backward_data_algo,
                                                             &required_work_space));
    max_work_space = std::max(max_work_space, required_work_space);
    for(int i=0; i<this->m_cudnn->get_num_gpus(); ++i) {
      if(max_work_space > this->m_cudnn->get_work_space_size(i)) {
        this->m_cudnn->set_work_space_size(i, max_work_space);
      }
    }

    // Allocate GPU memory
    this->m_cudnn->allocate_on_gpus(this->m_filter_weights_d,
                                    this->m_filter_weights_v->Height(),
                                    this->m_filter_weights_v->Width());
    this->m_cudnn->allocate_on_gpus(this->m_filter_weights_gradient_d,
                                    this->m_filter_weights_gradient_v->Height(),
                                    this->m_filter_weights_gradient_v->Width());
    this->m_cudnn->allocate_on_gpus(this->m_bias_weights_d,
                                    this->m_bias_weights_v->Height(),
                                    this->m_bias_weights_v->Width());
    this->m_cudnn->allocate_on_gpus(this->m_bias_weights_gradient_d,
                                    this->m_bias_weights_gradient_v->Height(),
                                    this->m_bias_weights_gradient_v->Width());

  #endif // #ifdef __LIB_CUDNN
  }

 protected:

  /// Convolution forward propagation with cuDNN
  void fp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Transfer filters and bias from CPU to GPUs
    this->m_cudnn->broadcast_to_gpus(m_filter_weights_d,
                                     this->m_filter_weights_v->LockedMatrix());
    if(m_bias_scaling_factor != DataType(0)) {
      this->m_cudnn->broadcast_to_gpus(m_bias_weights_d,
                                       this->m_bias_weights_v->LockedMatrix());
    }

    // Perform convolution on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnConvolutionForward(this->m_cudnn->get_handle(i),
                                          &one,
                                          this->m_prev_neurons_cudnn_desc,
                                          this->m_prev_activations_d[i],
                                          m_filter_desc,
                                          m_filter_weights_d[i],
                                          m_convolution_desc,
                                          m_forward_algo,
                                          this->m_cudnn->get_work_space(i),
                                          this->m_cudnn->get_work_space_size(i),
                                          &zero,
                                          this->m_neurons_cudnn_desc,
                                          this->m_activations_d[i]));
      CHECK_CUDNN(cudnnAddTensor(this->m_cudnn->get_handle(i),
                                 &m_bias_scaling_factor,
                                 m_bias_desc,
                                 m_bias_weights_d[i],
                                 &one,
                                 this->m_neurons_cudnn_desc,
                                 this->m_activations_d[i]));
    }

  #endif // #ifndef __LIB_CUDNN
  }

  /// Convolution backward propagation with cuDNN
  void bp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("base_convolution_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Clear unused columns
    this->m_cudnn->clear_unused_columns_on_gpus(this->m_prev_error_signal_d,
                                                this->m_num_neurons,
                                                this->m_prev_error_signal_v->LocalWidth(),
                                                this->m_mini_batch_size_per_gpu);

    // Perform back propagation on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnConvolutionBackwardBias(this->m_cudnn->get_handle(i),
                                               &m_bias_scaling_factor,
                                               this->m_neurons_cudnn_desc,
                                               this->m_prev_error_signal_d[i],
                                               &zero,
                                               m_bias_desc,
                                               m_bias_weights_gradient_d[i]));
      CHECK_CUDNN(cudnnConvolutionBackwardFilter(this->m_cudnn->get_handle(i),
                                                 &one,
                                                 this->m_prev_neurons_cudnn_desc,
                                                 this->m_prev_activations_d[i],
                                                 this->m_neurons_cudnn_desc,
                                                 this->m_prev_error_signal_d[i],
                                                 m_convolution_desc,
                                                 m_backward_filter_algo,
                                                 this->m_cudnn->get_work_space(i),
                                                 this->m_cudnn->get_work_space_size(i),
                                                 &zero,
                                                 m_filter_desc,
                                                 m_filter_weights_gradient_d[i]));
      CHECK_CUDNN(cudnnConvolutionBackwardData(this->m_cudnn->get_handle(i),
                                               &one,
                                               m_filter_desc,
                                               m_filter_weights_d[i],
                                               this->m_neurons_cudnn_desc,
                                               this->m_prev_error_signal_d[i],
                                               m_convolution_desc,
                                               m_backward_data_algo,
                                               this->m_cudnn->get_work_space(i),
                                               this->m_cudnn->get_work_space_size(i),
                                               &zero,
                                               this->m_prev_neurons_cudnn_desc,
                                               this->m_error_signal_d[i]));

    }

    // Transfer outputs from GPUs to CPU and reduce
    Mat filter_weights_gradient_per_gpu
      = m_weights_gradient_per_gpu.Matrix()(El::IR(0, m_weights_gradient_per_gpu.Height()-1), El::ALL);
    Mat bias_weights_gradient_per_gpu
      = m_weights_gradient_per_gpu.Matrix()(El::IR(m_weights_gradient_per_gpu.Height()-1), El::ALL);
    this->m_cudnn->gather_from_gpus(filter_weights_gradient_per_gpu,
                                    m_filter_weights_gradient_d,
                                    m_filter_weights_gradient_v->Width());
    if(m_bias_scaling_factor != DataType(0)) {
      this->m_cudnn->gather_from_gpus(bias_weights_gradient_per_gpu,
                                      m_bias_weights_gradient_d,
                                      m_bias_weights_gradient_v->Width());
    }
    else {
      El::Zero(bias_weights_gradient_per_gpu);
    }
    El::Zero(*this->m_weights_gradient);
    this->m_cudnn->synchronize();
    for(int i=0; i<num_gpus; ++i) {
      const El::Int col_start = i * this->m_weights_gradient->Width();
      const El::Int col_end = (i+1) * this->m_weights_gradient->Width();
      *this->m_weights_gradient
        += m_weights_gradient_per_gpu(El::ALL, El::IR(col_start, col_end));
    }
    *this->m_weights_gradient *= DataType(1) /
      this->m_neural_network_model->get_effective_mini_batch_size();
    El::AllReduce(*this->m_weights_gradient,
                  this->m_weights_gradient->RedundantComm());

  #endif // #ifndef __LIB_CUDNN
  }

  /// Convolution forward propagation with im2col GEMM algorithm
  void fp_compute_im2col() {

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    const Mat& filter_weights_local = this->m_filter_weights_v->LockedMatrix();
    const Mat& bias_weights_local = this->m_bias_weights_v->LockedMatrix();
    Mat& activations_local = this->m_activations_v->Matrix();

    int num_output_channels;
    int num_input_channels;
    // Input, output, and filter entries are divided amongst channels
    if(reverse_dimensions()) {
      num_output_channels = this->m_prev_neuron_dims[0];
      num_input_channels = this->m_neuron_dims[0];
    }else {
      num_input_channels = this->m_prev_neuron_dims[0];
      num_output_channels = this->m_neuron_dims[0];
    }
    const int num_per_output_channel = this->m_num_neurons / num_output_channels;
    const int current_filter_size = m_conv_size / num_output_channels;

    // Apply bias
    for(int i=0; i<num_output_channels; ++i) {
      Mat activations_channel
        = El::View(activations_local,
                   El::IR(i*num_per_output_channel, (i+1)*num_per_output_channel),
                   El::ALL);
      Fill(activations_channel,
           m_bias_scaling_factor * bias_weights_local(0,i));
    }

    // Initialize im2col matrix
    Mat im2col_mat(current_filter_size, num_per_output_channel);

    // Iterate through data samples
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {

      // Construct im2col matrix from input
      const Mat input_mat = El::LockedView(prev_activations_local, El::ALL, El::IR(sample));
      im2col(input_mat,
             im2col_mat,
             num_input_channels,
             this->m_num_prev_neuron_dims - 1,
             this->m_prev_neuron_dims.data() + 1,
             m_conv_pads.data(),
             m_conv_dims.data(),
             m_conv_strides.data());

      // Apply convolution to current data sample
      Mat output_mat(num_per_output_channel, num_output_channels,
                     activations_local.Buffer(0,sample), num_per_output_channel);
      El::Gemm(TRANSPOSE, NORMAL,
           DataType(1), im2col_mat, filter_weights_local,
           DataType(1), output_mat);

    }

  }

  /// Convolution backward propagation with im2col GEMM algorithm
  void bp_compute_im2col() {

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    const Mat& filter_weights_local = this->m_filter_weights_v->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal_v->LockedMatrix();
    Mat& filter_weights_gradient_local = this->m_filter_weights_gradient_v->Matrix();
    Mat& bias_weights_gradient_local = this->m_bias_weights_gradient_v->Matrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();

    // Initialize weight gradients to zero
    Zero(*this->m_weights_gradient);

    int num_output_channels;
    int num_input_channels;
    // Input, output, and filter entries are divided amongst channels
    if(reverse_dimensions()){
      num_output_channels = this->m_prev_neuron_dims[0];
      num_input_channels = this->m_neuron_dims[0];
    }else{
      num_input_channels = this->m_prev_neuron_dims[0];
      num_output_channels = this->m_neuron_dims[0];
    }
    const int num_per_output_channel = this->m_num_neurons / num_output_channels;
    const int current_filter_size = m_conv_size / num_output_channels;

    // Compute bias gradient
    #pragma omp parallel for
    for(int output_channel = 0;
        output_channel < num_output_channels;
        ++output_channel) {
      DataType& bias_weights_gradient_entry
        = bias_weights_gradient_local(0, output_channel);
      for(int col = 0; col < prev_error_signal_local.Width(); ++col) {
        for(int row = output_channel * num_per_output_channel;
            row < (output_channel+1) * num_per_output_channel;
            ++row) {
          bias_weights_gradient_entry += prev_error_signal_local(row, col);
        }
      }
      bias_weights_gradient_entry *= m_bias_scaling_factor;
    }

    // Initialize filter and im2col matrices
    Mat im2col_mat(current_filter_size, num_per_output_channel);

    // Iterate through data samples
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {

      // Reshape previous error signal into matrix
      const Mat prev_error_signal_mat(num_per_output_channel,
                                      num_output_channels,
                                      prev_error_signal_local.LockedBuffer(0,sample),
                                      num_per_output_channel);

      // Compute gradient w.r.t. input im2col matrix
      Gemm(NORMAL, TRANSPOSE,
           DataType(1), filter_weights_local, prev_error_signal_mat,
           DataType(0), im2col_mat);

      // Compute error signal (i.e. gradient w.r.t. input)
      Mat output_mat = El::View(error_signal_local, El::ALL, El::IR(sample));
      col2im(im2col_mat,
             output_mat,
             num_input_channels,
             this->m_num_prev_neuron_dims - 1,
             this->m_prev_neuron_dims.data() + 1,
             m_conv_pads.data(),
             m_conv_dims.data(),
             m_conv_strides.data());

      // Construct im2col matrix from input
      const Mat input_mat = El::LockedView(prev_activations_local,
                                           El::ALL, El::IR(sample));
      im2col(input_mat,
             im2col_mat,
             num_input_channels,
             this->m_num_prev_neuron_dims - 1,
             this->m_prev_neuron_dims.data() + 1,
             m_conv_pads.data(),
             m_conv_dims.data(),
             m_conv_strides.data());

      // Compute gradient w.r.t. filter
      Gemm(NORMAL, NORMAL,
           DataType(1), im2col_mat, prev_error_signal_mat,
           DataType(0), filter_weights_gradient_local);

    }

    // Scale and accumulate gradients
    *this->m_weights_gradient *= DataType(1) /
      this->m_neural_network_model->get_effective_mini_batch_size();
    El::AllReduce(*this->m_weights_gradient,
                  this->m_weights_gradient->RedundantComm());

  }

 public:

  /// Update convolution filters and biases
  bool update_compute() {
    if(this->m_execution_mode == execution_mode::training) {
      this->l2_regularize();
      this->m_optimizer->update(this->m_weights_gradient);
    }
    return true;
  }

};
}

#endif // LBANN_LAYER_BASE_CONVOLUTION_HPP_INCLUDED