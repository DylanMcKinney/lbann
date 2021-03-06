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
// lbann_callback_imcomm .hpp .cpp - Send gradient updates between models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "lbann/callbacks/callback.hpp"
#include "lbann/utils/quantizer.hpp"

namespace lbann {

/**
 * Support inter-model communication after each mini-batch to synchronize
 * gradient updates.
 * This optionally supports quantizing the gradient updates before communication
 * in order to reduce bandwidth requirements.
 */
class lbann_callback_imcomm : public lbann_callback {
 public:
  using lbann_callback::on_backward_prop_end;

  enum comm_type {
    NONE,  /** Do no gradient updates. */
    NORMAL,  /** Simply sum gradient updates. */
    ONEBIT_QUANTIZATION,  /** Do one-bit quantization. */
    THRESH_QUANTIZATION,  /** Do threshold quantization. */
    ADAPTIVE_QUANTIZATION,  /** Do adaptive quantization. */
  };

  /**
   * Initialize with ct being used for all weights.
   */
  lbann_callback_imcomm(comm_type ct = NORMAL,
                        lbann_summary *summarizer = nullptr);
  lbann_callback_imcomm(const lbann_callback_imcomm&) = default;
  lbann_callback_imcomm& operator=(const lbann_callback_imcomm&) = default;
  lbann_callback_imcomm* copy() const override {
    return new lbann_callback_imcomm(*this);
  }
  /**
   * Convenience initialization to do one update type for specific weights.
   * Implies no inter-model updates for other weights.
   */
  lbann_callback_imcomm(comm_type ct, std::unordered_set<weights *> weights_list,
                        lbann_summary *summarizer = nullptr);

  /** Choose comm type ct for weights. */
  void set_weights_comm(weights *w, comm_type ct);
  /** Set weights to use adaptive quantization with proportion. */
  void set_weights_adaptive(weights *w, int proportion);
  /** Set weights to use threshold quantization with given thresholds. */
  void set_weights_threshold(weights *w,
                             DataType pos_thresh,
                             DataType neg_thresh);

  /** Do initialization for this model. */
  void setup(model *m) override;
  /** Make sure all models have the same weights. */
  void on_train_begin(model *m) override;
  /** Clear out remaining error if needed. */
  void on_epoch_end(model *m) override;
  /** Do inter-model gradient updates. */
  void on_backward_prop_end(model *m) override;

  std::string name() const override { return "imcomm"; }

 private:
  /** Parameters for a given set of weights. */
  struct imcomm_params {
    /** Type of communication done. */
    comm_type ct = NONE;
    /** Accumulated error (e.g. from quantization). */
    Mat error;
    /** If >0, reshape (local) gradients to these dimensions. */
    El::Int reshape_height = 0;
    El::Int reshape_width = 0;
    /** Adaptive quantization proportion. */
    int proportion = 32;
    /** Threshold quantization thresholds. */
    DataType pos_thresh = 1.0;
    DataType neg_thresh = -1.0;
  };
  /** Default communication type. */
  comm_type m_default_ct;
  /** Per-weights parameters. */
  std::unordered_map<weights *, imcomm_params> m_weights_params;
  /** Quantizer for quantization of updates, if needed. */
  lbann_quantizer m_quantizer;

  /** Return true if the comm type does quantization. */
  inline bool ct_does_quantization(comm_type ct) const {
    return (ct == ONEBIT_QUANTIZATION ||
            ct == THRESH_QUANTIZATION ||
            ct == ADAPTIVE_QUANTIZATION);
  }

  /** Return true if the comm type prefers reshaping. */
  inline bool ct_needs_reshape(comm_type ct) const {
    return ct_does_quantization(ct);  // Currently, all quantization reshapes.
  }

  /**
   * Get a matrix that reinterprets mat as being height x width.
   * Assumes that mat.Height()*mat.Width() == height*width.
   */
  void reshape_mat(Mat& mat, Mat& reshaped, El::Int height, El::Int width) {
    reshaped.Attach(height, width, mat.Buffer(), height);
  }

  /** Summarize relevant statistics. */
  void do_summary(model *m, weights *w, EvalType im_time);
};


/** returns a string representation of the weight_initialization */
std::string get_comm_type_name(lbann_callback_imcomm::comm_type m);

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED
