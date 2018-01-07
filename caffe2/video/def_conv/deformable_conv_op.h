#ifndef DEFORMABLE_CONV_OP_H_
#define DEFORMABLE_CONV_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

#include "caffe2/video/deformable_im2col.h"

#include <vector>

namespace caffe2 {

template <typename T, class Context>
class DeformableConvOp final : public Operator<Context> {
 public:
  DeformableConvOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        kernel_size_(OperatorBase::GetSingleArgument<int>("kernel_size", 3)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // Lazy, not implementing the CPU version
    CAFFE_NOT_IMPLEMENTED;
    return true;
  }

  private:
   Tensor<Context> col_buffer_;
   Tensor<Context> bias_multiplier_;

   int num_kernels_im2col_;
   int num_kernels_col2im_;
   int conv_out_channels_;
   int conv_in_channels_;
   int conv_out_spatial_dim_;
   int kernel_dim_;
   int kernel_size_;
   int col_offset_;
   int output_offset_;
   int input_offset_dim_;
   int deformable_group_;

};

template <typename T, class Context>
class DeformableConvGradientOp final : public Operator<Context> {
 public:
  DeformableConvGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        kernel_size_(OperatorBase::GetSingleArgument<int>("kernel_size", 3)){}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // Lazy, not implementing the CPU version
    CAFFE_NOT_IMPLEMENTED;
    return true;
  }

  private:
   Tensor<Context> col_buffer_;
   Tensor<Context> bias_multiplier_;

   int num_kernels_im2col_;
   int num_kernels_col2im_;
   int conv_out_channels_;
   int conv_in_channels_;
   int conv_out_spatial_dim_;
   int kernel_dim_;
   int kernel_size_;
   int col_offset_;
   int output_offset_;
   int input_offset_dim_;
   int deformable_group_;

};

} // namespace caffe2

#endif // DEFORMABLE_CONV_OP_H_
