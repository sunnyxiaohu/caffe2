#include "caffe2/video/deformable_conv_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {


template <>
bool DeformableConvOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Offset = Input(1);
  auto& filter = Input(2); // channel_out, channel_in, kernel, kernel
  auto& bias = Input(3);
  auto* Y = Output(0);

  conv_in_channels_ = X.dim32(0);
  conv_out_channels_ = filter.dim32(0);
  conv_out_spatial_dim_ = X.dim32(2) * X.dim32(3);
  kernel_dim_ = conv_in_channels_ * kernel_size_ * kernel_size_;
  weight_offset_ = conv_out_channels_ * kernel_dim_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_;
  input_offset_dim_ = Offset.dim32(1) * Offset.dim32(2) * Offset.dim32(3);
  int input_dim_ = X.dim32(1) * X.dim32(2) * X.dim32(3);
  int output_dim_ = conv_out_channels_ * X.dim32(2) * X.dim32(3);

  bias_multiplier_.Resize(
      vector<TIndex>{1, conv_out_spatial_dim_});
  math::Set<float, CUDAContext>(bias_multiplier_->size(), static_cast<T>(1.0), bias_multiplier_.mutable_data<float>, &context_);

  CAFFE_ENFORCE(filter.ndim() == 4, "filter must be 4D tensor");
  CAFFE_ENFORCE(
      filter.dim32(1) == conv_in_channels_,
      "filter number must be equal to input channel number");
  CAFFE_ENFORCE(
      filter.dim32(2) == kernel_size_,
      "filter height must be equal to kernel height");
  CAFFE_ENFORCE(
      filter.dim32(3) == kernel_size_,
      "filter width must be equal to kernel width");

  // same height x width for X and Y
  vector<TIndex> out_shape(X.dims());
  out_shape[2] = conv_out_channels_;
  Y->Resize(out_shape);

  const int channels_col = conv_in_channels_ * kernel_size_ * kernel_size_;
  const int height_col  = X.dim32(2);
  const int width_col   = X.dim32(3);
  const int num = X.dim32(0);
  const int pad_h = 0;
  const int pad_w = 0;
  const int stride_h = 1;
  const int stride_w = 1;
  const int dilation_h = 0;
  const int dilation_w = 0;
  const int deformable_group = 1;

  col_buffer_.Resize(
      vector<TIndex>{channels_col, height_col, width_col});

  for (auto n = 0; n < num; ++n)
  {
    deformable_im2col<float>(context_,
      X.data<float>() + n * input_dim_, //data_col
      Offset.data<float>() + n * input_offset_dim_,//offset
      X.dim32(1), X.dim32(2), X.dim32(3),
      channels_col, height_col, width_col,
      kernel_size_, kernel_size_,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, deformable_group,
      col_buffer_.mutable_data<float>());

    // weight

    math::Gemm<float, CUDAContext>(
        CblasNoTrans,
        CblasNoTrans,
        conv_out_channels_,
        conv_out_spatial_dim_,
        kernel_dim_,
        1,
        filter.data<float>(),
        col_buffer_.data<float>(),
        0,
        Y->mutable_data<float>() + n * output_dim_,
        &context_);

    // bias
    math::Gemm<float, CUDAContext>(
        CblasNoTrans,
        CblasNoTrans,
        conv_out_channels_,
        conv_out_spatial_dim_,
        1,
        1,
        bias.data<float>(),
        bias_multiplier_.mutable_data<float>(),
        1,
        Y->mutable_data<float>() + n * output_dim_,
        &context_);
  }


  return true;
}

template <>
bool DeformableConvGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Offset = Input(1);
  auto& filter = Input(2);
  auto& bias = Input(3);
  auto& dY = Input(4);

  auto* dX = Output(0);
  auto* dfilter = Output(1);
  auto* dbias = Output(2);


  conv_in_channels_ = X.dim32(0);
  conv_out_channels_ = filter.dim32(0);
  conv_out_spatial_dim_ = X.dim32(2) * X.dim32(3);
  kernel_dim_ = conv_in_channels_ * kernel_size_ * kernel_size_;
  weight_offset_ = conv_out_channels_ * kernel_dim_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_;
  input_offset_dim_ = Offset.dim32(1) * Offset.dim32(2) * Offset.dim32(3);
  int input_dim_ = X.dim32(1) * X.dim32(2) * X.dim32(3);
  int output_dim_ = conv_out_channels_ * X.dim32(2) * X.dim32(3);

  bias_multiplier_.Resize(
      vector<TIndex>{1, conv_out_spatial_dim_});
  math::Set<float, CUDAContext>(bias_multiplier_->size(), static_cast<T>(1.0), bias_multiplier_.mutable_data<float>, &context_);


  dX->ResizeLike(X);
  dfilter->ResizeLike(filter);
  dbias->ResizeLike(bias);

  const int channels_col = conv_in_channels_ * kernel_size_ * kernel_size_;
  const int height_col  = X.dim32(2);
  const int width_col   = X.dim32(3);
  const int num = X.dim32(0);
  const int pad_h = 0;
  const int pad_w = 0;
  const int stride_h = 1;
  const int stride_w = 1;
  const int dilation_h = 0;
  const int dilation_w = 0;
  const int deformable_group = 1;
  col_buffer_.Resize(
      vector<TIndex>{channels_col, height_col, width_col});

  math::Set<float, CUDAContext>(dfilter->size(), 0, dfilter->mutable_data<float>, &context_);
  math::Set<float, CUDAContext>(dbias->size(), 0, dbias->mutable_data<float>, &context_);

  for (auto n = 0; n < num; ++n)
  {
    // bp data
    math::Gemm<float, CUDAContext>(
        CblasTrans,
        CblasNoTrans,
        kernel_dim_,
        conv_out_spatial_dim_,
        conv_out_channels_,
        1,
        filter.data<float>(),
        dY.data<float>() + n * output_dim_,
        0,
        col_buffer_.mutable_data<float>(),
        &context_);

    // do not bp to coord
    // deformable_col2im_coord<float>(context_,
    //   col_buffer_.data<float>(),
    //   X.data<float>() + n * input_dim_,
    //   Offset.data<float>() + n * input_offset_dim_,
    //   X.dim32(1), X.dim32(2), X.dim32(3),
    //   channels_col, height_col, width_col,
    //   kernel_size_, kernel_size_,
    //   pad_h, pad_w, stride_h, stride_w,
    //   dilation_h, dilation_w, deformable_group,
    //   // dOffset.mutable_data<float>() + n * input_offset_dim_
    // );

    // bp for data
    deformable_col2im<float>(context_,
      col_buffer_.data<float>(),
      Offset.data<float>() + n * input_offset_dim_,
      X.dim32(1), X.dim32(2), X.dim32(3),
      channels_col, height_col, width_col,
      kernel_size_, kernel_size_,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, deformable_group,
      dX->mutable_data<float>() + n * input_dim_
    );

    // bp for parameters
    deformable_im2col<float>(context_,
      X.data<float>() + n * input_dim_, //data_col
      Offset.data<float>() + n * input_offset_dim_,//offset
      X.dim32(1), X.dim32(2), X.dim32(3),
      channels_col, height_col, width_col,
      kernel_size_, kernel_size_,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, deformable_group,
      col_buffer_.mutable_data<float>());

    // weights
    math::Gemm<float, CUDAContext>(
        CblasNoTrans,
        CblasTrans,
        conv_out_channels_,
        kernel_dim_,
        conv_out_spatial_dim_,
        1,
        dY.data<float>() + n * output_dim_,
        col_buffer_.data<float>(),
        1,
        dfilter->mutable_data<float>(),
        &context_);

    // bias
    math::Gemv<float, CUDAContext>(
        CblasNoTrans,
        conv_out_channels_,
        conv_out_spatial_dim_,
        1,
        dY.data<float>() + n * output_dim_,
        bias_multiplier_.data<float>(),
        1,
        dbias->mutable_data<float>(),
        &context_);
  }


  return true;
}

REGISTER_CUDA_OPERATOR(DeformableConv, DeformableConvOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    DeformableConvGradient,
    DeformableConvGradientOp<float, CUDAContext>);
} // namespace caffe2
