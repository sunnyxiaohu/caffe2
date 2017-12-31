#include "caffe2/video/affine_channel_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void ScaleBiasForward(
    const int n,
    const T* in,
    const T* scale,
    const T* bias,
    const int scale_dim,
    const int hxw_dim,
    T* out) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int scale_index = (index / hxw_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index] + bias[scale_index];
  }
}

template <typename T>
__global__ void ScaleForward(
    const int n,
    const T* in,
    const T* scale,
    const int scale_dim,
    const int hxw_dim,
    T* out) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int scale_index = (index / hxw_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index];
  }
}
} // namespace

template <>
bool AffineChannelOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& scale = Input(1);
  auto& bias = Input(2);
  auto* Y = Output(0);

  Y->ResizeLike(X);
  const int output_size = Y->size();
  ScaleBiasForward<float>
      <<<CAFFE_GET_BLOCKS(output_size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          output_size,
          X.data<float>(),
          scale.data<float>(),
          bias.data<float>(),
          X.dim32(1),
          X.size() / X.dim32(0) / X.dim32(1),  // support TxHxW
          Y->mutable_data<float>());
  return true;
}

template <>
bool AffineChannelGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& scale = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);

  dX->ResizeLike(dY);
  ScaleForward<float>
      <<<CAFFE_GET_BLOCKS(dY.size()),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          dY.size(),
          dY.data<float>(),
          scale.data<float>(),
          dY.dim32(1),
          dY.size() / dY.dim32(0) / dY.dim32(1),  // support TxHxW
          dX->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(AffineChannel, AffineChannelOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    AffineChannelGradient,
    AffineChannelGradientOp<float, CUDAContext>);
} // namespace caffe2
