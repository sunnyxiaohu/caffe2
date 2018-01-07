#include "caffe2/video/maxpool_channel_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void SelectTopK(
    const int n,
    const int col_num,
    const int top_k,
    const float assigned_value,
    const T* x_data,
    T* mask_data,
    T* y_data) {
  CUDA_1D_KERNEL_LOOP(index, n) {

    int offset = index * col_num;
    for (int i = 0; i < col_num; i ++)
    {
      mask_data[offset + i] = 0;
      y_data[offset + i] = assigned_value;
    }
    for (int i = 0; i < top_k; i ++)
    {
      float max_num = -1e6;
      int max_id = 0;
      for (int j = 0; j < col_num; j ++)
      {
        if (max_num < x_data[offset + j] && mask_data[offset + j] < 0.5)
        {
          max_num = x_data[offset + j];
          max_id = j;
        }
      }
      // printf("%d ", max_id);
      mask_data[offset + max_id] = 1;
      y_data[offset + max_id] = x_data[offset + max_id];
    }

  }
}

template <typename T>
__global__ void AssignGradMask(
    const int n,
    const T* dy_data,
    const T* mask_data,
    T* dx_data) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    dx_data[index] = dy_data[index] * mask_data[index];
  }
}


} // namespace

template<>
bool MaxpoolChannelOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  auto* mask = Output(1);

  int batch_num = 1;
  int row_num = 1;
  int col_num = 1;

  if (X.ndim() == 3)
  {
    batch_num = X.dim32(0);
    row_num = X.dim32(1);
    col_num = X.dim32(2);
  }
  else
  {
    batch_num = 1;
    row_num = X.dim32(0);
    col_num = X.dim32(1);
  }

  Y->Resize(X.dims());
  mask->Resize(X.dims());
  // printf("got in fp");

  SelectTopK<float><<<CAFFE_GET_BLOCKS(batch_num * row_num),
                            CAFFE_CUDA_NUM_THREADS,
                            0, context_.cuda_stream()>>>(
      batch_num * row_num, col_num, top_k_, assigned_value_, X.data<float>(),
      mask->mutable_data<float>(), Y->mutable_data<float>());

  return true;
}

template<>
bool MaxpoolChannelGradientOp<float, CUDAContext>::RunOnDevice() {

  auto& mask = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);


  int batch_num = 1;
  int row_num = 1;
  int col_num = 1;

  if (dY.ndim() == 3)
  {
    batch_num = dY.dim32(0);
    row_num = dY.dim32(1);
    col_num = dY.dim32(2);
  }
  else
  {
    batch_num = 1;
    row_num = dY.dim32(0);
    col_num = dY.dim32(1);
  }

  dX->Resize(dY.dims());

  // printf("got in bp");

  const int output_size = dY.size();
  AssignGradMask<float><<<CAFFE_GET_BLOCKS(output_size),
                            CAFFE_CUDA_NUM_THREADS,
                            0, context_.cuda_stream()>>>(
      output_size, dY.data<float>(),
      mask.data<float>(), dX->mutable_data<float>());

  return true;
}


REGISTER_CUDA_OPERATOR(MaxpoolChannel,
                       MaxpoolChannelOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MaxpoolChannelGradient,
                       MaxpoolChannelGradientOp<float, CUDAContext>);


} // namespace caffe2
