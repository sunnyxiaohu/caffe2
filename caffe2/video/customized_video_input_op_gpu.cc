#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/video/customized_video_input_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
  CustomizedVideoInput, CustomizedVideoInputOp<CUDAContext>);
} // namespace caffe2
