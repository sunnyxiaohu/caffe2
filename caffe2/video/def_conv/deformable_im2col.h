#ifndef DEFORMABLE_IM2COL_H_
#define DEFORMABLE_IM2COL_H_
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

#include <cstring>
#include <vector>

namespace caffe2 {
namespace{
template <typename Dtype>
void deformable_im2col(CUDAContext& context,
  const Dtype* data_im, const Dtype* data_offset, const int channels,
	const int height, const int width,
  const int channels_col, const int height_col, const int width_col,
  const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int deformable_group,
	Dtype* data_col);

void ooop();

template <typename Dtype>
void deformable_col2im(CUDAContext& context,
    const Dtype* data_col, const Dtype* data_offset,
    const int channels, const int height, const int width,
    const int channels_col, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int deformable_group,
    Dtype* grad_im);

template <typename Dtype>
void deformable_col2im_coord(CUDAContext& context,
    const Dtype* data_col, const Dtype* data_im, const Dtype* data_offset, const int channels,
    const int height, const int width,
    const int channels_col, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int deformable_group, Dtype* grad_offset);
}  // namespace
}  // namespace caffe2


#endif  // DEFORMABLE_IM2COL_H_
