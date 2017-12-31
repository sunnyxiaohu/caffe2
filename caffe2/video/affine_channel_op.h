#ifndef AFFINE_CHANNEL_OP_H_
#define AFFINE_CHANNEL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class AffineChannelOp final : public Operator<Context> {
 public:
  AffineChannelOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // Lazy, not implementing the CPU version
    CAFFE_NOT_IMPLEMENTED;
    return true;
  }
};

template <typename T, class Context>
class AffineChannelGradientOp final : public Operator<Context> {
 public:
  AffineChannelGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // Lazy, not implementing the CPU version
    CAFFE_NOT_IMPLEMENTED;
    return true;
  }
};

} // namespace caffe2

#endif // AFFINE_CHANNEL_OP_H_
