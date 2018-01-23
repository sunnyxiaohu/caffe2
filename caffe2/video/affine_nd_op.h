#ifndef AFFINE_ND_OP_H_
#define AFFINE_ND_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class AffineNdOp final : public Operator<Context> {
 public:
  AffineNdOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // Lazy, not implementing the CPU version
    CAFFE_NOT_IMPLEMENTED;
    return true;
  }
};

template <typename T, class Context>
class AffineNdGradientOp final : public Operator<Context> {
 public:
  AffineNdGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // Lazy, not implementing the CPU version
    CAFFE_NOT_IMPLEMENTED;
    return true;
  }
};

} // namespace caffe2

#endif // AFFINE_ND_OP_H_
