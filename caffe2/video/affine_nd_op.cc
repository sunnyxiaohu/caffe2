#include "caffe2/video/affine_nd_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(AffineNd,
                      AffineNdOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(AffineNdGradient,
                      AffineNdGradientOp<float, CPUContext>);

// Input: X, scale, bias; Output: Y
OPERATOR_SCHEMA(AffineNd)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});
// Input: scale, dY; Output: dX
OPERATOR_SCHEMA(AffineNdGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}});

class GetAffineNdGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AffineNdGradient", "",
        vector<string>{I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(AffineNd, GetAffineNdGradient);

} // namespace caffe2
