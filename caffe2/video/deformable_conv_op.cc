#include "caffe2/video/deformable_conv_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(DeformableConv,
                      DeformableConvOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(DeformableConvGradient,
                      DeformableConvGradientOp<float, CPUContext>);

// Input: X, Offset, Weight, Bias; Output: Y
OPERATOR_SCHEMA(DeformableConv)
    .NumInputs(4)
    .NumOutputs(1);
// Input: dY; Output: dX, dWeight, dBias
OPERATOR_SCHEMA(DeformableConvGradient)
    .NumInputs(1)
    .NumOutputs(3);

class GetDeformableConvGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "DeformableConvGradient", "",
        vector<string>{GO(0)},
        vector<string>{GI(0), GI(2), GI(3)});
  }
};

REGISTER_GRADIENT(DeformableConv, GetDeformableConvGradient);

} // namespace caffe2
