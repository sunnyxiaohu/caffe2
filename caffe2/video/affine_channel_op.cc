#include "caffe2/video/affine_channel_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(AffineChannel,
                      AffineChannelOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(AffineChannelGradient,
                      AffineChannelGradientOp<float, CPUContext>);

// Input: X, scale, bias; Output: Y
OPERATOR_SCHEMA(AffineChannel)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});
// Input: scale, dY; Output: dX
OPERATOR_SCHEMA(AffineChannelGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}});

class GetAffineChannelGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AffineChannelGradient", "",
        vector<string>{I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(AffineChannel, GetAffineChannelGradient);

} // namespace caffe2
