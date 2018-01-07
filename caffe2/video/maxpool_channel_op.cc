#include "caffe2/video/maxpool_channel_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(MaxpoolChannel,
                      MaxpoolChannelOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MaxpoolChannelGradient,
                      MaxpoolChannelGradientOp<float, CPUContext>);

// Input: X; Output: Y, mask
OPERATOR_SCHEMA(MaxpoolChannel)
  .NumInputs(1)
  .NumOutputs(2)
  .AllowInplace({{0, 0}})
  // only allow inplace in forward;
  // it should report error if this layer has grad
  .Arg("top_k",
    "select the top k scores.")
  .Arg("assigned_value",
      "assign default value to Y.");
// Input:  mask, dY; Output: dX
OPERATOR_SCHEMA(MaxpoolChannelGradient)
  .NumInputs(2)
  .NumOutputs(1);

class GetMaxpoolChannelGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // int top_k = 10;
    // if (HasArgument(def_, "top_k")) {
    //   const auto& arg = GetArgument(def_, "top_k");
    //   CAFFE_ENFORCE(arg.has_i());
    //   top_k = arg.i();
    // }
    return SingleGradientDef(
        "MaxpoolChannelGradient", "",
        vector<string>{O(1), GO(0)},
        vector<string>{GI(0)});

  }
};

REGISTER_GRADIENT(MaxpoolChannel, GetMaxpoolChannelGradient);

} // namespace
} // namespace caffe2
