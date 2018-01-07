#ifndef MAXPOOL_CHANNEL_OP_H_
#define MAXPOOL_CHANNEL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include <algorithm>
#include <vector>


namespace caffe2 {

template <typename T, class Context>
class MaxpoolChannelOp final : public Operator<Context> {

 public:
  MaxpoolChannelOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        top_k_(OperatorBase::GetSingleArgument<int>("top_k", 10)),
        assigned_value_(OperatorBase::GetSingleArgument<float>("assigned_value", 0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  static int cmp_affinity(std::pair<float, int> &a, std::pair<float, int>  &b)
  {
    return a.first > b.first;
  }

  bool RunOnDevice() override {

    /*auto& X = Input(0);
    auto* Y = Output(0);
    auto* mask = Output(1);

    int batch_num = X.dim32(0);
    int row_num = X.dim32(1);
    int col_num = X.dim32(2);
    Y->Resize(X.dims());
    mask->Resize(X.dims());

    const T* x_data = X.template data<T>();
    T* mask_data = mask->template mutable_data<T>();
    T* Y_data = Y->template mutable_data<T>();

    vector<std::pair<float, int> > vec;
    for(int i = 0; i < col_num; i ++) vec.push_back(std::make_pair(0.0, 0) );

    for (int i = 0; i < batch_num; i ++)
    {
      for (int h = 0; h < row_num; h ++)
      {
        for (int w = 0; w < col_num; w ++)
        {
          int idx = i * row_num * col_num + h * col_num + w;
          vec[w].first = x_data[idx];
          vec[w].second = w;
        }
        sort(vec.begin(), vec.end(), cmp_affinity);
        for (int w = 0; w < col_num; w ++)
        {
          int idx = i * row_num * col_num + h * col_num + w;
          mask_data[idx] = 0;
        }
        for (int k = 0; k < top_k_; k ++)
        {
          int idx = i * row_num * col_num + h * col_num + vec[k].second ;
          mask_data[idx] = 1;
        }
      }

      for (int h = 0; h < row_num; h ++)
      {
        for (int w = 0; w < col_num; w ++)
        {
          int idx = i * row_num * col_num + h * col_num + w;
          Y_data[idx] = x_data[idx] * mask_data[idx];
        }
      }
    }*/
    CAFFE_NOT_IMPLEMENTED;

    return true;
  }

protected:
  int top_k_;
  int assigned_value_;

};

template <typename T, class Context>
class MaxpoolChannelGradientOp final : public Operator<Context> {
 public:
  MaxpoolChannelGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {

    /*auto& mask = Input(0);
    auto& dY = Input(1);
    auto* dX = Output(0);

    int batch_num = dY.dim32(0);
    int row_num = dY.dim32(1);
    int col_num = dY.dim32(2);

    dX->Resize(dY.dims());
    const T* dy_data = dY.template data<T>();
    const T* mask_data = mask.template data<T>();
    T* dx_data = dX->template mutable_data<T>();

    for (int i = 0; i < batch_num; i ++)
    {
      for (int h = 0; h < row_num; h ++)
      {
        for (int w = 0; w < col_num; w ++)
        {
          int idx = i * row_num * col_num + h * col_num + w;
          dx_data[idx] = dy_data[idx] * mask_data[idx];
        }
      }
    }*/
    CAFFE_NOT_IMPLEMENTED;

    return true;
  }

};

} // namespace caffe2

#endif
