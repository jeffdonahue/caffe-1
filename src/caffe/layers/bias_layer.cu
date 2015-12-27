#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BiasLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bias = this->blobs_[0]->gpu_data();
  Dtype* output = top[0]->mutable_gpu_data();
  if (bottom[0] != top[0]) {
    const Dtype* input = bottom[0]->gpu_data();
    caffe_copy(bottom[0]->count(), input, output);
  }
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, channels_,
      bias_multiplier_.count(), Dtype(1), Dtype(1), bias,
      bias_multiplier_.gpu_data(), Dtype(1), output);
}

template <typename Dtype>
void BiasLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    const Dtype* input = top[0]->gpu_diff();
    Dtype* output = bottom[0]->mutable_gpu_diff();
    caffe_copy(bottom[0]->count(), input, output);
  }
  // in-place; we don't need to do anything with the data diff
  if (this->param_propagate_down_[0]) {
    Dtype* bias = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* input = top[0]->gpu_diff();
    caffe_gpu_gemv(CblasNoTrans, channels_, bias_multiplier_.count(), Dtype(1),
        input, bias_multiplier_.gpu_data(), Dtype(1), bias);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BiasLayer);

}  // namespace caffe
