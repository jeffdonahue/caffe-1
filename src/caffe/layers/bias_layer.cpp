#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BiasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, channels_));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().bias_filler()));
    bias_filler->Fill(this->blobs_[0].get());
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BiasLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), channels_);
  top[0]->ReshapeLike(*bottom[0]);
  bias_multiplier_.Reshape(1, 1, 1, top[0]->height() * top[0]->width());
  if (bias_multiplier_.cpu_data()[bias_multiplier_.count() - 1] != Dtype(1)) {
    caffe_set(bias_multiplier_.count(), Dtype(1),
              bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bias = this->blobs_[0]->cpu_data();
  Dtype* output = top[0]->mutable_cpu_data();
  if (bottom[0] != top[0]) {
    const Dtype* input = bottom[0]->cpu_data();
    caffe_copy(bottom[0]->count(), input, output);
  }
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels_,
      bias_multiplier_.count(), Dtype(1), Dtype(1), bias,
      bias_multiplier_.cpu_data(), Dtype(1), output);
}

template <typename Dtype>
void BiasLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    const Dtype* input = top[0]->cpu_diff();
    Dtype* output = bottom[0]->mutable_cpu_diff();
    caffe_copy(bottom[0]->count(), input, output);
  }
  // in-place, we don't need to do anything with the data diff
  if (this->param_propagate_down_[0]) {
    Dtype* bias = this->blobs_[0]->mutable_cpu_diff();
    const Dtype* input = top[0]->cpu_diff();
    caffe_cpu_gemv(CblasNoTrans, channels_, bias_multiplier_.count(), Dtype(1),
        input, bias_multiplier_.cpu_data(), Dtype(1), bias);
  }
}

#ifdef CPU_ONLY
STUB_GPU(BiasLayer);
#endif

INSTANTIATE_CLASS(BiasLayer);
REGISTER_LAYER_CLASS(Bias);

}  // namespace caffe
