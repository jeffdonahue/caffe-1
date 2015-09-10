#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype tanh(Dtype x) {
  return 2. * sigmoid(2. * x) - 1.;
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->blobs_.size() > 0) {
    LOG(INFO) << this->layer_param_.name()
        << " Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> weight_shape(2);
    hidden_dim_ = bottom[0]->shape(2);
    weight_shape[0] = 4 * hidden_dim_;
    weight_shape[1] = hidden_dim_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }
}
template <typename Dtype>
void LSTMUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < MinBottomBlobs(); ++i) {
    if (i == 3)
      CHECK_EQ(2, bottom[i]->num_axes());
    else
	  CHECK_EQ(3, bottom[i]->num_axes());
      CHECK_EQ(1, bottom[i]->shape(0));
  }
  const int num_instances = bottom[0]->shape(1);
  hidden_dim_ = bottom[0]->shape(2);
  CHECK_EQ(num_instances, bottom[1]->shape(1));
  CHECK_EQ(hidden_dim_, bottom[1]->shape(2));
  CHECK_EQ(num_instances, bottom[2]->shape(1));
  CHECK_EQ(4 * hidden_dim_, bottom[2]->shape(2));
  CHECK_EQ(1, bottom[3]->shape(0));
  CHECK_EQ(num_instances, bottom[3]->shape(1));
  if (bottom.size() > 4) {
    CHECK_EQ(2, bottom[4]->num_axes());
    CHECK_EQ(num_instances, bottom[4]->shape(0));
    CHECK_EQ(4 * hidden_dim_, bottom[4]->shape(1));
  }
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[1]);
  X_acts_.ReshapeLike(*bottom[2]);
  this->param_propagate_down_.resize(1, true);
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const int num = bottom[0]->shape(1);
  const int x_dim = hidden_dim_ * 4;
  const Dtype* C_prev = bottom[0]->cpu_data();
  const Dtype* H_prev = bottom[1]->cpu_data();
  Dtype* X            = bottom[2]->mutable_cpu_data();
  const Dtype* flush  = bottom[3]->cpu_data();
  Dtype* C = top[0]->mutable_cpu_data();
  Dtype* H = top[1]->mutable_cpu_data();
  Dtype *buffer_h_ptr = bottom[1]->mutable_cpu_diff();
  caffe_copy(bottom[1]->count(), H_prev, buffer_h_ptr);

  // H_prev(flush <= 0) = 0
  for (int n = 0; n < num; ++n) {
    if (flush[n] <= 0)
      caffe_set(hidden_dim_, Dtype(0), buffer_h_ptr);
    buffer_h_ptr += hidden_dim_;
  }

  // X = W_h * H_prev + X
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, x_dim, hidden_dim_,
    (Dtype)1., bottom[1]->cpu_diff(), this->blobs_[0]->cpu_data(),
    (Dtype)1., X);
  if (bottom.size() > 4)
    caffe_add(x_dim * num, X, bottom[4]->cpu_data(), X);
  for (int n = 0; n < num; ++n) {
    for (int d = 0; d < hidden_dim_; ++d) {
      const Dtype i = sigmoid(X[d]);
      const Dtype f = flush[n] <= 0 ? 0 :
          sigmoid(X[1 * hidden_dim_ + d]);
      const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
      const Dtype g = tanh(X[3 * hidden_dim_ + d]);
      const Dtype c_prev = C_prev[d];
      const Dtype c = f * c_prev + i * g;
      C[d] = c;
      const Dtype tanh_c = tanh(c);
      H[d] = o * tanh_c;
    }
    C_prev += hidden_dim_;
    X += x_dim;
    C += hidden_dim_;
    H += hidden_dim_;
  }
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[3]) << "Cannot backpropagate to sequence indicators.";
  if (!propagate_down[0] && !propagate_down[1] && !propagate_down[2] ) return;

  const int num = bottom[0]->shape(1);
  const int x_dim = hidden_dim_ * 4;
  const Dtype* C_prev = bottom[0]->cpu_data();
  const Dtype* X = bottom[2]->cpu_data();
  const Dtype* flush = bottom[3]->cpu_data();
  const Dtype* C = top[0]->cpu_data();
  const Dtype* H = top[1]->cpu_data();
  const Dtype* C_diff = top[0]->cpu_diff();
  const Dtype* H_diff = top[1]->cpu_diff();
  Dtype* C_prev_diff = bottom[0]->mutable_cpu_diff();
  Dtype* X_diff = bottom[2]->mutable_cpu_diff();
  for (int n = 0; n < num; ++n) {
    for (int d = 0; d < hidden_dim_; ++d) {
      const Dtype i = sigmoid(X[d]);
      const Dtype f = flush[n] <= 0 ? 0 :
          sigmoid(X[1 * hidden_dim_ + d]);
      const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
      const Dtype g = tanh(X[3 * hidden_dim_ + d]);
      const Dtype c_prev = C_prev[d];
      const Dtype c = C[d];
      const Dtype tanh_c = tanh(c);
      Dtype* c_prev_diff = C_prev_diff + d;
      Dtype* i_diff = X_diff + d;
      Dtype* f_diff = X_diff + 1 * hidden_dim_ + d;
      Dtype* o_diff = X_diff + 2 * hidden_dim_ + d;
      Dtype* g_diff = X_diff + 3 * hidden_dim_ + d;
      const Dtype c_term_diff =
          C_diff[d] + H_diff[d] * o * (1 - tanh_c * tanh_c);
      *c_prev_diff = c_term_diff * f;
      *i_diff = c_term_diff * g * i * (1 - i);
      *f_diff = c_term_diff * c_prev * f * (1 - f);
      *o_diff = H_diff[d] * tanh_c * o * (1 - o);
      *g_diff = c_term_diff * i * (1 - g * g);
    }
    C_prev += hidden_dim_;
    X += x_dim;
    C += hidden_dim_;
    H += hidden_dim_;
    C_diff += hidden_dim_;
    H_diff += hidden_dim_;
    X_diff += x_dim;
    C_prev_diff += hidden_dim_;
  }
  X_diff = bottom[2]->mutable_cpu_diff();
  if (bottom.size() > 4)
    caffe_copy(num * x_dim, X_diff, bottom[4]->mutable_cpu_diff());
  Dtype * H_prev_diff = bottom[1]->mutable_cpu_diff();
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, x_dim, hidden_dim_, num,
    (Dtype)1., X_diff, H_prev_diff,
    (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, hidden_dim_, x_dim,
    (Dtype)1., X_diff, this->blobs_[0]->cpu_data(),
    (Dtype)0., H_prev_diff);
  for (int n = 0; n < num; ++n) {
    if (flush[n] <= 0)
      caffe_set(hidden_dim_, Dtype(0), H_prev_diff);
    H_prev_diff += hidden_dim_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(LSTMUnitLayer);
#endif

INSTANTIATE_CLASS(LSTMUnitLayer);
REGISTER_LAYER_CLASS(LSTMUnit);

}  // namespace caffe
