#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype sigmoid(const Dtype x) {
  return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename Dtype>
__device__ Dtype tanh(const Dtype x) {
  return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
}

template <typename Dtype>
__global__ void LSTMActsForward(const int nthreads, const int dim,
                                const Dtype* X, Dtype* X_acts) {
  const int x_dim = 4 * dim;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % x_dim;
    if (d < 3 * dim) {
      X_acts[index] = sigmoid(X[index]);
    } else {
      X_acts[index] = tanh(X[index]);
    }
  }
}

template <typename Dtype>
__global__ void LSTMUnitForward(const int nthreads, const int dim,
    const Dtype* C_prev, const Dtype* X, const Dtype* flush,
    Dtype* C, Dtype* H) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int d = index % dim;
    const Dtype* X_offset = X + 4 * dim * n;
    const Dtype i = X_offset[d];
    const Dtype f = X_offset[1 * dim + d];
    const Dtype o = X_offset[2 * dim + d];
    const Dtype g = X_offset[3 * dim + d];
    const Dtype c_prev = C_prev[index];
    const Dtype c = (flush[n] <= 0? 0 : (f * c_prev) ) + i * g;
    C[index] = c;
    const Dtype tanh_c = tanh(c);
    H[index] = o * tanh_c;
  }
}

template <typename Dtype>
__global__ void indicator(const int count, const int dim,
    const Dtype *flush, Dtype *dst) {
  CUDA_KERNEL_LOOP(index, count) {
    const int n = index / dim;
    if (flush[n] <= 0) dst[index] = 0;
  }
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = top[1]->count();
  const int num = bottom[0]->shape(1);
  const int x_dim = hidden_dim_ * 4;
  const Dtype* C_prev = bottom[0]->gpu_data();
  const Dtype* H_prev = bottom[1]->gpu_data();
  Dtype* X            = bottom[2]->mutable_gpu_data();
  const Dtype* flush  = bottom[3]->gpu_data();
  Dtype* X_acts = X_acts_.mutable_gpu_data();
  Dtype* C = top[0]->mutable_gpu_data();
  Dtype* H = top[1]->mutable_gpu_data();
  Dtype *buffer_h_ptr = bottom[1]->mutable_gpu_diff();
  caffe_copy(count, H_prev, buffer_h_ptr);
  // H_prev(flush <= 0) = 0
  // NOLINT_NEXT_LINE(whitespace/operators)
  indicator<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, hidden_dim_, flush, buffer_h_ptr);
  // X = W_h * H_prev + X
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, x_dim, hidden_dim_,
     (Dtype)1., buffer_h_ptr, this->blobs_[0]->gpu_data(), (Dtype)1., X);
  flush = bottom[3]->gpu_data();
  const int X_count = bottom[2]->count();
  if (bottom.size() > 4)
    caffe_gpu_add(X_count, X, bottom[4]->gpu_data(), X);
  // NOLINT_NEXT_LINE(whitespace/operators)
  LSTMActsForward<Dtype>
      <<<CAFFE_GET_BLOCKS(X_count), CAFFE_CUDA_NUM_THREADS>>>(
      X_count, hidden_dim_, X, X_acts);
  CUDA_POST_KERNEL_CHECK;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LSTMUnitForward<Dtype>
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, hidden_dim_, C_prev, X_acts, flush, C, H);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void LSTMUnitBackward(const int nthreads, const int dim,
    const Dtype* C_prev, const Dtype* X, const Dtype* C, const Dtype* H,
    const Dtype* flush, const Dtype* C_diff, const Dtype* H_diff,
    Dtype* C_prev_diff, Dtype* X_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int d = index % dim;
    const Dtype* X_offset = X + 4 * dim * n;
    const Dtype i = X_offset[d];
    const Dtype f = X_offset[1 * dim + d];
    const Dtype o = X_offset[2 * dim + d];
    const Dtype g = X_offset[3 * dim + d];
    const Dtype c_prev = C_prev[index];
    const Dtype c = C[index];
    const Dtype tanh_c = tanh(c);
    Dtype* c_prev_diff = C_prev_diff + index;
    Dtype* X_diff_offset = X_diff + 4 * dim * n;
    Dtype* i_diff = X_diff_offset + d;
    Dtype* f_diff = X_diff_offset + 1 * dim + d;
    Dtype* o_diff = X_diff_offset + 2 * dim + d;
    Dtype* g_diff = X_diff_offset + 3 * dim + d;
    const Dtype c_term_diff =
        C_diff[index] + H_diff[index] * o * (1 - tanh_c * tanh_c);
    const Dtype flush_n = flush[n];
    *c_prev_diff = flush_n <= 0? 0 : (c_term_diff * f);
    *i_diff = c_term_diff * g;
    *f_diff = flush_n <= 0? 0 : (c_term_diff * c_prev);
    *o_diff = H_diff[index] * tanh_c;
    *g_diff = c_term_diff * i;
  }
}

template <typename Dtype>
__global__ void LSTMActsBackward(const int nthreads, const int dim,
    const Dtype* X_acts, const Dtype* X_acts_diff, Dtype* X_diff) {
  const int x_dim = 4 * dim;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % x_dim;
    const Dtype X_act = X_acts[index];
    if (d < 3 * dim) {
      X_diff[index] = X_acts_diff[index] * X_act * (Dtype(1) - X_act);
    } else {
      X_diff[index] = X_acts_diff[index] * (Dtype(1) - X_act * X_act);
    }
  }
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[3]) << "Cannot backpropagate to sequence indicators.";
  if (!propagate_down[0] && !propagate_down[1] && !propagate_down[2]) return;

  const int count = top[1]->count();
  const int num = bottom[0]->shape(1);
  const int x_dim = hidden_dim_ * 4;
  const Dtype* C_prev = bottom[0]->gpu_data();
  const Dtype* X_acts = X_acts_.gpu_data();
  const Dtype* flush = bottom[3]->gpu_data();
  const Dtype* C = top[0]->gpu_data();
  const Dtype* H = top[1]->gpu_data();
  const Dtype* C_diff = top[0]->gpu_diff();
  const Dtype* H_diff = top[1]->gpu_diff();
  Dtype* C_prev_diff = bottom[0]->mutable_gpu_diff();
  Dtype* X_acts_diff = X_acts_.mutable_gpu_diff();
  LSTMUnitBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, hidden_dim_,
      C_prev, X_acts, C, H, flush, C_diff, H_diff, C_prev_diff, X_acts_diff);
  CUDA_POST_KERNEL_CHECK;
  const int X_count = bottom[2]->count();
  Dtype* X_diff = bottom[2]->mutable_gpu_diff();
  LSTMActsBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(X_count), CAFFE_CUDA_NUM_THREADS>>>(
      X_count, hidden_dim_, X_acts, X_acts_diff, X_diff);
  CUDA_POST_KERNEL_CHECK;
  if (bottom.size() > 4)
    caffe_copy(X_count, X_diff, bottom[4]->mutable_gpu_diff());
  Dtype * H_prev_diff = bottom[1]->mutable_gpu_diff();
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, x_dim, hidden_dim_, num,
    (Dtype)1., X_diff, H_prev_diff,
    (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, hidden_dim_, x_dim,
    (Dtype)1., X_diff, this->blobs_[0]->gpu_data(), (Dtype)0., H_prev_diff);
  indicator<Dtype>
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, hidden_dim_, flush, H_prev_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(LSTMUnitLayer);

}  // namespace caffe
