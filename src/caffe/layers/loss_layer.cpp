// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void LossLayer<Dtype>::FurtherSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  (*top)[0]->Reshape(1, 1, 1, 1);
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.top_loss_weight_size() == 0) {
    this->layer_param_.add_top_loss_weight(Dtype(1));
  }
}

INSTANTIATE_CLASS(LossLayer);

}  // namespace caffe
