// Copyright 2014 BVLC and contributors.

#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class NeuronLayerTest : public ::testing::Test {
 protected:
  NeuronLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~NeuronLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(NeuronLayerTest, Dtypes);

TYPED_TEST_ALL_DEVICES(NeuronLayerTest, TestReLU,
  LayerParameter layer_param;
  ReLULayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
  }
)

TYPED_TEST_ALL_DEVICES(NeuronLayerTest, TestReLUGradient,
  LayerParameter layer_param;
  ReLULayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
)

TYPED_TEST_ALL_DEVICES(NeuronLayerTest, TestSigmoid,
  LayerParameter layer_param;
  SigmoidLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_FLOAT_EQ(top_data[i], 1. / (1 + exp(-bottom_data[i])));
    // check that we squashed the value between 0 and 1
    EXPECT_GE(top_data[i], 0.);
    EXPECT_LE(top_data[i], 1.);
  }
)

TYPED_TEST_ALL_DEVICES(NeuronLayerTest, TestSigmoidGradient,
  LayerParameter layer_param;
  SigmoidLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
)

TYPED_TEST_ALL_DEVICES(NeuronLayerTest, TestDropout,
  LayerParameter layer_param;
  Caffe::set_phase(Caffe::TRAIN);
  DropoutLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  float scale = 1. / (1. - layer_param.dropout_param().dropout_ratio());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], bottom_data[i] * scale);
    }
  }
)

TYPED_TEST_ALL_DEVICES(NeuronLayerTest, TestDropoutTestPhase,
  LayerParameter layer_param;
  Caffe::set_phase(Caffe::TEST);
  DropoutLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], bottom_data[i]);
    }
  }
)

TYPED_TEST_ALL_DEVICES(NeuronLayerTest, TestDropoutGradient,
  LayerParameter layer_param;
  Caffe::set_phase(Caffe::TRAIN);
  DropoutLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
)

TYPED_TEST_ALL_DEVICES(NeuronLayerTest, TestDropoutGradientTest,
  LayerParameter layer_param;
  Caffe::set_phase(Caffe::TEST);
  DropoutLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
)

TYPED_TEST_ALL_DEVICES(NeuronLayerTest, TestBNLL,
  LayerParameter layer_param;
  BNLLLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_GE(top_data[i], bottom_data[i]);
  }
)

TYPED_TEST_ALL_DEVICES(NeuronLayerTest, TestBNLLGradient,
  LayerParameter layer_param;
  BNLLLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
)


}  // namespace caffe
