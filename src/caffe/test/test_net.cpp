// Copyright 2014 BVLC and contributors.

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::ostringstream;

namespace caffe {

template <typename Dtype>
class NetTest : public ::testing::Test {
 protected:
  NetTest() : seed_(1701) {}

  virtual void InitNetFromProtoString(const string& proto) {
    NetParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    net_.reset(new Net<Dtype>(param));
  }

  virtual void CopyNetBlobs(const bool copy_diff,
      vector<shared_ptr<Blob<Dtype> > >* blobs_copy) {
    CHECK(net_);
    const vector<shared_ptr<Blob<Dtype> > >& net_blobs = net_->blobs();
    blobs_copy->clear();
    blobs_copy->resize(net_blobs.size());
    const bool kReshape = true;
    for (int i = 0; i < net_blobs.size(); ++i) {
      (*blobs_copy)[i].reset(new Blob<Dtype>());
      (*blobs_copy)[i]->CopyFrom(*net_blobs[i], copy_diff, kReshape);
    }
  }

  virtual void CopyNetParams(const bool copy_diff,
      vector<shared_ptr<Blob<Dtype> > >* params_copy) {
    CHECK(net_);
    const vector<shared_ptr<Blob<Dtype> > >& net_params = net_->params();
    params_copy->clear();
    params_copy->resize(net_params.size());
    const bool kReshape = true;
    for (int i = 0; i < net_params.size(); ++i) {
      (*params_copy)[i].reset(new Blob<Dtype>());
      (*params_copy)[i]->CopyFrom(*net_params[i], copy_diff, kReshape);
    }
  }

  virtual void InitTinyNet(const bool force_backward = false,
                           const bool accuracy_layer = false) {
    string proto =
        "name: 'TinyTestNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    num: 5 "
        "    channels: 1 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layers: { "
        "  name: 'innerproduct' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 1000 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  blobs_lr: 1. "
        "  blobs_lr: 2. "
        "  weight_decay: 1. "
        "  weight_decay: 0. "
        "  bottom: 'data' "
        "  top: 'innerproduct' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: SOFTMAX_LOSS "
        "  bottom: 'innerproduct' "
        "  bottom: 'label' "
        "  top: 'top_loss' "
        "} ";
    if (accuracy_layer) {
      proto +=
          "layers: { "
          "  name: 'loss' "
          "  type: ACCURACY "
          "  bottom: 'innerproduct' "
          "  bottom: 'label' "
          "  top: 'accuracy' "
          "} ";
    }
    if (force_backward) {
      proto += "force_backward: true ";
    }
    InitNetFromProtoString(proto);
  }

  virtual void InitTinyNetEuclidean(const bool force_backward = false) {
    string proto =
        "name: 'TinyTestEuclidLossNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    num: 5 "
        "    channels: 1 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layers: { "
        "  name: 'innerproduct' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 1 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  blobs_lr: 1. "
        "  blobs_lr: 2. "
        "  weight_decay: 1. "
        "  weight_decay: 0. "
        "  bottom: 'data' "
        "  top: 'innerproduct' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS "
        "  bottom: 'innerproduct' "
        "  bottom: 'label' "
        "} ";
    if (force_backward) {
      proto += "force_backward: true ";
    }
    InitNetFromProtoString(proto);
  }

  virtual void InitTrickyNet(Dtype* loss_weight = NULL) {
    ostringstream loss_weight_stream;
    if (loss_weight) {
      loss_weight_stream << "  top_loss_weight: " << *loss_weight << " ";
    }
    const string& proto =
        "name: 'TrickyTestNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    num: 5 "
        "    channels: 1 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layers: { "
        "  name: 'innerproduct' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 1000 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  blobs_lr: 1. "
        "  blobs_lr: 2. "
        "  weight_decay: 1. "
        "  weight_decay: 0. "
        "  bottom: 'data' "
        "  top: 'transformed_data' "
        "} "
        "layers: { "
        "  name: 'innerproduct' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 1 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  blobs_lr: 1. "
        "  blobs_lr: 2. "
        "  weight_decay: 1. "
        "  weight_decay: 0. "
        "  bottom: 'label' "
        "  top: 'transformed_label' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: SOFTMAX_LOSS " +
        loss_weight_stream.str() +
        "  bottom: 'transformed_data' "
        "  bottom: 'transformed_label' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitUnsharedWeightsNet(Dtype* loss_weight = NULL,
      Dtype* midnet_loss_weight = NULL, bool force_backward = false) {
    // Loss weight for the EUCLIDEAN_LOSS layer output.
    // Should default to 1.0 if unspecified (i.e., if a NULL loss_weight is
    // passed to this function).
    ostringstream loss_weight_stream;
    if (loss_weight) {
      loss_weight_stream << "  top_loss_weight: " << *loss_weight << " ";
    }
    // Loss weight for the first INNER_PRODUCT layer output.
    // Should default to 0.0 if unspecified (i.e., if a NULL midnet_loss_weight
    // is passed to this function).
    ostringstream midnet_loss_weight_stream;
    if (midnet_loss_weight) {
      midnet_loss_weight_stream << "  top_loss_weight: "
                                << *midnet_loss_weight << " ";
    }
    const string& force_backward_string =
        force_backward ? " force_backward: true " : "";
    const string& proto =
        "name: 'UnsharedWeightsNetwork' " + force_backward_string +
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "} "
        "layers: { "
        "  name: 'innerproduct1' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param: 'unsharedweights1' "
        "  bottom: 'data' "
        "  top: 'innerproduct1' " +
        midnet_loss_weight_stream.str() +
        "} "
        "layers: { "
        "  name: 'innerproduct2' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param: 'unsharedweights2' "
        "  bottom: 'data' "
        "  top: 'innerproduct2' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS " +
        loss_weight_stream.str() +
        "  bottom: 'innerproduct1' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitSharedWeightsNet() {
    const string& proto =
        "name: 'SharedWeightsNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "} "
        "layers: { "
        "  name: 'innerproduct1' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param: 'sharedweights' "
        "  bottom: 'data' "
        "  top: 'innerproduct1' "
        "} "
        "layers: { "
        "  name: 'innerproduct2' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param: 'sharedweights' "
        "  bottom: 'data' "
        "  top: 'innerproduct2' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS "
        "  bottom: 'innerproduct1' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitDiffDataUnsharedWeightsNet() {
    const string& proto =
        "name: 'DiffDataUnsharedWeightsNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  top: 'data1' "
        "  top: 'data2' "
        "} "
        "layers: { "
        "  name: 'innerproduct1' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param: 'unsharedweights1' "
        "  bottom: 'data1' "
        "  top: 'innerproduct1' "
        "} "
        "layers: { "
        "  name: 'innerproduct2' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param: 'unsharedweights2' "
        "  bottom: 'innerproduct1' "
        "  top: 'innerproduct2' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS "
        "  bottom: 'data2' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitDiffDataSharedWeightsNet() {
    const string& proto =
        "name: 'DiffDataSharedWeightsNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  top: 'data1' "
        "  top: 'data2' "
        "} "
        "layers: { "
        "  name: 'innerproduct1' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param: 'sharedweights' "
        "  bottom: 'data1' "
        "  top: 'innerproduct1' "
        "} "
        "layers: { "
        "  name: 'innerproduct2' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param: 'sharedweights' "
        "  bottom: 'innerproduct1' "
        "  top: 'innerproduct2' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS "
        "  bottom: 'data2' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  int seed_;
  shared_ptr<Net<Dtype> > net_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(NetTest, Dtypes);

TYPED_TEST(NetTest, TestHasBlob) {
  this->InitTinyNet();
  EXPECT_TRUE(this->net_->has_blob("data"));
  EXPECT_TRUE(this->net_->has_blob("label"));
  EXPECT_TRUE(this->net_->has_blob("innerproduct"));
  EXPECT_FALSE(this->net_->has_blob("loss"));
  EXPECT_TRUE(this->net_->has_blob("top_loss"));
}

TYPED_TEST(NetTest, TestGetBlob) {
  this->InitTinyNet();
  EXPECT_EQ(this->net_->blob_by_name("data"), this->net_->blobs()[0]);
  EXPECT_EQ(this->net_->blob_by_name("label"), this->net_->blobs()[1]);
  EXPECT_EQ(this->net_->blob_by_name("innerproduct"), this->net_->blobs()[2]);
  EXPECT_FALSE(this->net_->blob_by_name("loss"));
  EXPECT_EQ(this->net_->blob_by_name("top_loss"), this->net_->blobs()[3]);
}

TYPED_TEST(NetTest, TestHasLayer) {
  this->InitTinyNet();
  EXPECT_TRUE(this->net_->has_layer("data"));
  EXPECT_TRUE(this->net_->has_layer("innerproduct"));
  EXPECT_TRUE(this->net_->has_layer("loss"));
  EXPECT_FALSE(this->net_->has_layer("label"));
}

TYPED_TEST(NetTest, TestGetLayerByName) {
  this->InitTinyNet();
  EXPECT_EQ(this->net_->layer_by_name("data"), this->net_->layers()[0]);
  EXPECT_EQ(this->net_->layer_by_name("innerproduct"), this->net_->layers()[1]);
  EXPECT_EQ(this->net_->layer_by_name("loss"), this->net_->layers()[2]);
  EXPECT_FALSE(this->net_->layer_by_name("label"));
}

TYPED_TEST(NetTest, TestBottomNeedBackward) {
  this->InitTinyNet();
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(3, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(false, bottom_need_backward[1][0]);
  EXPECT_EQ(2, bottom_need_backward[2].size());
  EXPECT_EQ(true, bottom_need_backward[2][0]);
  EXPECT_EQ(false, bottom_need_backward[2][1]);
}

TYPED_TEST(NetTest, TestBottomNeedBackwardForce) {
  const bool force_backward = true;
  this->InitTinyNet(force_backward);
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(3, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(true, bottom_need_backward[1][0]);
  EXPECT_EQ(2, bottom_need_backward[2].size());
  EXPECT_EQ(true, bottom_need_backward[2][0]);
  EXPECT_EQ(false, bottom_need_backward[2][1]);
}

TYPED_TEST(NetTest, TestBottomNeedBackwardEuclideanForce) {
  const bool force_backward = true;
  this->InitTinyNetEuclidean(force_backward);
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(3, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(true, bottom_need_backward[1][0]);
  EXPECT_EQ(2, bottom_need_backward[2].size());
  EXPECT_EQ(true, bottom_need_backward[2][0]);
  EXPECT_EQ(true, bottom_need_backward[2][1]);
}

TYPED_TEST(NetTest, TestBottomNeedBackwardTricky) {
  this->InitTrickyNet();
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(4, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(false, bottom_need_backward[1][0]);
  EXPECT_EQ(1, bottom_need_backward[2].size());
  EXPECT_EQ(false, bottom_need_backward[2][0]);
  EXPECT_EQ(2, bottom_need_backward[3].size());
  EXPECT_EQ(true, bottom_need_backward[3][0]);
  // The label input to the SoftmaxLossLayer should say it "needs backward"
  // since it has weights under it, even though we expect this to cause a crash
  // at training/test time.
  EXPECT_EQ(true, bottom_need_backward[3][1]);
}

TYPED_TEST(NetTest, TestLossWeight) {
  // First, compute the loss and gradients with no top_loss_weight specified.
  // In this case, the loss weight for the EUCLIDEAN_LOSS layer should default
  // to 1.
  vector<Blob<TypeParam>*> bottom;
  Caffe::set_random_seed(this->seed_);
  const bool kForceBackward = true;
  this->InitUnsharedWeightsNet(NULL, NULL, kForceBackward);
  const TypeParam loss = this->net_->ForwardBackward(bottom);
  const bool kCopyDiff = true;
  vector<shared_ptr<Blob<TypeParam> > > blob_grads;
  this->CopyNetBlobs(kCopyDiff, &blob_grads);
  vector<shared_ptr<Blob<TypeParam> > > param_grads;
  this->CopyNetParams(kCopyDiff, &param_grads);
  // Check that the loss is non-trivial, otherwise the test doesn't prove much.
  const TypeParam kMinLossAbsValue = 1e-2;
  ASSERT_GE(fabs(loss), kMinLossAbsValue);
  const TypeParam kErrorMargin = 1e-5;
  const int kNumLossWeights = 6;
  TypeParam kLossWeights[kNumLossWeights] = {2, 0, 1, -1, -2.5, 3.7};
  for (int i = 0; i < kNumLossWeights; ++i) {
    Caffe::set_random_seed(this->seed_);
    this->InitUnsharedWeightsNet(&kLossWeights[i], NULL, kForceBackward);
    const TypeParam weighted_loss = this->net_->ForwardBackward(bottom);
    const TypeParam error_margin = kErrorMargin * fabs(kLossWeights[i]);
    EXPECT_NEAR(loss * kLossWeights[i], weighted_loss, error_margin)
        << "loss weight = " << kLossWeights[i];
    const vector<shared_ptr<Blob<TypeParam> > >& weighted_blobs =
        this->net_->blobs();
    ASSERT_EQ(blob_grads.size(), weighted_blobs.size());
    for (int j = 0; j < blob_grads.size(); ++j) {
      ASSERT_EQ(blob_grads[j]->count(), weighted_blobs[j]->count());
      for (int k = 0; k < blob_grads[j]->count(); ++k) {
        EXPECT_NEAR(blob_grads[j]->cpu_diff()[k] * kLossWeights[i],
                    weighted_blobs[j]->cpu_diff()[k], error_margin);
      }
    }
    const vector<shared_ptr<Blob<TypeParam> > >& weighted_params =
        this->net_->params();
    ASSERT_EQ(param_grads.size(), weighted_params.size());
    for (int j = 0; j < param_grads.size(); ++j) {
      ASSERT_EQ(param_grads[j]->count(), weighted_params[j]->count());
      for (int k = 0; k < param_grads[j]->count(); ++k) {
        EXPECT_NEAR(param_grads[j]->cpu_diff()[k] * kLossWeights[i],
                    weighted_params[j]->cpu_diff()[k], error_margin);
      }
    }
  }
}

TYPED_TEST(NetTest, TestLossWeightMidNet) {
  vector<Blob<TypeParam>*> bottom;
  Caffe::set_random_seed(this->seed_);
  const bool kForceBackward = true;
  TypeParam loss_weight = 0;
  TypeParam midnet_loss_weight = 1;
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const TypeParam loss = this->net_->ForwardBackward(bottom);
  const bool kCopyDiff = true;
  const bool kReshape = true;
  Blob<TypeParam> data_grad;
  data_grad.CopyFrom(*this->net_->blob_by_name("data"), kCopyDiff, kReshape);
  // Check that the loss is non-trivial, otherwise the test doesn't prove much.
  const TypeParam kMinLossAbsValue = 1e-2;
  ASSERT_GE(fabs(loss), kMinLossAbsValue);
  const TypeParam kErrorMargin = 1e-5;
  const int kNumLossWeights = 6;
  TypeParam kLossWeights[kNumLossWeights] = {2, 0, 1, -1, -2.5, 3.7};
  for (int i = 0; i < kNumLossWeights; ++i) {
    Caffe::set_random_seed(this->seed_);
    this->InitUnsharedWeightsNet(&loss_weight, &kLossWeights[i],
                                 kForceBackward);
    const TypeParam weighted_loss = this->net_->ForwardBackward(bottom);
    const TypeParam error_margin = kErrorMargin * fabs(kLossWeights[i]);
    EXPECT_NEAR(loss * kLossWeights[i], weighted_loss, error_margin)
        << "loss weight = " << kLossWeights[i];
    const shared_ptr<Blob<TypeParam> >& weighted_blob =
        this->net_->blob_by_name("data");
    ASSERT_EQ(data_grad.count(), weighted_blob->count());
    for (int j = 0; j < data_grad.count(); ++j) {
      EXPECT_NEAR(data_grad.cpu_diff()[j] * kLossWeights[i],
                  weighted_blob->cpu_diff()[j], error_margin);
    }
  }
}

TYPED_TEST(NetTest, TestComboLossWeight) {
  vector<Blob<TypeParam>*> bottom;
  TypeParam loss_weight;
  TypeParam midnet_loss_weight;
  const bool kForceBackward = true;
  const TypeParam kErrorMargin = 1e-4;

  // Get the loss and gradients with EUCLIDEAN_LOSS weight 1,
  // INNER_PRODUCT weight 1.
  loss_weight = 1;
  midnet_loss_weight = 1;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const TypeParam loss = this->net_->ForwardBackward(bottom);
  const bool kCopyDiff = true;
  vector<shared_ptr<Blob<TypeParam> > > blob_grads;
  this->CopyNetBlobs(kCopyDiff, &blob_grads);
  vector<shared_ptr<Blob<TypeParam> > > param_grads;
  this->CopyNetParams(kCopyDiff, &param_grads);

  loss_weight = 2;
  midnet_loss_weight = 1;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const TypeParam loss_main_2 = this->net_->ForwardBackward(bottom);
  vector<shared_ptr<Blob<TypeParam> > > blob_grads_loss_2;
  this->CopyNetBlobs(kCopyDiff, &blob_grads_loss_2);
  vector<shared_ptr<Blob<TypeParam> > > param_grads_loss_2;
  this->CopyNetParams(kCopyDiff, &param_grads_loss_2);

  loss_weight = 3;
  midnet_loss_weight = 1;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const TypeParam loss_main_3 = this->net_->ForwardBackward(bottom);
  const vector<shared_ptr<Blob<TypeParam> > >& blob_grads_loss_3 =
      this->net_->blobs();
  ASSERT_EQ(blob_grads.size(), blob_grads_loss_3.size());
  ASSERT_EQ(blob_grads_loss_2.size(), blob_grads_loss_3.size());
  for (int j = 0; j < blob_grads.size(); ++j) {
    const string& blob_name = this->net_->blob_names()[j];
    bool grad_should_change = true;
    if (blob_name == "innerproduct1_innerproduct1_0_split_0") {
      grad_should_change = false;
    }
    ASSERT_EQ(blob_grads[j]->count(), blob_grads_loss_3[j]->count());
    ASSERT_EQ(blob_grads_loss_2[j]->count(), blob_grads_loss_3[j]->count());
    for (int k = 0; k < blob_grads[j]->count(); ++k) {
      const TypeParam grad_diff_2 = blob_grads_loss_2[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      const TypeParam grad_diff_3 = blob_grads_loss_3[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      if (grad_should_change) {
        // Test non-triviality.
        const TypeParam kMinGradDiffAbsValue = 1e-4;
        EXPECT_GT(fabs(grad_diff_2), kMinGradDiffAbsValue) << blob_name;
        EXPECT_NEAR(2 * grad_diff_2, grad_diff_3, kErrorMargin) << blob_name;
      } else {
        EXPECT_EQ(0, grad_diff_2) << blob_name;
        EXPECT_EQ(0, grad_diff_3) << blob_name;
      }
    }
  }

  loss_weight = 1;
  midnet_loss_weight = 2;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const TypeParam loss_midnet_2 = this->net_->ForwardBackward(bottom);
  this->CopyNetBlobs(kCopyDiff, &blob_grads_loss_2);
  this->CopyNetParams(kCopyDiff, &param_grads_loss_2);

  loss_weight = 1;
  midnet_loss_weight = 3;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const TypeParam loss_midnet_3 = this->net_->ForwardBackward(bottom);
  const vector<shared_ptr<Blob<TypeParam> > >& blob_grads_midnet_loss_3 =
      this->net_->blobs();
  ASSERT_EQ(blob_grads.size(), blob_grads_midnet_loss_3.size());
  ASSERT_EQ(blob_grads_loss_2.size(), blob_grads_midnet_loss_3.size());
  const vector<string>& blob_names = this->net_->blob_names();
  for (int j = 0; j < blob_grads.size(); ++j) {
    const string& blob_name = blob_names[j];
    bool grad_should_change = false;
    if (blob_name == "innerproduct1" ||
        blob_name == "innerproduct1_innerproduct1_0_split_0" ||
        blob_name == "data_data_0_split_0" || blob_name == "data") {
      grad_should_change = true;
    }
    ASSERT_EQ(blob_grads[j]->count(), blob_grads_midnet_loss_3[j]->count());
    ASSERT_EQ(blob_grads[j]->count(), blob_grads_loss_2[j]->count());
    for (int k = 0; k < blob_grads[j]->count(); ++k) {
      const TypeParam grad_diff_2 = blob_grads_loss_2[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      const TypeParam grad_diff_3 = blob_grads_midnet_loss_3[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      if (grad_should_change) {
        // Test non-triviality.
        const TypeParam kMinGradDiffAbsValue = 1e-4;
        EXPECT_GT(fabs(grad_diff_2), kMinGradDiffAbsValue) << blob_name;
        EXPECT_NEAR(2 * grad_diff_2, grad_diff_3, kErrorMargin) << blob_name;
      } else {
        EXPECT_EQ(0, grad_diff_2) << blob_name;
        EXPECT_EQ(0, grad_diff_3) << blob_name;
      }
    }
  }

  const TypeParam kMinLossDiffAbsValue = 1e-4;

  TypeParam loss_diff_2 = loss_main_2 - loss;
  // Test non-triviality.
  EXPECT_GT(fabs(loss_diff_2), kMinLossDiffAbsValue);
  TypeParam loss_diff_3 = loss_main_3 - loss;
  EXPECT_NEAR(2 * loss_diff_2, loss_diff_3, kErrorMargin);

  loss_diff_2 = loss_midnet_2 - loss;
  // Test non-triviality.
  EXPECT_GT(fabs(loss_diff_2), kMinLossDiffAbsValue);
  loss_diff_3 = loss_midnet_3 - loss;
  EXPECT_NEAR(2 * loss_diff_2, loss_diff_3, kErrorMargin);
}

TYPED_TEST(NetTest, TestBackwardWithAccuracyLayer) {
  const bool kForceBackward = false;
  const bool kAccuracyLayer = true;
  this->InitTinyNet(kForceBackward, kAccuracyLayer);
  EXPECT_TRUE(this->net_->has_blob("accuracy"));
  vector<Blob<TypeParam>*> bottom;
  // Test that we can do Backward even though we have an ACCURACY layer.
  this->net_->ForwardBackward(bottom);
}

TYPED_TEST(NetTest, TestUnsharedWeightsDataNet) {
  this->InitUnsharedWeightsNet();
  vector<Blob<TypeParam>*> bottom;
  TypeParam loss;
  this->net_->Forward(bottom, &loss);
  EXPECT_GT(loss, 0);
}

TYPED_TEST(NetTest, TestSharedWeightsDataNet) {
  this->InitSharedWeightsNet();
  vector<Blob<TypeParam>*> bottom;
  TypeParam loss;
  this->net_->Forward(bottom, &loss);
  EXPECT_FLOAT_EQ(loss, 0);
}

TYPED_TEST(NetTest, TestUnsharedWeightsDiffNet) {
  this->InitUnsharedWeightsNet();
  vector<Blob<TypeParam>*> bottom;
  Net<TypeParam>* net = this->net_.get();
  net->Forward(bottom);
  net->Backward();
  Layer<TypeParam>* ip1_layer = net->layer_by_name("innerproduct1").get();
  Layer<TypeParam>* ip2_layer = net->layer_by_name("innerproduct2").get();
  const int count = ip1_layer->blobs()[0]->count();
  const TypeParam* grad1 = ip1_layer->blobs()[0]->cpu_diff();
  const TypeParam* grad2 = ip2_layer->blobs()[0]->cpu_diff();
  for (int i = 0; i < count; ++i) {
    EXPECT_GT(fabs(grad1[i]), 0);
    EXPECT_FLOAT_EQ(-1 * grad1[i], grad2[i]);
  }
}

TYPED_TEST(NetTest, TestSharedWeightsDiffNet) {
  this->InitSharedWeightsNet();
  vector<Blob<TypeParam>*> bottom;
  Net<TypeParam>* net = this->net_.get();
  TypeParam loss;
  net->Forward(bottom, &loss);
  net->Backward();
  EXPECT_FLOAT_EQ(loss, 0);
  Layer<TypeParam>* ip1_layer = net->layer_by_name("innerproduct1").get();
  Layer<TypeParam>* ip2_layer = net->layer_by_name("innerproduct2").get();
  const int count = ip1_layer->blobs()[0]->count();
  const TypeParam* grad1 = ip1_layer->blobs()[0]->cpu_diff();
  const TypeParam* grad2 = ip2_layer->blobs()[0]->cpu_diff();
  for (int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(0, grad1[i]);
    EXPECT_FLOAT_EQ(0, grad2[i]);
  }
}

TYPED_TEST(NetTest, TestSharedWeightsUpdate) {
  Caffe::set_random_seed(this->seed_);
  this->InitDiffDataSharedWeightsNet();
  vector<Blob<TypeParam>*> bottom;
  EXPECT_EQ(this->net_->layer_names()[1], "innerproduct1");
  EXPECT_EQ(this->net_->layer_names()[2], "innerproduct2");
  Blob<TypeParam>* ip1_weights = this->net_->layers()[1]->blobs()[0].get();
  Blob<TypeParam>* ip2_weights = this->net_->layers()[2]->blobs()[0].get();
  // Check that data blobs of shared weights share the same location in memory.
  EXPECT_EQ(ip1_weights->cpu_data(), ip2_weights->cpu_data());
  // Check that diff blobs of shared weights are at different locations in
  // locations.  (The diffs should be accumulated at update time.)
  EXPECT_NE(ip1_weights->cpu_diff(), ip2_weights->cpu_diff());
  this->net_->Forward(bottom);
  this->net_->Backward();
  // Compute the expected update as the data minus the two diffs.
  Blob<TypeParam> shared_params;
  const bool reshape = true;
  const bool copy_diff = false;
  shared_params.CopyFrom(*ip1_weights, copy_diff, reshape);
  shared_params.CopyFrom(*ip1_weights, !copy_diff, reshape);
  const int count = ip1_weights->count();
  // Make sure the diffs are non-trivial.
  for (int i = 0; i < count; ++i) {
    EXPECT_NE(0, ip1_weights->cpu_diff()[i]);
    EXPECT_NE(0, ip2_weights->cpu_diff()[i]);
    EXPECT_NE(ip1_weights->cpu_diff()[i], ip2_weights->cpu_diff()[i]);
  }
  caffe_axpy(count, TypeParam(1), ip2_weights->cpu_diff(),
             shared_params.mutable_cpu_diff());
  caffe_axpy(count, TypeParam(-1), shared_params.cpu_diff(),
             shared_params.mutable_cpu_data());
  const TypeParam* expected_updated_params = shared_params.cpu_data();
  this->net_->Update();
  const TypeParam* actual_updated_params = ip1_weights->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(expected_updated_params[i], actual_updated_params[i]);
  }
  // Check that data blobs of shared weights STILL point to the same memory
  // location (because ... who knows).
  EXPECT_EQ(ip1_weights->cpu_data(), ip2_weights->cpu_data());

  Caffe::set_random_seed(this->seed_);
  this->InitDiffDataUnsharedWeightsNet();
  EXPECT_EQ(this->net_->layer_names()[1], "innerproduct1");
  EXPECT_EQ(this->net_->layer_names()[2], "innerproduct2");
  ip1_weights = this->net_->layers()[1]->blobs()[0].get();
  ip2_weights = this->net_->layers()[2]->blobs()[0].get();
  // Check that data and diff blobs of unshared weights are at different
  // locations in memory.
  EXPECT_NE(ip1_weights->cpu_data(), ip2_weights->cpu_data());
  EXPECT_NE(ip1_weights->cpu_diff(), ip2_weights->cpu_diff());
  this->net_->Forward(bottom);
  this->net_->Backward();
  // Compute the expected update.
  Blob<TypeParam> unshared_params1;
  unshared_params1.CopyFrom(*ip1_weights, copy_diff, reshape);
  unshared_params1.CopyFrom(*ip1_weights, !copy_diff, reshape);
  Blob<TypeParam> unshared_params2;
  unshared_params2.CopyFrom(*ip2_weights, copy_diff, reshape);
  unshared_params2.CopyFrom(*ip2_weights, !copy_diff, reshape);
  // Make sure the diffs are non-trivial and sum to the diff in the shared net.
  for (int i = 0; i < count; ++i) {
    EXPECT_NE(0, ip1_weights->cpu_diff()[i]);
    EXPECT_NE(0, ip2_weights->cpu_diff()[i]);
    EXPECT_NE(ip1_weights->cpu_diff()[i], ip2_weights->cpu_diff()[i]);
    EXPECT_EQ(ip1_weights->cpu_diff()[i] + ip2_weights->cpu_diff()[i],
              shared_params.cpu_diff()[i]);
  }
  caffe_axpy(count, TypeParam(-1), ip1_weights->cpu_diff(),
             unshared_params1.mutable_cpu_data());
  caffe_axpy(count, TypeParam(-1), ip2_weights->cpu_diff(),
             unshared_params2.mutable_cpu_data());
  const TypeParam* expected_updated_params1 = unshared_params1.cpu_data();
  const TypeParam* expected_updated_params2 = unshared_params2.cpu_data();
  this->net_->Update();
  const TypeParam* actual_updated_params1 = ip1_weights->cpu_data();
  const TypeParam* actual_updated_params2 = ip2_weights->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(expected_updated_params1[i], actual_updated_params1[i]);
    EXPECT_EQ(expected_updated_params2[i], actual_updated_params2[i]);
    EXPECT_NE(actual_updated_params1[i], actual_updated_params2[i]);
    EXPECT_NE(expected_updated_params, expected_updated_params1);
  }
}

}  // namespace caffe
