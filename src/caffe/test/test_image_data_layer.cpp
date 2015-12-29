#ifdef USE_OPENCV
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ImageDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ImageDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 5; ++i) {
      outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << i;
    }
    outfile.close();
    // Create test input file for images of distinct sizes.
    MakeTempFilename(&filename_reshape_);
    std::ofstream reshapefile(filename_reshape_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_reshape_;
    reshapefile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << 0;
    reshapefile << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg " << 1;
    reshapefile.close();
  }

  virtual ~ImageDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_;
  string filename_reshape_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ImageDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_new_height(256);
  image_data_param->set_new_width(256);
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(1);
  image_data_param->set_source(this->filename_reshape_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  // fish-bike.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 323);
  EXPECT_EQ(this->blob_top_data_->width(), 481);
}

TYPED_TEST(ImageDataLayerTest, TestReshapeMinorEdgeLength) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_minor_edge_length(120);
  image_data_param->set_batch_size(1);
  image_data_param->set_source(this->filename_reshape_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  for (int i = 0; i < 5; ++i) {
    // cat.jpg: (360, 480) -> (120, 160)
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_data_->num(), 1);
    EXPECT_EQ(this->blob_top_data_->channels(), 3);
    EXPECT_EQ(this->blob_top_data_->height(), 120);
    EXPECT_EQ(this->blob_top_data_->width(), 160);
    // fish-bike.jpg: (323, 481) -> (120, round(481/323 * 120))
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_data_->num(), 1);
    EXPECT_EQ(this->blob_top_data_->channels(), 3);
    EXPECT_EQ(this->blob_top_data_->height(), 120);
    const double ratio = static_cast<double>(481) / 323;
    const int expected_width = static_cast<int>(ratio * 120 + 0.5);
    EXPECT_EQ(this->blob_top_data_->width(), expected_width);
  }
}

TYPED_TEST(ImageDataLayerTest, TestReshapeMinorEdgeLengthRange) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  const int kMinorEdgeLength = 120;
  image_data_param->set_minor_edge_length(kMinorEdgeLength);
  const int kMinorEdgeMaxLength = 122;
  image_data_param->set_minor_edge_max_length(kMinorEdgeMaxLength);
  image_data_param->set_batch_size(1);
  image_data_param->set_source(this->filename_reshape_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  int height, max_height = 0, min_height = INT_MAX;
  for (int i = 0; i < 10; ++i) {
    {
      // cat.jpg: (360, 480)
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      EXPECT_EQ(this->blob_top_data_->num(), 1);
      EXPECT_EQ(this->blob_top_data_->channels(), 3);
      height = this->blob_top_data_->height();
      EXPECT_GE(height, kMinorEdgeLength);
      EXPECT_LE(height, kMinorEdgeMaxLength);
      max_height = std::max(max_height, height);
      min_height = std::min(min_height, height);
      const double ratio = static_cast<double>(480) / 360;
      const int expected_width = static_cast<int>(ratio * height + 0.5);
      EXPECT_EQ(this->blob_top_data_->width(), expected_width);
    }
    {
      // fish-bike.jpg: (323, 481)
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      EXPECT_EQ(this->blob_top_data_->num(), 1);
      EXPECT_EQ(this->blob_top_data_->channels(), 3);
      height = this->blob_top_data_->height();
      EXPECT_GE(height, kMinorEdgeLength);
      EXPECT_LE(height, kMinorEdgeMaxLength);
      max_height = std::max(max_height, height);
      min_height = std::min(min_height, height);
      const double ratio = static_cast<double>(481) / 323;
      const int expected_width = static_cast<int>(ratio * height + 0.5);
      EXPECT_EQ(this->blob_top_data_->width(), expected_width);
    }
  }
  // Check that the entire range of possible sizes was seen.
  // (Not true in general, but likely with only 3 possible sizes and 20 trials.)
  EXPECT_EQ(max_height, kMinorEdgeMaxLength);
  EXPECT_EQ(min_height, kMinorEdgeLength);
}

TYPED_TEST(ImageDataLayerTest, TestReshapeMinorEdgeLengthWithDistortion) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  const int kMinorEdgeLength = 6;
  const float kMaxDistortion = 1.5;
  image_data_param->set_minor_edge_length(kMinorEdgeLength);
  image_data_param->set_max_distortion(kMaxDistortion);
  image_data_param->set_batch_size(1);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // cat.jpg (height=360, width=480)
  // Distortion factor in [2/3, 3/2] ->
  //     height in [360 / 1.5, 360 * 1.5] = [240, 540]
  // Resulting height in [240, 540] could be the major or minor axis:
  // height minor (240 <= height <= width == 480) ->
  //     output height is 6; width in range [6, (480/240)*6] = [6, 12]
  const int kExpectedMaxWidth = 12;
  // width minor (width == 480 <= height <= 540)
  //     output width is 6; height in range [6, round((540/480)*6)]
  //                                      = [6, round(6.75)]
  //                                      = [6, 7]
  const int kExpectedMaxHeight = 7;
  int min_width = INT_MAX, max_width = 0;
  int min_height = INT_MAX, max_height = 0;
  for (int i = 0; i < 50; ++i) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_data_->num(), 1);
    EXPECT_EQ(this->blob_top_data_->channels(), 3);
    const int height = this->blob_top_data_->height();
    const int width = this->blob_top_data_->width();
    if (height <= width) {
      EXPECT_EQ(height, kMinorEdgeLength);
      EXPECT_GE(width, kMinorEdgeLength);
      EXPECT_LE(width, kExpectedMaxWidth);
    }
    if (height >= width) {
      EXPECT_EQ(width, kMinorEdgeLength);
      EXPECT_GE(height, kMinorEdgeLength);
      EXPECT_LE(height, kExpectedMaxHeight);
    }
    max_height = std::max(max_height, height);
    min_height = std::min(min_height, height);
    max_width = std::max(max_width, width);
    min_width = std::min(min_width, width);
  }
  // Check that we saw all the full range of possible image sizes.
  // (Note that this is not guaranteed in general, but is reasonably likely
  // to occur with the chosen (small) edge length and (large) number of trials,
  // and in particular, does occur with the seed set here.)
  EXPECT_EQ(min_width, kMinorEdgeLength);
  EXPECT_EQ(min_height, kMinorEdgeLength);
  EXPECT_EQ(max_width, kExpectedMaxWidth);
  EXPECT_EQ(max_height, kExpectedMaxHeight);
}

TYPED_TEST(ImageDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(true);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      Dtype value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == Dtype(i));
    }
    EXPECT_EQ(5, values_to_indices.size());
    EXPECT_GT(5, num_in_order);
  }
}

}  // namespace caffe
#endif  // USE_OPENCV
