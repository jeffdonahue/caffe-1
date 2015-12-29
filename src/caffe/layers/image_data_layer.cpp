#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
cv::Mat ImageDataLayer<Dtype>::ReadCurrentImageToCVMat() {
  const ImageDataParameter& param = this->layer_param_.image_data_param();
  const string& image_filename = param.root_folder() + lines_[lines_id_].first;
  int minor_edge_length = param.minor_edge_length();
  if (minor_edge_length > 0) {
    if (num_length_choices_ > 1) {
      minor_edge_length += this->data_transformer_->Rand(num_length_choices_);
    }
    double distortion = 1;
    if (log_max_distortion_ > 0) {
      const Dtype log_distortion = this->data_transformer_->RandFloat(
          -log_max_distortion_, +log_max_distortion_);
      distortion = std::exp(log_distortion);
    }
    cv::Mat image = ReadImageToCVMatMinorEdge(image_filename, minor_edge_length,
                                              distortion, param.is_color());
    CHECK(image.data) << "Could not load " << image_filename;
    return image;
  }
  cv::Mat image = ReadImageToCVMat(image_filename,
      param.new_height(), param.new_width(), param.is_color());
  CHECK(image.data) << "Could not load " << image_filename;
  return image;
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ImageDataParameter& param = this->layer_param_.image_data_param();
  const int new_height = param.new_height();
  const int new_width = param.new_width();
  const int minor_edge_length = param.minor_edge_length();

  CHECK((new_height == 0 && new_width == 0) || (minor_edge_length == 0))
      << "Both new_height+new_width and minor_edge_length may not be set.";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  CHECK_GE(minor_edge_length, 0) << "minor_edge_length must be non-negative";
  if (param.has_minor_edge_max_length()) {
    CHECK_GT(minor_edge_length, 0) << "If minor_edge_max_length is set, "
        << "minor_edge_length must also be set.";
    CHECK_GE(param.minor_edge_max_length(), minor_edge_length)
        << "minor_edge_max_length must be at least minor_edge_length.";
  }
  num_length_choices_ =
      std::max<int>(1, param.minor_edge_max_length() - minor_edge_length + 1);
  if (param.has_max_distortion()) {
    CHECK_GT(minor_edge_length, 0) << "If max_distortion is set, "
        << "minor_edge_length must also be set.";
    CHECK_GE(param.max_distortion(), 1) << "max_distortion must be >= 1.";
  }
  log_max_distortion_ = std::log(param.max_distortion());
  // Read the file with filenames and labels
  const string& source = param.source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }

  if (param.shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadCurrentImageToCVMat();
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = param.batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const ImageDataParameter& param = this->layer_param_.image_data_param();
  const int batch_size = param.batch_size();

  Dtype* prefetch_data = NULL;
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadCurrentImageToCVMat();
    if (item_id == 0) {
      // Reshape according to the first image of each batch.
      // For batch_size == 1, this allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from cv_img.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
      prefetch_data = batch->data_.mutable_cpu_data();
    }
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (param.shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
#endif  // USE_OPENCV
