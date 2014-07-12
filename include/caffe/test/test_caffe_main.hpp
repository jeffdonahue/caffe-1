// Copyright 2014 BVLC and contributors.

// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstdio>

using std::cout;
using std::endl;

int main(int argc, char** argv);

// *TEST*_ALL_DEVICES creates a copy of a test for each device.
// Whenever possible, this should be used rather than creating copy-paste
// test duplicates for each device.
// (Currently should really just be TEST_BOTH_DEVICES as Caffe currently just
// two devices: CPU and GPU, but named TEST_ALL_DEVCIES for future-proofing.)
#define TEST_F_ALL_DEVICES(CaseName, TestName, TestBody) \
  TEST_F_DEVICE(CaseName, TestName, CPU, TestBody) \
  TEST_F_DEVICE(CaseName, TestName, GPU, TestBody)

#define TEST_F_DEVICE(CaseName, TestName, DeviceName, TestBody) \
  TEST_F(CaseName, TestName##DeviceName) { \
    Caffe::set_mode(Caffe::DeviceName); \
    TestBody \
  }

#define TYPED_TEST_ALL_DEVICES(CaseName, TestName, TestBody) \
  TYPED_TEST_DEVICE(CaseName, TestName, CPU, TestBody) \
  TYPED_TEST_DEVICE(CaseName, TestName, GPU, TestBody)

#define TYPED_TEST_DEVICE(CaseName, TestName, DeviceName, TestBody) \
  TYPED_TEST(CaseName, TestName##DeviceName) { \
    Caffe::set_mode(Caffe::DeviceName); \
    TestBody \
  }

#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
