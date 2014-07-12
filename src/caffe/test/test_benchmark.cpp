// Copyright 2014 BVLC and contributors.

#include <unistd.h>  // for usleep
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

class BenchmarkTest : public ::testing::Test {};

TEST_F_ALL_DEVICES(BenchmarkTest, TestTimerConstructor,
  Timer timer;
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
)

TEST_F_ALL_DEVICES(BenchmarkTest, TestTimerStart,
  Timer timer;
  timer.Start();
  EXPECT_TRUE(timer.initted());
  EXPECT_TRUE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
  timer.Start();
  EXPECT_TRUE(timer.initted());
  EXPECT_TRUE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
  timer.Stop();
  timer.Start();
  EXPECT_TRUE(timer.initted());
  EXPECT_TRUE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
)

TEST_F_ALL_DEVICES(BenchmarkTest, TestTimerStop,
  Timer timer;
  timer.Stop();
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
  timer.Start();
  timer.Stop();
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
  timer.Stop();
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
)

TEST_F_ALL_DEVICES(BenchmarkTest, TestTimerMilliSeconds,
  Timer timer;
  CHECK_EQ(timer.MilliSeconds(), 0);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
  timer.Start();
  usleep(300 * 1000);
  CHECK_GE(timer.MilliSeconds(), 298);
  CHECK_LE(timer.MilliSeconds(), 302);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
)

TEST_F_ALL_DEVICES(BenchmarkTest, TestTimerSeconds,
  Timer timer;
  CHECK_EQ(timer.Seconds(), 0);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
  timer.Start();
  usleep(300 * 1000);
  CHECK_GE(timer.Seconds(), 0.298);
  CHECK_LE(timer.Seconds(), 0.302);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
)

}  // namespace caffe
