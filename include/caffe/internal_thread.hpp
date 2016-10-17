#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
//此基类封装了boost：thread，其子类可通过重写虚函数InternalThreadEntry将具有运行单个线程的能力，
//这是因为在封装好的start thread函数中调用了InteralThreadEntry了。
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  //改start thread的函数主要是使用caffe中的设备id， 求解器的index参数初始化thread线程
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  void StopInternalThread();

  bool is_started() const;

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}

  /* Should be tested when running loops to exit when requested. */
  bool must_stop();

 private:
  void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
      bool root_solver);

  shared_ptr<boost::thread> thread_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
