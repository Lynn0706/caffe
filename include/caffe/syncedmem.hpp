#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.

	//caffe在机器上分配和释放内存，如果机器上有gpu，也包括使用cuda进行分配和释放内存
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  free(ptr);
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  ~SyncedMemory();
  const void* cpu_data();				//获取cpu的data，注意const
  void set_cpu_data(void* data);		//设置cpu的data
  const void* gpu_data();				//获取gpu的data
  void set_gpu_data(void* data);		//设置gpu的data
  void* mutable_cpu_data();				//获取cpu的data，该操作有可能是改变cpu的data之后再获取的，原因它可能先需要从gpu中获取数据更新cpu数据
  void* mutable_gpu_data();				//获取gpu的data，该操作有可能是改变gpu的data之后再获取的，原因它可能先需要从cpu中获取数据更新gpu数据
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED }; //定义了四中cpu和gpu数据更新状态
  SyncedHead head() { return head_; }	//获取数据众泰
  size_t size() { return size_; }		//获取data的size

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);  //异步向cuda数据流推送cpu中的数据，即异步更新数据到gpu
#endif

 private:
  void to_cpu();				//如果需要从gpu中更新，将gpu中的数据拷贝到cpu中，若cpu数据较新的话，不操作
  void to_gpu();				//如果需要从cpu中更新，将cpu中的数据拷贝到gpu中，若gpu数据较新的话，不操作
  void* cpu_ptr_;				//数据在cpu中位置
  void* gpu_ptr_;				//数据在gpu中的位置
  size_t size_;					//数据块的大小
  SyncedHead head_;				//表明当前的sync 表明哪里的数据是最新的（be head of ...），可以使在cpu，gpu，或者是uninitialized,或者是sync的
								//分别表示cpu数据最新，gpu数据最新，或者该SyncMemory第一次使用还未被初始化（即都不在，按照数据使用CPU读取，再通信同步到
								//GPU的存储体系，在初次初始化之后，应该将标志位置位CPU head）和cpu和gpu刚刚进行同步
  bool own_cpu_data_;			//是否使用了cpu操作数据
  bool cpu_malloc_use_cuda_;    //是否使用cuda分配内存
  bool own_gpu_data_;			//是否使用了gpu操作数据
  int gpu_device_;				//可以使用多卡，记录使用所在的gpu卡设备编号

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
