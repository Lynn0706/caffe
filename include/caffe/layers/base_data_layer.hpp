#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
//提供了数据结构batch将data和lable成对存放
//prefetch layer 用于开辟线程预读取数据
//base layer 用于作为datalayer的父类，给data layer， image layer， window layer提供基类

template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  //做些通用的操作，具体的DataLayerSetUp交由子类具体实现。
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /*
  template <typename Dtype>
  void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
  {
	  //输出的blob的个数为1，那么只可能计算数据，而包括标签
	  if (top.size() == 1) 
	  {
	  output_labels_ = false;
	  } else {
	  output_labels_ = true;
	  }
	  //对数据的transform做初始化，比如随机数
	  data_transformer_.reset( new DataTransformer<Dtype>(transform_param_, this->phase_));
	  data_transformer_->InitRand();
	  // The subclasses should setup the size of bottom and top
	  DataLayerSetUp(bottom, top);
  }
  */
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }

  //提供接口，默认数据层不做data上的setup, 但根据注释The subclasses should setup the size of bottom and top，应该是必须做操作，这里是因为LayerSetUp调用了该函数，该函数不能是纯虚函数。
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  //提供接口，默认数据层不做reshape，因为没有bottom，所以data layer的bottom 的blob数最小可以为零
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  
  //提供接口，默认数据层不做任何后向计算，除非重写，比如data layer就没有前后向计算。
  //至于前向计算，本质也是没有的，但为了加载数据，前向的操作在prefetchdatalayer中其实是有操作的
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;					//数据做变换的参数
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_;
};


template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread 
{
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  //读取数据肯定是cpu先去读取，根据是否有gpu去做同步，预读取的过程只需要数据异步同步即可
  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  Batch<Dtype> prefetch_[PREFETCH_COUNT];					//读取解析好的数据（data和label都有）存放空间
  BlockingQueue<Batch<Dtype>*> prefetch_free_;				//提供batch的容器放在free里，初始化时分配好空间，每读取一条记录丢弃一个
  BlockingQueue<Batch<Dtype>*> prefetch_full_;				//预读取的数据的数据放在full里

  Blob<Dtype> transformed_data_;							//转换后的参数？？？？
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
