#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */

//对数据库的读写涉及到多线程编程时，需要考虑互斥问题。
//为提高了读取效率，这里将读取资源source到queue中供data layer中读取。
//一个资源只允许单独一个读取线程操作。即使是在多卡gpu训练时，多个solver同时并行运行的同时也必须这样
//上述这种策略保证了数据库在访问时是顺序读取的，而且每个solver访问的是完全不同的数据库的子集
//各个solver使用了一个叫round-robin的方式去访问数据库，这使得在并行化训练时，数据是以一种确定性的
//方式分发给solvers的

/* 在DataReader中，reader读取queue中的数据库记录把内容存放在body中*/
class DataReader 
{
 public:
  explicit DataReader(const LayerParameter& param);
  ~DataReader();

  inline BlockingQueue<Datum*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<Datum*>& full() const {
    return queue_pair_->full_;
  }

 protected:
  // Queue pairs are shared between a body and its readers
  class QueuePair {
   public:
    explicit QueuePair(int size);  //size = prefetch * batch
    ~QueuePair();

    BlockingQueue<Datum*> free_;  //队列中释放出来的Datum的集合，该数据结构支持push,pop,peek等操作且是个支持sync的mutex
    BlockingQueue<Datum*> full_;  //队列中full的Datum的集合
	//根据以上可以猜测，这类free_ full_为一个pair 其用途？？？

  DISABLE_COPY_AND_ASSIGN(QueuePair);
  };

  // A single body is created per source
  //一个资源创建一个单独的single body，根据之前的描述，这必须是一个single thread读取的
  class Body : public InternalThread 
  {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();

   protected:
    void InternalThreadEntry();								//起读数据的线程的入口
    void read_one(db::Cursor* cursor, QueuePair* qp);       //读取数据

    const LayerParameter param_;							
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

    friend class DataReader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }

  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;

  static map<const string, boost::weak_ptr<DataReader::Body> > bodies_;

DISABLE_COPY_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
