#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe 
{

using boost::weak_ptr;

//body是string和data body的map。注意这里的DataReader::bodies_在类里面声明的是static，说明这个bodies是个所有的类对象
//共用的变量, 即用这个bodies来存储所有的reader的行为

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;

static boost::mutex bodies_mutex_;  //静态全局变量



// LayerParameter是prototxt中定义的结构体，
DataReader::DataReader(const LayerParameter& param)
    : queue_pair_( new QueuePair( param.data_param().prefetch() * param.data_param().batch_size() ) ) 
{
//注意上面的queue_pair_的new操作，说明此时free_的datum中是没有数据的
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];   
  body_ = weak.lock();

  //实体body指针构建成功，分配相应的内存，并将map中的两部分关联起来
  if (!body_) {
    body_.reset(new Body(param));    //参数传入到body里去
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);  //这里的body只存进了一个queue_pair_, 不过这个body只是bodies的一个键值
}

DataReader::~DataReader() 
{
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//
DataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

DataReader::QueuePair::~QueuePair() {
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

DataReader::Body::Body(const LayerParameter& param)
    : param_(param), new_queue_pairs_() {
  StartInternalThread();
}

DataReader::Body::~Body() {
  StopInternalThread();
}

void DataReader::Body::InternalThreadEntry()
{
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));  //根据参数参数lmdb new出一个lmdb子类构造了一个db父类，这会造成子类数据结构被截断
  db->Open(param_.data_param().source(), db::READ);                 //根据source的文件地址打开文件，这里使用了lmdb的open接口，尽管遭到构造截断，但是父类db提供了open接口，接口函数地址相同，不妨碍使用结果
  shared_ptr<db::Cursor> cursor(db->NewCursor());					//同样使用了lmdb的NewCursor的函数接口，这个例子很好的体现了纯虚函数的用途。
  vector<shared_ptr<QueuePair> > qps;								//queue pair的share_ptr模式放入vector容器中
  try 
  {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;  //支持多卡训练，不支持多卡test，这个很显然的

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.

//									qp->free_.pop()	<----new_queue_pairs_.pop().get()
//									   ^					  ^
//									  /	\					 / \
//                                     |                      |
//								  / full_(BQ)   ----|
//	queue --\                    /					|
//			 > BlockingQueue  -->					  --> QueuePair   --\
//	sync  --/					 \					|					 \
//								  \ free_(BQ)	----|					  > BlockingQueue(new_queue_pairs_)																			
//																		 /
//																sync  --/
//
//细心理清这里的结构，可以得到每个solver只读取了一个item，然后等待下一个求解器
//这是为了初始化,之后qps中还存储了各自的指向的记录位置，方便接下来各个solver的数据读取
	for (int i = 0; i < solver_count; ++i) 
	{
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(cursor.get(), qp.get());
      qps.push_back(qp);
    }

// Main loop
//在主循环中，各个solver分别读取自己的记录，不过也是大家排队挨个去读取。
//问题出在new_queue_pairs_是如何给大家最外层分配数据的
    while (!must_stop()) 
	{
      for (int i = 0; i < solver_count; ++i) 
	  {
        read_one(cursor.get(), qps[i].get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } 
  catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) 
{ 

  Datum* datum = qp->free_.pop();                  //数据库记录放在free_中，free表明Datum中数据为空
  // TODO deserialize in-place instead of copy?
  datum->ParseFromString(cursor->value());         //对数据库记录解析
  qp->full_.push(datum);						   //解析结果存放在full_中，full表示Datum中数据已填充
  //揣测的套路是原先记录放在free_，并且解析一个丢弃一个，并且将解析结果存放在full_中

  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

}  // namespace caffe
