#pragma once

#include <pthread.h>
#include <queue>
#include <functional>
#include <future>
#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>

template <class F, class... Args>
using RetType = typename std::result_of<F(Args...)>::type;

class ThreadPool
{
  private:
    unsigned int                      ActiveTasks;
    std::condition_variable           CondVar;
    std::condition_variable           FinishedVar;
    std::mutex                        QueueMutex;
    bool                              Stop;
    std::queue<std::function<void()>> Tasks;
    std::vector<pthread_t>            Workers;

  public:
    ThreadPool(unsigned int ThreadNum);
    ~ThreadPool();

  public:
    template <class F, class... Args>
    std::future<RetType<F, Args...>> EnQueue(F&& ThFunc, Args&&... args);
    void                             Run();
    void                             Sync();
};

#include "ThreadPool.tpp"
