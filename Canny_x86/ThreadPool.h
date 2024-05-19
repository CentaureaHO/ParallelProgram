#pragma once

#include <pthread.h>
#include <queue>
#include <functional>
#include <future>
#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <type_traits>

template <class F, class... Args>
using RetType = typename std::invoke_result<F, Args...>::type;

class ThreadPool
{
  private:
    std::vector<pthread_t>            Workers;
    std::queue<std::function<void()>> Tasks;
    std::mutex                        QueueMutex;
    std::condition_variable           CondVar;
    std::condition_variable           FinishedVar;
    bool                              Stop;
    unsigned int                      ActiveTasks;

  public:
    ThreadPool(unsigned int ThreadNum);
    ~ThreadPool();

    template <class F, class... Args>
    std::future<RetType<F, Args...>> EnQueue(F&& ThFunc, Args&&... args);
    void                             Run();
    void                             Sync();
};

#include "ThreadPool.tpp"
