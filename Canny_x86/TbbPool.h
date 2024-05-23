#pragma once

#include <tbb/tbb.h>
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

class TbbPool
{
private:
    tbb::task_arena             arena;
    tbb::task_group             task_group;
    std::mutex                  QueueMutex;
    std::condition_variable     FinishedVar;
    unsigned int                ActiveTasks;

public:
    TbbPool(unsigned int ThreadNum);
    ~TbbPool();

    template <class F, class... Args>
    std::future<RetType<F, Args...>> EnQueue(F&& ThFunc, Args&&... args);
    void                             Sync();
};

#include "TbbPool.tpp"