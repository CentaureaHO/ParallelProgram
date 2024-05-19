#include "ThreadPool.h"

void* ThreadEntry(void* args)
{
    auto* Pool = static_cast<ThreadPool*>(args);
    Pool->Run();
    return nullptr;
}

ThreadPool::ThreadPool(unsigned int ThreadNum) : Stop(0), ActiveTasks(0)
{
    Workers.resize(ThreadNum);
    for (auto& Worker : Workers) { pthread_create(&Worker, nullptr, ThreadEntry, this); }
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(QueueMutex);
        Stop = 1;
    }
    CondVar.notify_all();
    for (auto& Worker : Workers) { pthread_join(Worker, nullptr); }
}

void ThreadPool::Run()
{
    while (1 == 1)
    {
        std::function<void()> Task;
        {
            std::unique_lock<std::mutex> Lock(QueueMutex);
            CondVar.wait(Lock, [this] { return Stop || !Tasks.empty(); });
            if (Stop && Tasks.empty()) break;
            Task = std::move(Tasks.front());
            Tasks.pop();
            ++ActiveTasks;
        }
        Task();
        {
            std::unique_lock<std::mutex> Lock(QueueMutex);
            --ActiveTasks;
            if (ActiveTasks == 0 && Tasks.empty()) FinishedVar.notify_all();
        }
    }
}

void ThreadPool::Sync()
{
    std::unique_lock<std::mutex> Lock(QueueMutex);
    FinishedVar.wait(Lock, [this] { return Tasks.empty() && ActiveTasks == 0; });
}
