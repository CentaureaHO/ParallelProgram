template <class F, class... Args>
std::future<RetType<F, Args...>> TbbPool::EnQueue(F&& ThFunc, Args&&... args)
{
    auto Task = std::make_shared<std::packaged_task<RetType<F, Args...>()>>(
        std::bind(std::forward<F>(ThFunc), std::forward<Args>(args)...));

    std::future<RetType<F, Args...>> Res = Task->get_future();
    {
        std::unique_lock<std::mutex> Lock(QueueMutex);
        ++ActiveTasks;
    }
    arena.enqueue([Task, this] {
        (*Task)();
        {
            std::unique_lock<std::mutex> Lock(QueueMutex);
            --ActiveTasks;
            if (ActiveTasks == 0) FinishedVar.notify_all();
        }
    });
    return Res;
}