template <class F, class... Args>
std::future<RetType<F, Args...>> ThreadPool::EnQueue(F&& ThFunc, Args&&... args)
{
    auto Task = std::make_shared<std::packaged_task<RetType<F, Args...>()>>(
        std::bind(std::forward<F>(ThFunc), std::forward<Args>(args)...));

    std::future<RetType<F, Args...>> Res = Task->get_future();
    {
        std::unique_lock<std::mutex> Lock(QueueMutex);
        Tasks.emplace([Task] { (*Task)(); });
    }
    CondVar.notify_one();
    return Res;
}
