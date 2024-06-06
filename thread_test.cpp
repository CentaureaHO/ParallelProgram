#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <functional>
#include <future>
#include <condition_variable>
#include <chrono>
#include <type_traits>

// 线程池类的定义
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
        std::unique_lock<std::mutex> Lock(QueueMutex);
        Stop = 1;
    }
    CondVar.notify_all();
    for (auto& Worker : Workers) { pthread_join(Worker, nullptr); }
}

void ThreadPool::Run()
{
    while (true)
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

// 矩阵乘法函数
void matrix_multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B,
    std::vector<std::vector<double>>& C, int start_row, int end_row)
{
    int N = B[0].size();
    for (int i = start_row; i < end_row; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) { C[i][j] += A[i][k] * B[k][j]; }
        }
    }
}

int main()
{
    const int N           = 1000;  // 矩阵大小 (NxN)
    const int num_threads = 15;    // 线程池中的线程数

    std::vector<std::vector<double>> A(N, std::vector<double>(N, 1.0));
    std::vector<std::vector<double>> B(N, std::vector<double>(N, 1.0));
    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));

    ThreadPool pool(num_threads);

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    // 将任务加入线程池
    int rows_per_thread = N / num_threads;
    for (int t = 0; t < num_threads; ++t)
    {
        int thread_start_row = t * rows_per_thread;
        int thread_end_row   = (t == num_threads - 1) ? N : thread_start_row + rows_per_thread;
        pool.EnQueue(matrix_multiply, std::ref(A), std::ref(B), std::ref(C), thread_start_row, thread_end_row);
    }

    // 等待所有任务完成
    pool.Sync();

    // 结束计时
    auto                                     end     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> elapsed = end - start;

    std::cout << "First element of C: " << C[0][0] << "\n";
    std::cout << "Elapsed time: " << elapsed.count() << " ns" << std::endl;

    return 0;
}
