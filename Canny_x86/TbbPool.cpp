#include "TbbPool.h"

TbbPool::TbbPool(unsigned int ThreadNum) : arena(ThreadNum), ActiveTasks(0) {}

TbbPool::~TbbPool() { Sync(); }

void TbbPool::Sync()
{
    std::unique_lock<std::mutex> Lock(QueueMutex);
    FinishedVar.wait(Lock, [this] { return ActiveTasks == 0; });
}