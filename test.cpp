#include <mpi.h>
#include <iostream>
#include <mutex>

class Singleton
{
  public:
    static Singleton* getInstance()
    {
        std::call_once(initFlag, &Singleton::initSingleton);
        return instance;
    }

    void printAddress() const { std::cout << "Singleton instance address: " << this << std::endl; }

  private:
    Singleton()  = default;
    ~Singleton() = default;

    Singleton(const Singleton&)            = delete;
    Singleton& operator=(const Singleton&) = delete;

    static void initSingleton() { instance = new Singleton(); }

    static Singleton*     instance;
    static std::once_flag initFlag;
};

Singleton*     Singleton::instance = nullptr;
std::once_flag Singleton::initFlag;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Singleton* singleton = Singleton::getInstance();
    std::cout << "Process " << rank << ": ";
    singleton->printAddress();

    MPI_Finalize();
    return 0;
}
