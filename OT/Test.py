import numpy as np
from mpi4py import MPI
import time

def parameter_server(data, rank, size, comm, root=0):
    """ 使用Parameter Server模式汇总数据 """
    if rank == root:
        total_data = np.copy(data)
        for i in range(1, size):
            total_data += comm.recv(source=i, tag=11)
    else:
        comm.send(data, dest=root, tag=11)
    return comm.bcast(total_data if rank == root else None, root=root)

def ring_allreduce(data, rank, size, comm):
    """ 使用Ring AllReduce模式汇总数据 """
    send_data = np.copy(data)
    recv_data = np.zeros_like(data)
    for i in range(1, size):
        send_rank = (rank + i) % size
        recv_rank = (rank - i + size) % size
        req = comm.Isend([send_data, MPI.DOUBLE], dest=send_rank, tag=15)
        comm.Recv([recv_data, MPI.DOUBLE], source=recv_rank, tag=15)
        req.Wait()
        send_data = recv_data
        data += recv_data
    return data

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = np.ones(100000000) * rank

    start_time = time.time()
    total_data_ps = parameter_server(data, rank, size, comm)
    ps_time = time.time() - start_time

    start_time = time.time()
    total_data_ra = ring_allreduce(data, rank, size, comm)
    ra_time = time.time() - start_time

    if rank == 0:
        print(f"Parameter Server Time: {ps_time} seconds")
        print(f"Ring AllReduce Time: {ra_time} seconds")

if __name__ == "__main__":
    main()
