import numpy as np
from mpi4py import MPI
from redistribution import redistribute, sequential_redistribution


all_ncopies = [np.array([8, 1, 2, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # 0
               np.array([0, 3, 0, 0, 3, 0, 0, 0, 5, 0, 1, 1, 0, 2, 1, 0]),  # 1
               np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16]),  # 2
               np.array([16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # 3
               np.array([15, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # 4
               np.array([4, 0, 2, 2, 0, 4, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0]),  # 5
               np.array([2, 0, 2, 0, 2, 1, 0, 1, 1, 0, 0, 2, 3, 1, 0, 1]),  # 6
               np.array([5, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0]),  # 7
               np.array([7, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]),  # 8
               np.array([0, 3, 0, 0, 3, 0, 0, 0, 5, 0, 1, 1, 0, 2, 1, 0]),  # 9
               np.array([0, 5, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 1, 0]),  # 10
               np.array([5, 9, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # 11
               np.array([1, 2, 3, 1, 3, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]),  # 12
               np.array([2, 2, 0, 0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 1, 3, 1]),  # 13
               np.array([2, 2, 1, 1, 1, 1, 2, 1, 1, 3, 1, 0, 0, 0, 0, 0]),  # 14
               np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),  # 15
               np.array([0, 0, 0, 0, 13, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]),  # 16
               np.array([0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]),  # 17
               np.array([13, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # 18
               np.array([1, 0, 1, 0, 2, 1, 0, 0, 2, 1, 2, 1, 1, 0, 3, 1]),  # 19
               np.array([1, 0, 4, 0, 1, 0, 0, 0, 4, 0, 0, 1, 4, 0, 1, 0]),  # 20
               np.array([8, 2, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]),  # 21
               np.array([0, 8, 1, 0, 0, 1, 0, 0, 3, 0, 1, 0, 0, 0, 2, 0]),  # 22
               np.array([7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # 23
               np.array([4, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # 24
               np.array([8, 1, 2, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]  # 25

for test in range(len(all_ncopies)):
    comm = MPI.COMM_WORLD
    P = comm.Get_size()
    N = len(all_ncopies[test])
    loc_n = int(N/P)
    rank = comm.Get_rank()
    x = np.array([np.repeat(i+rank*loc_n, 2) for i in range(loc_n)]).astype('d')
    ncopies = all_ncopies[test][rank*loc_n: rank*loc_n+loc_n]
    all_x = np.zeros([N, x.shape[1]], dtype='d')
    comm.Gather(sendbuf=[x, MPI.DOUBLE], recvbuf=[all_x, MPI.DOUBLE], root=0)

    x = redistribute(x, ncopies)

    all_x_test = np.zeros([N, x.shape[1]], dtype='d')
    comm.Gather(sendbuf=[x, MPI.DOUBLE], recvbuf=[all_x_test, MPI.DOUBLE], root=0)

    if rank == 0:
        if sum(all_x_test.flatten() - sequential_redistribution(all_x, all_ncopies[test]).flatten()) == 0.0:
            print("test " + str(test) + ": Passed")
        else:
            print("test " + str(test) + ": Failed")

