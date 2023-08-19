import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
np.set_printoptions(suppress=True)


class Sinkhorn:

    def __init__(self, L, W, people_num, iter_num, eps):
        self.L = L
        self.W = W
        assert(len(L) == len(W))
        self.n = len(L)
        self.people_num = people_num
        self.num_iter = iter_num
        self.eps = eps
        self.multistage_i = 0

    def sinkhorn(self, k, cost_matrix, lambda_W_prev, lambda_L_prev):

        if k % 2 == 0:
            lambda_W = lambda_W_prev
            lambda_L = np.log(np.nansum(
                (np.exp(-lambda_W_prev - 1 - cost_matrix)).T
                / self.L, axis=0
            ))
        else:
            lambda_L = lambda_L_prev
            lambda_W = np.log(np.nansum(
                (np.exp(-lambda_L - 1 - cost_matrix.T)).T
                / self.W, axis=0
            ))

        return lambda_W, lambda_L

    def iterate(self, cost_matrix):

        lambda_L = np.zeros(self.n)
        lambda_W = np.zeros(self.n)
        # r = None

        for k in range(self.num_iter):

            lambda_Wn, lambda_Ln = self.sinkhorn(k, cost_matrix, lambda_W, lambda_L)

            delta = np.linalg.norm(np.concatenate((lambda_Ln - lambda_L,
                                                   lambda_Wn - lambda_W)))

            lambda_L, lambda_W = lambda_Ln, lambda_Wn

            if delta < self.eps:
                break

        r = self.rec_d_i_j(lambda_Ln, lambda_Wn, cost_matrix)
        # self.multistage_i += 1
        return r, lambda_L, lambda_W

    def rec_d_i_j(self, lambda_L, lambda_W, cost_matrix):
        er = np.exp(-1 - cost_matrix - (np.reshape(lambda_L, (self.n, 1)) + lambda_W))
        return er * self.people_num