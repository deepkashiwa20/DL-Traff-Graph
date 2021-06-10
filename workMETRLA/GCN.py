from torch import nn
import torch



class GCN(nn.Module):
    def __init__(self, K:int, input_dim:int, hidden_dim:int, bias=True, activation=nn.ReLU):
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.activation = activation() if activation is not None else None
        self.init_params(n_supports=K)

    def init_params(self, n_supports:int, b_init=0):
        self.W = nn.Parameter(torch.empty(n_supports*self.input_dim, self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W)      # sampled from a normal distribution N(0, std^2), also known as Glorot initialization
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)

    def forward(self, G:torch.Tensor, x:torch.Tensor):
        '''
        Batch-wise graph convolution operation on given n support adj matrices
        :param G: support adj matrices - torch.Tensor (K, n_nodes, n_nodes)
        :param x: graph feature/signal - torch.Tensor (batch_size, n_nodes, input_dim)
        :return: hidden representation - torch.Tensor (batch_size, n_nodes, hidden_dim)
        '''
        assert self.K == G.shape[0]

        support_list = list()
        for k in range(self.K):
            support = torch.einsum('ij,bjp->bip', [G[k,:,:], x])
            support_list.append(support)
        support_cat = torch.cat(support_list, dim=-1)

        output = torch.einsum('bip,pq->biq', [support_cat, self.W])
        if self.bias:
            output += self.b
        output = self.activation(output) if self.activation is not None else output
        return output

    def __repr__(self):
        return self.__class__.__name__ + f'({self.K} * input {self.input_dim} -> hidden {self.hidden_dim})'



class Adj_Processor():
    def __init__(self, kernel_type:str, K:int):
        self.kernel_type = kernel_type
        # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
        self.K = K if self.kernel_type != 'localpool' else 1
        # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

    def process(self, flow:torch.Tensor):
        '''
        Generate adjacency matrices
        :param flow: batch flow stat - (batch_size, Origin, Destination) torch.Tensor
        :return: processed adj matrices - (batch_size, K_supports, O, D) torch.Tensor
        '''
        batch_list = list()

        for b in range(flow.shape[0]):
            adj = flow[b, :, :]
            kernel_list = list()

            if self.kernel_type in ['localpool', 'chebyshev']:  # spectral
                adj_norm = self.symmetric_normalize(adj)
                if self.kernel_type == 'localpool':
                    localpool = torch.eye(adj_norm.shape[0]) + adj_norm  # same as add self-loop first
                    kernel_list.append(localpool)

                else:  # chebyshev
                    laplacian_norm = torch.eye(adj_norm.shape[0]) - adj_norm
                    laplacian_rescaled = self.rescale_laplacian(laplacian_norm)
                    kernel_list = self.compute_chebyshev_polynomials(laplacian_rescaled, kernel_list)

            elif self.kernel_type == 'random_walk_diffusion':  # spatial
                # diffuse k steps on transition matrix P
                P_forward = self.random_walk_normalize(adj)
                kernel_list = self.compute_chebyshev_polynomials(P_forward.T, kernel_list)

            elif self.kernel_type == 'dual_random_walk_diffusion':
                # diffuse k steps bidirectionally on transition matrix P
                P_forward = self.random_walk_normalize(adj)
                P_backward = self.random_walk_normalize(adj.T)
                forward_series, backward_series = [], []
                forward_series = self.compute_chebyshev_polynomials(P_forward.T, forward_series)
                backward_series = self.compute_chebyshev_polynomials(P_backward.T, backward_series)
                kernel_list += forward_series + backward_series[1:]  # 0-order Chebyshev polynomial is same: I

            else:
                raise ValueError('Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion, dual_random_walk_diffusion].')

            # print(f"Minibatch {b}: {self.kernel_type} kernel has {len(kernel_list)} support kernels.")
            kernels = torch.stack(kernel_list, dim=0)
            batch_list.append(kernels)
        batch_adj = torch.stack(batch_list, dim=0)
        return batch_adj

    @staticmethod
    def random_walk_normalize(A):   # asymmetric
        d_inv = torch.pow(A.sum(dim=1), -1)   # OD matrix Ai,j sum on j (axis=1)
        d_inv[torch.isinf(d_inv)] = 0.
        D = torch.diag(d_inv)
        P = torch.mm(D, A)
        return P

    @staticmethod
    def symmetric_normalize(A):
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A_norm = torch.mm(torch.mm(D, A), D)
        return A_norm

    @staticmethod
    def rescale_laplacian(L):
        # rescale laplacian to arccos range [-1,1] for input to Chebyshev polynomials of the first kind
        try:
            lambda_ = torch.eig(L)[0][:,0]      # get the real parts of eigenvalues
            lambda_max = lambda_.max()      # get the largest eigenvalue
        except:
            print("Eigen_value calculation didn't converge, using max_eigen_val=2 instead.")
            lambda_max = 2
        L_rescaled = (2 / lambda_max) * L - torch.eye(L.shape[0])
        return L_rescaled

    def compute_chebyshev_polynomials(self, x, T_k):
        # compute Chebyshev polynomials up to order k. Return a list of matrices.
        # print(f"Computing Chebyshev polynomials up to order {self.K}.")
        for k in range(self.K + 1):
            if k == 0:
                T_k.append(torch.eye(x.shape[0]))
            elif k == 1:
                T_k.append(x)
            else:
                T_k.append(2 * torch.mm(x, T_k[k-1]) - T_k[k-2])
        return T_k


