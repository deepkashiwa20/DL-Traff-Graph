import torch
from torch import nn
import sys
import numpy as np
import scipy.sparse as sp
from torchsummary import summary
from Utils import load_pickle
import pandas as pd

class GCN(nn.Module):
    def __init__(self, K: int, input_dim: int, hidden_dim: int, bias=True, activation=nn.ReLU):
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.activation = activation() if activation is not None else None
        self.init_params(n_supports=K)

    def init_params(self, n_supports: int, b_init=0):
        self.W = nn.Parameter(torch.empty(n_supports * self.input_dim, self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(
            self.W)  # sampled from a normal distribution N(0, std^2), also known as Glorot initialization
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)

    def forward(self, G: torch.Tensor, x: torch.Tensor):
        '''
        Batch-wise graph convolution operation on given n support adj matrices
        :param G: support adj matrices - torch.Tensor (K, n_nodes, n_nodes)
        :param x: graph feature/signal - torch.Tensor (batch_size, n_nodes, input_dim)
        :return: hidden representation - torch.Tensor (batch_size, n_nodes, hidden_dim)
        '''
        assert self.K == G.shape[0]

        support_list = list()
        for k in range(self.K):
            support = torch.einsum('ij,bjp->bip', [G[k, :, :], x])
            support_list.append(support)
        support_cat = torch.cat(support_list, dim=-1)

        output = torch.einsum('bip,pq->biq', [support_cat, self.W])
        if self.bias:
            output += self.b
        output = self.activation(output) if self.activation is not None else output
        return output

    def __repr__(self):
        return self.__class__.__name__ + f'({self.K} * input {self.input_dim} -> hidden {self.hidden_dim})'


class DCGRU_Cell(nn.Module):
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int, K: int, bias=True, activation=None):
        super(DCGRU_Cell, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv_gate = GCN(K=K,
                             input_dim=input_dim + hidden_dim,
                             hidden_dim=hidden_dim * 2,  # for update_gate, reset_gate
                             bias=bias,
                             activation=activation)
        self.conv_cand = GCN(K=K,
                             input_dim=input_dim + hidden_dim,
                             hidden_dim=hidden_dim,  # for candidate
                             bias=bias,
                             activation=activation)

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(batch_size, self.num_nodes, self.hidden_dim))
        return hidden

    def forward(self, P: torch.Tensor, x_t: torch.Tensor, h_t_1: torch.Tensor):
        assert len(P.shape) == len(x_t.shape) == len(
            h_t_1.shape) == 3, 'DCGRU cell must take in 3D tensor as input [x, h]'

        x_h = torch.cat([x_t, h_t_1], dim=-1)
        x_h_conv = self.conv_gate(G=P, x=x_h)

        z, r = torch.split(x_h_conv, self.hidden_dim, dim=-1)
        update_gate = torch.sigmoid(z)
        reset_gate = torch.sigmoid(r)

        candidate = torch.cat([x_t, reset_gate * h_t_1], dim=-1)
        cand_conv = torch.tanh(self.conv_cand(G=P, x=candidate))

        h_t = (1.0 - update_gate) * h_t_1 + update_gate * cand_conv
        return h_t


class DCGRU_Encoder(nn.Module):
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim, K: int, num_layers: int,
                 bias=True, activation=None, return_all_layers=False):
        super(DCGRU_Encoder, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.append(DCGRU_Cell(num_nodes=num_nodes,
                                             input_dim=cur_input_dim,
                                             hidden_dim=self.hidden_dim[i],
                                             K=K,
                                             bias=bias,
                                             activation=activation))

    def forward(self, P: torch.Tensor, x_seq: torch.Tensor, h_0_l=None):
        '''
            P: (K, N, N)
            x_seq: (B, T, N, C)
            h_0_l: [(B, N, C)] * L
            return - out_seq_lst: [(B, T, N, C)] * L
                     h_t_lst: [(B, N, C)] * L
        '''
        assert len(x_seq.shape) == 4, 'DCGRU must take in 4D tensor as input x_seq'
        batch_size, seq_len, _, _ = x_seq.shape
        if h_0_l is None:
            h_0_l = self._init_hidden(batch_size)

        out_seq_lst = list()  # layerwise output seq
        h_t_lst = list()  # layerwise last state
        in_seq_l = x_seq  # current input seq

        for l in range(self.num_layers):
            h_t = h_0_l[l]
            out_seq_l = list()
            for t in range(seq_len):
                h_t = self.cell_list[l](P=P, x_t=in_seq_l[:, t, :, :], h_t_1=h_t)
                out_seq_l.append(h_t)

            out_seq_l = torch.stack(out_seq_l, dim=1)  # (B, T, N, C)
            in_seq_l = out_seq_l  # update input seq

            out_seq_lst.append(out_seq_l)
            h_t_lst.append(h_t)

        if not self.return_all_layers:
            out_seq_lst = out_seq_lst[-1:]
            h_t_lst = h_t_lst[-1:]
        return out_seq_lst, h_t_lst

    def _init_hidden(self, batch_size: int):
        h_0_l = []
        for i in range(self.num_layers):
            h_0_l.append(self.cell_list[i].init_hidden(batch_size))
        return h_0_l

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class DCGRU_Decoder(nn.Module):  # projected output as input at the next timestep
    def __init__(self, num_nodes: int, out_horizon: int, out_dim: int, hidden_dim, K: int, num_layers: int,
                 bias=True, activation=None):
        super(DCGRU_Decoder, self).__init__()
        self.num_nodes = num_nodes
        self.out_horizon = out_horizon  # output steps
        self.out_dim = out_dim
        self.hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        self.num_layers = num_layers
        self.bias = bias
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.out_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.append(DCGRU_Cell(num_nodes=num_nodes,
                                             input_dim=cur_input_dim,
                                             hidden_dim=self.hidden_dim[i],
                                             K=K,
                                             bias=bias,
                                             activation=activation))
        # self.out_projector = nn.Sequential(nn.Linear(in_features=self.hidden_dim[-1], out_features=out_dim, bias=bias), nn.ReLU())
        self.out_projector = nn.Linear(in_features=self.hidden_dim[-1], out_features=out_dim, bias=bias)

    def forward(self, P: torch.Tensor, x_t: torch.Tensor, h_0_l: list):
        '''
            P: (K, N, N)
            x_t: (B, N, C)
            h_0_l: [(B, N, C)] * L
        '''
        assert len(x_t.shape) == 3, 'DCGRU cell decoder must take in 3D tensor as input x_t'

        h_t_lst = list()  # layerwise hidden state
        x_in_l = x_t

        for l in range(self.num_layers):
            h_t_l = self.cell_list[l](P=P, x_t=x_in_l, h_t_1=h_0_l[l])
            h_t_lst.append(h_t_l)
            x_in_l = h_t_l  # update input for next layer

        output = self.out_projector(h_t_l)  # output
        return output, h_t_lst

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class DCRNN(nn.Module):
    def __init__(self, device, num_nodes: int, input_dim: int, out_horizon: int, P: list, K=3, hidden_dim=64, num_layers=2, bias=True,
                 activation=None):
        super(DCRNN, self).__init__()
        self.K = K
        self.P = self.compute_cheby_poly(P).to(device)  
        self.encoder = DCGRU_Encoder(num_nodes=num_nodes, input_dim=input_dim, hidden_dim=hidden_dim, K=self.P.shape[0],
                                     num_layers=num_layers, bias=bias, activation=activation, return_all_layers=True)
        self.decoder = DCGRU_Decoder(num_nodes=num_nodes, out_dim=input_dim, hidden_dim=hidden_dim, K=self.P.shape[0],
                                     num_layers=num_layers, bias=bias, activation=activation, out_horizon=out_horizon)

    def compute_cheby_poly(self, P: list):
        P_k = []
        for p in P:
            p = torch.from_numpy(p).float().T
            T_k = [torch.eye(p.shape[0]), p]    # order 0, 1
            for k in range(2, self.K):
                T_k.append(2*torch.mm(p, T_k[-1]) - T_k[-2])    # recurrent to order K
            P_k += T_k
        return torch.stack(P_k, dim=0)    # (K, N, N) or (2*K, N, N) for bidirection

    def forward(self, x_seq: torch.Tensor):
        '''
            x_seq: (B, T, N, C)
        '''
        assert len(x_seq.shape) == 4, 'DCGRU must take in 4D tensor as input x_seq'
        
        # encoding
        _, h_t_lst = self.encoder(P=self.P, x_seq=x_seq,
                                  h_0_l=None)  # encoder returns layerwise last hidden state [(B, N, C)] * L
        # decoding
        # deco_input = self.decoder.out_projector(h_t_lst[-1])        # initiate decoder input
        deco_input = torch.zeros((x_seq.shape[0], x_seq.shape[2], x_seq.shape[3]),
                                 device=x_seq.device)  # original initialization

        outputs = list()
        for t in range(self.decoder.out_horizon):
            output, h_t_lst = self.decoder(P=self.P, x_t=deco_input, h_0_l=h_t_lst)
            deco_input = output  # update decoder input
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (B, horizon, N, C)
        return outputs


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = sp.linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_adj(pkl_filename, adjtype):
    adj_mx = pd.read_csv(pkl_filename).values
    distances = adj_mx[~np.isinf(adj_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(adj_mx / std))
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

def main():
    from Param import ADJPATH,CHANNEL, N_NODE, TIMESTEP_IN, TIMESTEP_OUT
    from Param_DCRNN import ADJTYPE
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    adj_mx = load_adj(ADJPATH, ADJTYPE)
    model = DCRNN(device, num_nodes=N_NODE, input_dim=CHANNEL, out_horizon=TIMESTEP_OUT, P=adj_mx).to(device)
    summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)

if __name__ == '__main__':
    main()