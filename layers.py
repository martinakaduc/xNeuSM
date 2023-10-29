import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature, nhop, directed=False, gpu=False):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature * 2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.zeros = torch.zeros(1)
        if gpu > 0:
            self.zeros = self.zeros.cuda()

        self.nhop = nhop
        self.directed = directed

    def forward(self, x, adj, get_attention=False):
        h = self.W(x)

        e = torch.einsum("ijl,ikl->ijk", (torch.matmul(h, self.A), h))
        if not self.directed:
            e = e + e.permute((0, 2, 1))

        attention = torch.where(adj > 0, e, self.zeros)
        attention = F.softmax(attention, dim=1)
        attention = attention * adj

        z = h
        az = F.relu(torch.einsum("aij,ajk->aik", (attention, z)))
        coeff = torch.sigmoid(self.gate(torch.cat([h, az], -1))).repeat(
            1, 1, h.size(-1)
        )
        for _ in range(self.nhop):
            az = F.relu(torch.einsum("aij,ajk->aik", (attention, z)))
            z = coeff * h + (1 - coeff) * az

        if get_attention:
            return z, attention
        return z
