import torch
import torch.nn as nn
import torch.nn.functional as F


class GLeMa(torch.nn.Module):
    def __init__(
        self,
        n_in_feature,
        n_out_feature,
        nhop,
        nhead=1,
        aggregation="mean",
        directed=False,
    ):
        super(GLeMa, self).__init__()
        self.W_h = nn.Linear(n_in_feature, n_out_feature * nhead)
        self.W_e = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.W_beta = nn.Linear(n_out_feature * 2, 1)

        assert aggregation in ["mean", "weight"], "Unknown aggregation"
        self.aggr = aggregation
        if aggregation == "weight":
            self.W_o = nn.Linear(n_out_feature * nhead, n_out_feature, bias=False)

        self.nhop = nhop
        self.nhead = nhead
        self.hidden_dim = n_out_feature
        self.directed = directed

    def __aggregate__(self, z):
        if self.aggr == "mean":
            return z.mean(-2)  # mean over heads
        else:
            z = z.reshape(z.size(0), -1, self.nhead * self.hidden_dim)
            return self.W_o(z)

    def forward(self, x, adj, get_attention=False):
        # Embedding
        h = self.W_h(x)
        h = h.view(h.size(0), -1, self.nhead, self.hidden_dim)

        # Attention
        e = torch.einsum("bjil,bkil->bjik", (torch.matmul(h, self.W_e), h))
        if not self.directed:
            e = e + e.permute((0, 3, 2, 1))

        attention = e * (adj > 0).unsqueeze(2).repeat(1, 1, self.nhead, 1)
        attention = F.softmax(attention, dim=1)
        attention = attention * adj.unsqueeze(2).repeat(1, 1, self.nhead, 1)

        # Multi-hop attention
        z = h
        az = F.relu(torch.einsum("biaj,bjak->biak", (attention, z)))
        beta = torch.sigmoid(self.W_beta(torch.cat([h, az], -1))).repeat(
            1, 1, 1, self.hidden_dim
        )
        for _ in range(self.nhop):
            az = F.relu(torch.einsum("biaj,bjak->biak", (attention, z)))
            z = beta * h + (1 - beta) * az

        # Output
        z = self.__aggregate__(z)

        if get_attention:
            return z, attention.mean(2)
        return z


class GLeMaNet(torch.nn.Module):
    def __init__(self, args):
        super(GLeMaNet, self).__init__()
        n_graph_layer = args.n_graph_layer
        d_graph_layer = args.d_graph_layer
        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer
        self.dropout_rate = args.dropout_rate
        self.branch = args.branch

        if args.tatic == "static":

            def cal_nhop(x):
                return args.nhop

        elif args.tatic == "cont":

            def cal_nhop(x):
                return x + 1

        elif args.tatic == "jump":

            def cal_nhop(x):
                return 2 * x + 1

        else:
            raise ValueError("Unknown multi-hop tatic: {}".format(args.tatic))

        self.layers1 = [d_graph_layer for i in range(n_graph_layer + 1)]
        self.gconv1 = nn.ModuleList(
            [
                GLeMa(
                    n_in_feature=self.layers1[i],
                    n_out_feature=self.layers1[i + 1],
                    nhop=cal_nhop(i),
                    nhead=args.nhead,
                    directed=args.directed,
                )
                for i in range(len(self.layers1) - 1)
            ]
        )

        self.FC = nn.ModuleList(
            [
                (
                    nn.Linear(self.layers1[-1], d_FC_layer)
                    if i == 0
                    else (
                        nn.Linear(d_FC_layer, 1)
                        if i == n_FC_layer - 1
                        else nn.Linear(d_FC_layer, d_FC_layer)
                    )
                )
                for i in range(n_FC_layer)
            ]
        )

        self.embede = nn.Linear(2 * args.embedding_dim, d_graph_layer, bias=False)
        self.theta = args.al_scale
        self.zeros = torch.zeros(1)
        if args.ngpu > 0:
            self.zeros = self.zeros.cuda()

    def embede_graph(self, X):
        c_hs, c_adjs1, c_adjs2, c_valid = X
        c_hs = self.embede(c_hs)
        attention = None

        for k in range(len(self.gconv1)):
            if self.branch == "left":
                if k == len(self.gconv1) - 1:
                    c_hs1, attention = self.gconv1[k](c_hs, c_adjs1, True)
                else:
                    c_hs1 = self.gconv1[k](c_hs, c_adjs1)
                c_hs1 = -c_hs1
            elif self.branch == "right":
                c_hs1 = 0
            else:
                c_hs1 = self.gconv1[k](c_hs, c_adjs1)

            if self.branch == "left":
                c_hs2 = 0
            else:
                if k == len(self.gconv1) - 1:
                    c_hs2, attention = self.gconv1[k](c_hs, c_adjs2, True)
                else:
                    c_hs2 = self.gconv1[k](c_hs, c_adjs2)

            c_hs = c_hs2 - c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)

        c_hs = c_hs * c_valid.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
        c_hs = c_hs.sum(1) / c_valid.sum(1, keepdim=True)
        return c_hs, F.normalize(attention)

    def fully_connected(self, c_hs):
        for k in range(len(self.FC)):
            if k < len(self.FC) - 1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)

        c_hs = torch.sigmoid(c_hs)

        return c_hs

    def forward(self, X, attn_masking=None, training=False):
        # embede a graph to a vector
        c_hs, attention = self.embede_graph(X)

        # fully connected NN
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)

        if training:
            return c_hs, self.cal_attn_loss(attention, attn_masking)
        else:
            return c_hs

    def cal_attn_loss(self, attention, attn_masking):
        mapping, samelb = attn_masking

        top = torch.exp(-(attention * mapping))
        top = torch.where(mapping == 1.0, top, self.zeros)
        top = top.sum((1, 2))

        bot = torch.exp(-(attention * (samelb - mapping)))
        bot = torch.where((samelb - mapping) == 1.0, bot, self.zeros)
        bot = bot.sum((1, 2))

        return (top / (bot + 1)).sum(0) * self.theta / attention.shape[0]

    def get_refined_adjs2(self, X):
        _, attention = self.embede_graph(X)
        return attention
