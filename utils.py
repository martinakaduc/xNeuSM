from collections import defaultdict
import os
import argparse
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import random
from matplotlib import pyplot as plt


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed", type=int, default=42)
    parser.add_argument("--lr", help="learning rate",
                        type=float, default=0.0001)
    parser.add_argument("--epoch", help="epoch", type=int, default=30)
    parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
    parser.add_argument("--dataset", help="dataset", type=str, default="tiny")
    parser.add_argument("--batch_size", help="batch_size",
                        type=int, default=32)
    parser.add_argument(
        "--num_workers", help="number of workers", type=int, default=os.cpu_count()
    )
    parser.add_argument(
        "--embedding_dim",
        help="node embedding dim aka number of distinct node label",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--tatic",
        help="tactic of defining number of hops",
        type=str,
        default="static",
        choices=["static", "cont", "jump"],
    )
    parser.add_argument("--directed", action="store_true",
                        help="directed graph")
    parser.add_argument("--nhop", help="number of hops", type=int, default=1)
    parser.add_argument(
        "--nhead", help="number of attention heads", type=int, default=1
    )
    parser.add_argument(
        "--branch",
        help="choosing branch",
        type=str,
        default="both",
        choices=["both", "left", "right"],
    )
    parser.add_argument(
        "--n_graph_layer", help="number of GNN layer", type=int, default=4
    )
    parser.add_argument(
        "--d_graph_layer", help="dimension of GNN layer", type=int, default=140
    )
    parser.add_argument(
        "--n_FC_layer", help="number of FC layer", type=int, default=4)
    parser.add_argument(
        "--d_FC_layer", help="dimension of FC layer", type=int, default=128
    )
    parser.add_argument(
        "--data_path", help="path to the data", type=str, default="data_processed"
    )
    parser.add_argument(
        "--save_dir",
        help="save directory of model parameter",
        type=str,
        default="save/",
    )
    parser.add_argument("--log_dir", help="logging directory",
                        type=str, default="log/")
    parser.add_argument(
        "--result_dir",
        help="save directory of model parameter",
        type=str,
        default="results/",
    )
    parser.add_argument("--dropout_rate", help="dropout_rate",
                        type=float, default=0.0)
    parser.add_argument("--al_scale", help="attn_loss scale",
                        type=float, default=1.0)
    parser.add_argument("--ckpt", help="Load ckpt file", type=str, default="")
    parser.add_argument(
        "--train_keys", help="train keys", type=str, default="train_keys.pkl"
    )
    parser.add_argument(
        "--test_keys", help="test keys", type=str, default="test_keys.pkl"
    )
    parser.add_argument(
        "--tag",
        help="Additional tag for saving and logging folder",
        type=str,
        default="",
    )
    parser.add_argument(
        "--confidence", help="isomorphism threshold", type=float, default=0.5
    )
    parser.add_argument(
        "--mapping_threshold", help="mapping threshold", type=float, default=1e-5
    )
    args = parser.parse_args()
    return args


def ensure_dir(dir, args):
    dir = os.path.join(dir, f"{args.dataset}_{args.tatic}")
    if args.tatic == "static":
        dir += f"{args.nhop}"

    if args.nhead > 1:
        dir += f"_nhead{args.nhead}"

    if args.branch != "both":
        dir += "_" + args.branch

    if args.directed:
        dir += "_directed"

    if args.tag != "":
        dir += "_" + args.tag

    # Make save dir if it doesn't exist
    if not os.path.isdir(dir):
        os.system("mkdir -p " + dir)

    return dir


def plotGraph(
    graph: nx.Graph,
    nodeLabels=None,
    showLabel=True,
    nodeColors=None,
    edgeColors=None,
    fig_name=None,
):
    pos = nx.circular_layout(graph)
    plt.figure()

    if not nodeLabels:
        nodeLabels = (
            {node: node for node in graph.nodes}
            if not showLabel
            else {node: (node, graph.nodes[node]["label"]) for node in graph.nodes}
        )
    if not nodeColors:
        nodeColors = "lime"
    if not edgeColors:
        edgeColors = "black"

    latex_code = nx.to_latex(
        graph,
        pos,
        node_label=nodeLabels,
    )
    with open(fig_name[:-3] + "txt", "w") as f:
        f.write(latex_code)

    nx.draw(
        graph,
        pos,
        edge_color=edgeColors,
        width=1,
        linewidths=0.1,
        node_size=500,
        node_color=nodeColors,
        alpha=0.9,
        labels=nodeLabels,
    )

    # edgeLabels = {}
    # for edge in graph.edges():
    #     edgeLabels[edge] = graph[edge[0]][edge[1]]["label"]
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edgeLabels, font_color='red')

    plt.axis("off")
    plt.savefig(fig_name, dpi=300)


def write_graphs(graphs, out_file_name):
    with open(out_file_name, "w", encoding="utf-8") as f:
        for i, g in enumerate(graphs):
            f.write("t # %d\n" % i)
            node_mapping = {}
            for nid, nod in enumerate(g.nodes):
                f.write("v %d %d\n" % (nid, g.nodes[nod]["label"]))
                node_mapping[nod] = nid

            for nod1, nod2 in g.edges:
                nid1 = node_mapping[nod1]
                nid2 = node_mapping[nod2]
                f.write("e %d %d %d\n" %
                        (nid1, nid2, g.edges[(nod1, nod2)]["label"]))


def read_mapping(mapping_file, sg2g=False):
    mapping = dict()
    with open(mapping_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
        tmapping, graph_cnt = None, 0
        for i, line in enumerate(lines):
            cols = line.split(" ")
            if cols[0] == "t":
                if tmapping is not None:
                    mapping[graph_cnt] = tmapping
                    tmapping = None
                if cols[-1] == "-1":
                    break

                tmapping = defaultdict(lambda: -1)
                graph_cnt = int(cols[2])

            elif cols[0] == "v":
                if sg2g:
                    tmapping[int(cols[1])] = int(cols[2])
                else:
                    tmapping[int(cols[2])] = int(cols[1])

        # adapt to input files that do not end with 't # -1'
        if tmapping is not None:
            mapping[graph_cnt] = tmapping

    return mapping


def read_graphs(database_file_name):
    graphs = dict()
    max_size = 0
    with open(database_file_name, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
        tgraph, graph_cnt = None, 0
        graph_size = 0
        for i, line in enumerate(lines):
            cols = line.split(" ")
            if cols[0] == "t":
                if tgraph is not None:
                    graphs[graph_cnt] = tgraph
                    if max_size < graph_size:
                        max_size = graph_size
                    graph_size = 0
                    tgraph = None
                if cols[-1] == "-1":
                    break

                tgraph = nx.Graph()
                graph_cnt = int(cols[2])

            elif cols[0] == "v":
                tgraph.add_node(int(cols[1]), label=int(cols[2]))
                graph_size += 1

            elif cols[0] == "e":
                tgraph.add_edge(int(cols[1]), int(cols[2]), label=int(cols[3]))

        # adapt to input files that do not end with 't # -1'
        if tgraph is not None:
            graphs[graph_cnt] = tgraph
            if max_size < graph_size:
                max_size = graph_size

    return graphs


def initialize_model(model, device, load_save_file=False):
    if load_save_file:
        if device == "cpu":
            model.load_state_dict(
                torch.load(load_save_file, map_location=torch.device("cpu"))
            )
        else:
            model.load_state_dict(torch.load(load_save_file))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
            else:
                nn.init.xavier_normal_(param)

    model.to(device)
    return model


def onehot_encoding(x, max_x):
    onehot_vector = [0] * max_x
    onehot_vector[x - 1] = 1  # label start from 1
    return onehot_vector


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def node_feature(m, node_i, max_nodes):
    node = m.nodes[node_i]
    return onehot_encoding(node["label"], max_nodes)
