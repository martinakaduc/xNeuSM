from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    
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
                f.write("e %d %d %d\n" % (nid1, nid2, g.edges[(nod1, nod2)]["label"]))


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


def initialize_model(model, device, load_save_file=False, gpu=True):
    if load_save_file:
        if not gpu:
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
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def node_feature(m, node_i, max_nodes):
    node = m.nodes[node_i]
    return onehot_encoding(node["label"], max_nodes)
