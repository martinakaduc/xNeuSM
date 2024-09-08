import argparse
import json
import os
import signal
from contextlib import contextmanager
from multiprocessing import Process
from random import choice, seed, shuffle

import networkx as nx
import numpy as np

from tqdm import tqdm


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise Exception("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic graphs")
    parser.add_argument(
        "--config", "-c", default="configs/base.json", type=str, help="Config file"
    )
    parser.add_argument("--cont", action="store_true", help="Continue generating")
    return parser.parse_args()


def read_config(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def add_labels(graph, NN, NE):
    nodes = np.array(list(graph.nodes))
    edges = np.array(list(graph.edges))

    node_labels = np.random.randint(1, NN + 1, len(nodes)).tolist()
    edge_labels = np.random.randint(1, NE + 1, len(edges)).tolist()

    labelled_nodes = [
        (nodes[k], {"label": node_labels[k], "color": "green"})
        for k in range(len(nodes))
    ]
    labelled_edges = [
        (edges[k][0], edges[k][1], {"label": edge_labels[k], "color": "green"})
        for k in range(len(edges))
    ]

    G = nx.Graph()
    G.add_nodes_from(labelled_nodes)
    G.add_edges_from(labelled_edges)

    return G


def node_match(first_node, second_node):
    return first_node["label"] == second_node["label"]


def edge_match(first_edge, second_edge):
    return first_edge["label"] == second_edge["label"]


def generate_iso_subgraph(
    graph, no_of_nodes, avg_degree, std_degree, number_label_node, number_label_edge
):
    graph_nodes = graph.number_of_nodes()
    node_ratio = no_of_nodes / graph_nodes
    if node_ratio > 1:
        node_ratio = 1

    min_edges = int(no_of_nodes * (avg_degree - std_degree) / 2)
    max_edges = int(no_of_nodes * (avg_degree + std_degree) / 2)
    subgraph = None
    iteration = 0

    while (
        subgraph is None
        or subgraph.number_of_nodes() < 2
        or not nx.is_connected(subgraph)
    ):
        chose_nodes = np.random.choice(
            [0, 1], size=graph_nodes, replace=True, p=[1 - node_ratio, node_ratio]
        )
        remove_nodes = np.where(chose_nodes == 0)[0]
        subgraph = graph.copy()
        subgraph.remove_nodes_from(remove_nodes)

        iteration += 1
        if iteration > 5:
            node_ratio *= 1.05
            if node_ratio > 1:
                node_ratio = 1
            iteration = 0

    # Remove for induced subgraph
    # high = subgraph.number_of_edges() - subgraph.number_of_nodes() + 2
    # if high > 0:
    #     modify_times = np.random.randint(0, high)
    #     for _ in range(modify_times):
    #         if subgraph.number_of_edges() <= min_edges:
    #             break
    #         subgraph = remove_random_edge(subgraph)

    return subgraph


def remove_random_node(graph):
    new_graph = None

    while new_graph is None or not nx.is_connected(new_graph):
        delete_node = np.random.choice(graph.nodes)
        new_graph = graph.copy()
        new_graph.remove_node(delete_node)

    return new_graph


def remove_random_nodes(graph, num_nodes):
    while graph.number_of_nodes() > num_nodes:
        graph = remove_random_node(graph)

    return graph


def remove_random_edge(graph):
    new_graph = None

    while new_graph is None or not nx.is_connected(new_graph):
        delete_edge = choice(list(graph.edges))
        new_graph = graph.copy()
        new_graph.remove_edge(*delete_edge)

    return new_graph


def add_random_edges(current_graph, NE, min_edges=61, max_edges=122):
    """
    randomly adds edges between nodes with no existing edges.
    based on: https://stackoverflow.com/questions/42591549/add-and-delete-a-random-edge-in-networkx
    :param probability_of_new_connection:
    :return: None
    """
    if current_graph:
        connected = []
        for i in current_graph.nodes:
            # find the other nodes this one is connected to
            connected = connected + [to for (fr, to) in current_graph.edges(i)]
            connected = list(dict.fromkeys(connected))
            # and find the remainder of nodes, which are candidates for new edges

        unconnected = [j for j in current_graph.nodes if j not in connected]
        # print('Connected:', connected)
        # print('Unconnected', unconnected)
        is_connected = nx.is_connected(current_graph)
        while not is_connected:  # randomly add edges until the graph is connected
            if len(unconnected) == 0:
                break
            new = choice(unconnected)
            if not connected:
                old = choice(unconnected)
                while old == new:
                    old = choice(unconnected)
            else:
                old = choice(connected)
            edge_label = np.random.randint(1, NE + 1)

            # for visualise only
            current_graph.add_edges_from([(old, new, {"label": edge_label})])
            current_graph.nodes[old]["modified"] = True
            # book-keeping, in case both add and remove done in same cycle
            if not connected:
                unconnected.remove(old)
                connected.append(old)

            unconnected.remove(new)
            connected.append(new)

            is_connected = nx.is_connected(current_graph)
            # print('Connected:', connected)
            # print('Unconnected', unconnected

        num_edges = np.random.randint(min_edges, max_edges + 1)

        while current_graph.number_of_edges() < num_edges:
            old_1, old_2 = np.random.choice(current_graph.nodes, 2, replace=False)
            while current_graph.has_edge(old_1, old_2):
                old_1, old_2 = np.random.choice(current_graph.nodes, 2, replace=False)
            edge_label = np.random.randint(1, NE + 1)
            current_graph.add_edges_from([(old_1, old_2, {"label": edge_label})])
            current_graph.nodes[old_1]["modified"] = True
            current_graph.nodes[old_2]["modified"] = True

    return current_graph


def add_random_nodes(
    graph,
    num_nodes,
    id_node_start,
    number_label_node,
    number_label_edge,
    min_edges,
    max_edges,
):
    graph_nodes = graph.number_of_nodes()
    number_of_possible_nodes_to_add = num_nodes - graph_nodes

    # start node_id from the number of nodes already in the common graph (note that the node ids are numbered from 0)
    node_id = id_node_start
    # so if there were 5 nodes in the common graph (0,1,2,3,4) start adding new nodes from node 5 on wards
    added_nodes = []
    for i in range(number_of_possible_nodes_to_add):
        node_label = np.random.randint(1, number_label_node + 1)
        added_nodes.append((node_id, {"label": node_label, "modified": True}))
        node_id += 1

    # add all nodes to current graph
    graph.add_nodes_from(added_nodes)
    graph = add_random_edges(graph, number_label_edge, min_edges, max_edges)
    return graph, node_id


def random_modify(graph, NN, NE, node_start_id, min_edges, max_edges):
    num_steps = np.random.randint(1, graph.number_of_nodes() + graph.number_of_edges())
    modify_type = None

    while num_steps > 0:
        modify_type = np.random.randint(0, 3)

        if modify_type == 0:  # Change node label
            chose_node = np.random.choice(graph.nodes)
            origin_label = graph.nodes[chose_node]["label"]
            new_label = np.random.randint(1, NN + 1)
            while new_label == origin_label:
                new_label = np.random.randint(1, NN + 1)

            graph.nodes[chose_node]["label"] = new_label
            graph.nodes[chose_node]["modified"] = True

        # elif modify_type == 1:
        #     chose_edge = np.random.choice(graph.nodes, size=2, replace=False)
        #     while not graph.has_edge(*chose_edge):
        #         chose_edge = np.random.choice(graph.nodes, size=2, replace=False)

        #     origin_label = graph[chose_edge[0]][chose_edge[1]]["label"]
        #     new_label = np.random.randint(1, NE+1)
        #     while new_label == origin_label:
        #         new_label = np.random.randint(1, NE+1)

        #     graph[chose_edge[0]][chose_edge[1]]["label"] = new_label
        #     graph.nodes[chose_edge[0]]["modified"] = True
        #     graph.nodes[chose_edge[1]]["modified"] = True

        elif modify_type == 1:  # Remove & add random node
            graph, node_start_id = add_random_nodes(
                graph,
                graph.number_of_nodes() + 1,
                node_start_id,
                NN,
                NE,
                min_edges,
                max_edges,
            )
            graph = remove_random_nodes(graph, graph.number_of_nodes() - 1)

        elif modify_type == 2:  # Remove & add random edge
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()

            if n_nodes * (n_nodes - 1) / 2 > n_edges:
                graph = add_random_edges(graph, NE, n_edges + 1, n_edges + 1)
            if graph.number_of_edges() >= n_nodes:
                graph = remove_random_edge(graph)

        num_steps -= 1

    return graph, node_start_id


def generate_noniso_subgraph(
    graph,
    no_of_nodes,
    avg_degree,
    std_degree,
    number_label_node,
    number_label_edge,
    *args,
    **kwargs
):
    graph_nodes = graph.number_of_nodes()
    node_ratio = no_of_nodes / graph_nodes
    if node_ratio > 1:
        node_ratio = 1

    min_edges = int(no_of_nodes * min(no_of_nodes - 1, avg_degree - std_degree) / 2)
    max_edges = int(no_of_nodes * min(no_of_nodes - 1, avg_degree + std_degree) / 2)
    subgraph = None
    iteration = 0

    while (
        subgraph is None
        or subgraph.number_of_nodes() < 2
        or not nx.is_connected(subgraph)
    ):
        chose_nodes = np.random.choice(
            [0, 1], size=graph_nodes, replace=True, p=[1 - node_ratio, node_ratio]
        )
        remove_nodes = np.where(chose_nodes == 0)[0]
        subgraph = graph.copy()
        subgraph.remove_nodes_from(remove_nodes)

        iteration += 1
        if iteration > 5:
            node_ratio *= 1.05
            if node_ratio > 1:
                node_ratio = 1
            iteration = 0

    for nid in subgraph.nodes:
        subgraph.nodes[nid]["modified"] = False

    if subgraph.number_of_nodes() > no_of_nodes:
        subgraph = remove_random_nodes(subgraph, no_of_nodes)
    elif subgraph.number_of_nodes() < no_of_nodes:
        subgraph, graph_nodes = add_random_nodes(
            subgraph,
            no_of_nodes,
            graph_nodes,
            number_label_node,
            number_label_edge,
            min_edges,
            max_edges,
        )

    # Remove for induced subgraph
    # high = subgraph.number_of_edges() - subgraph.number_of_nodes() + 2
    # if high > 0:
    #     modify_times = np.random.randint(0, high)
    #     for _ in range(modify_times):
    #         subgraph = remove_random_edge(subgraph)

    subgraph, graph_nodes = random_modify(
        subgraph,
        number_label_node,
        number_label_edge,
        graph_nodes,
        min_edges,
        max_edges,
    )
    graph_matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph, subgraph, node_match=node_match, edge_match=edge_match
    )

    retry = 0
    while True:
        try:
            with time_limit(10):
                subgraph_is_isomorphic = graph_matcher.subgraph_is_isomorphic()
        except:
            subgraph_is_isomorphic = False

        if subgraph_is_isomorphic:
            subgraph, graph_nodes = random_modify(
                subgraph,
                number_label_node,
                number_label_edge,
                graph_nodes,
                min_edges,
                max_edges,
            )
            graph_matcher = nx.algorithms.isomorphism.GraphMatcher(
                graph, subgraph, node_match=node_match, edge_match=edge_match
            )
            retry += 1
            if retry > 2:
                return None
        else:
            break

    return subgraph


def generate_subgraphs(graph, number_subgraph_per_source, *args, **kwargs):
    list_iso_subgraphs = []
    list_noniso_subgraphs = []

    for _ in tqdm(range(number_subgraph_per_source)):
        generated_subgraph = None
        while generated_subgraph is None:
            no_of_nodes = np.random.randint(2, graph.number_of_nodes() + 1)
            prob = np.random.randint(0, 2)
            if prob == 1:
                generated_subgraph = generate_iso_subgraph(
                    graph, no_of_nodes, *args, **kwargs
                )
            else:
                generated_subgraph = generate_noniso_subgraph(
                    graph, no_of_nodes, *args, **kwargs
                )

        if prob == 1:
            list_iso_subgraphs.append(generated_subgraph)
        else:
            list_noniso_subgraphs.append(generated_subgraph)

    return list_iso_subgraphs, list_noniso_subgraphs


def generate_one_sample(
    number_subgraph_per_source,
    avg_source_size,
    std_source_size,
    avg_degree,
    std_degree,
    number_label_node,
    number_label_edge,
):
    generated_pattern = None
    iteration = 0
    no_of_nodes = int(np.random.normal(avg_source_size, std_source_size))
    while no_of_nodes < 2:
        no_of_nodes = int(np.random.normal(avg_source_size, std_source_size))
    degree = np.random.normal(avg_degree, std_degree)
    if degree < 1:
        degree = 1
    if degree > no_of_nodes - 1:
        degree = no_of_nodes - 1
    probability_for_edge_creation = degree / (no_of_nodes - 1)

    while (
        generated_pattern is None
        or nx.is_empty(generated_pattern)
        or not nx.is_connected(generated_pattern)
    ):  # make sure the generated graph is connected
        generated_pattern = nx.erdos_renyi_graph(
            no_of_nodes, probability_for_edge_creation, directed=False
        )
        iteration += 1
        if iteration > 5:
            probability_for_edge_creation *= 1.05
            iteration = 0

    labelled_pattern = add_labels(
        generated_pattern, number_label_node, number_label_edge
    )

    iso_subgraphs, noniso_subgraphs = generate_subgraphs(
        labelled_pattern,
        number_subgraph_per_source,
        avg_degree,
        std_degree,
        number_label_node,
        number_label_edge,
    )
    return labelled_pattern, iso_subgraphs, noniso_subgraphs


def generate_batch(start_idx, stop_idx, number_source, dataset_path, *args, **kwargs):
    for idx in range(start_idx, stop_idx):
        print("SAMPLE %d/%d" % (idx + 1, number_source))
        graph, iso_subgraphs, noniso_subgraphs = generate_one_sample(*args, **kwargs)
        save_per_source(idx, graph, iso_subgraphs, noniso_subgraphs, dataset_path)


def generate_dataset(dataset_path, is_continue, number_source, *args, **kwargs):
    print("Generating...")
    list_processes = []

    if is_continue is not False:
        print("Continue generating...")
        generated_sample = os.listdir(dataset_path)
        generated_sample = [int(x) for x in generated_sample]
        remaining_sample = np.array(
            sorted(set(range(number_source)) - set(generated_sample))
        )
        gap_list = remaining_sample[1:] - remaining_sample[:-1]
        gap_idx = np.where(gap_list > 1)[0] + 1
        if len(gap_idx) < 1:
            list_idx = [(remaining_sample[0], remaining_sample[-1] + 1)]
        else:
            list_idx = (
                [(remaining_sample[0], remaining_sample[gap_idx[0]])]
                + [
                    (remaining_sample[gap_idx[i]], remaining_sample[gap_idx[i + 1]])
                    for i in range(gap_idx.shape[0] - 1)
                ]
                + [(remaining_sample[gap_idx[-1]], remaining_sample[-1] + 1)]
            )

        for start_idx, stop_idx in list_idx:
            list_processes.append(
                Process(
                    target=generate_batch,
                    args=(start_idx, stop_idx, number_source, dataset_path),
                    kwargs=kwargs,
                )
            )

    else:
        batch_size = int(number_source / os.cpu_count()) + 1
        start_idx = 0
        stop_idx = start_idx + batch_size

        for idx in range(os.cpu_count()):
            list_processes.append(
                Process(
                    target=generate_batch,
                    args=(start_idx, stop_idx, number_source, dataset_path),
                    kwargs=kwargs,
                )
            )

            start_idx = stop_idx
            stop_idx += batch_size
            if stop_idx > number_source:
                stop_idx = number_source

    for idx in range(len(list_processes)):
        list_processes[idx].start()

    for idx in range(len(list_processes)):
        list_processes[idx].join()


def save_per_source(graph_id, H, iso_subgraphs, noniso_subgraphs, dataset_path):
    # Ensure path
    subgraph_path = os.path.join(dataset_path, str(graph_id))
    ensure_path(subgraph_path)

    # Save source graphs
    source_graph_file = os.path.join(subgraph_path, "source.lg")
    with open(source_graph_file, "w", encoding="utf-8") as file:
        file.write("t # {0}\n".format(graph_id))
        for node in H.nodes:
            file.write("v {} {}\n".format(node, H.nodes[node]["label"]))
        for edge in H.edges:
            file.write(
                "e {} {} {}\n".format(
                    edge[0], edge[1], H.edges[(edge[0], edge[1])]["label"]
                )
            )

    # Save subgraphs
    iso_subgraph_file = os.path.join(subgraph_path, "iso_subgraphs.lg")
    noniso_subgraph_file = os.path.join(subgraph_path, "noniso_subgraphs.lg")
    iso_subgraph_mapping_file = os.path.join(subgraph_path, "iso_subgraphs_mapping.lg")
    noniso_subgraph_mapping_file = os.path.join(
        subgraph_path, "noniso_subgraphs_mapping.lg"
    )

    isf = open(iso_subgraph_file, "w", encoding="utf-8")
    ismf = open(iso_subgraph_mapping_file, "w", encoding="utf-8")

    for subgraph_id, S in enumerate(iso_subgraphs):
        isf.write("t # {0}\n".format(subgraph_id))
        ismf.write("t # {0}\n".format(subgraph_id))
        node_mapping = {}
        list_nodes = list(S.nodes)
        shuffle(list_nodes)

        for node_idx, node_emb in enumerate(list_nodes):
            isf.write("v {} {}\n".format(node_idx, S.nodes[node_emb]["label"]))
            ismf.write("v {} {}\n".format(node_idx, node_emb))
            node_mapping[node_emb] = node_idx

        for edge in S.edges:
            edge_0 = node_mapping[edge[0]]
            edge_1 = node_mapping[edge[1]]
            isf.write(
                "e {} {} {}\n".format(
                    edge_0, edge_1, S.edges[(edge[0], edge[1])]["label"]
                )
            )

    isf.close()
    ismf.close()

    nisf = open(noniso_subgraph_file, "w", encoding="utf-8")
    nismf = open(noniso_subgraph_mapping_file, "w", encoding="utf-8")
    for subgraph_id, S in enumerate(noniso_subgraphs):
        nisf.write("t # {0}\n".format(subgraph_id))
        nismf.write("t # {0}\n".format(subgraph_id))
        node_mapping = {}
        list_nodes = list(S.nodes)
        shuffle(list_nodes)

        for node_idx, node_emb in enumerate(list_nodes):
            nisf.write("v {} {}\n".format(node_idx, S.nodes[node_emb]["label"]))
            if not S.nodes[node_emb]["modified"]:
                nismf.write("v {} {}\n".format(node_idx, node_emb))
            node_mapping[node_emb] = node_idx

        for edge in S.edges:
            edge_0 = node_mapping[edge[0]]
            edge_1 = node_mapping[edge[1]]
            nisf.write(
                "e {} {} {}\n".format(
                    edge_0, edge_1, S.edges[(edge[0], edge[1])]["label"]
                )
            )

    nisf.close()
    nismf.close()


def main(config_file, is_continue):
    seed(42)
    np.random.seed(42)
    dataset_path = os.path.join(
        "datasets", os.path.basename(config_file).split(".")[0] + "_train"
    )
    ensure_path(dataset_path)
    config = read_config(config_file)

    generate_dataset(dataset_path=dataset_path, is_continue=is_continue, **config)


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.cont)
