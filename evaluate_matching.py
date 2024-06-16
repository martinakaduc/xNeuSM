import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import utils
from dataset import BaseDataset, collate_fn
from model import GLeMaNet
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval_mapping(groundtruth, predict_list, predict_prob):
    acc = []
    MRR = []

    for sgn in groundtruth:
        # Calculate precision
        list_acc = []
        for i in range(1, 11):
            if groundtruth[sgn] in predict_list[sgn][:i]:
                list_acc.append(1)
            else:
                list_acc.append(0)

        acc.append(list_acc)

        if groundtruth[sgn] in predict_list[sgn]:
            MRR.append(1 / (predict_list[sgn].index(groundtruth[sgn]) + 1))
        else:
            MRR.append(0)

    acc = np.mean(np.array(acc), axis=0)
    MRR = np.mean(np.array(MRR))
    return np.concatenate([acc, np.array([MRR])])


def evaluate(args):
    data_path = os.path.join(args.data_path, args.dataset)
    args.train_keys = os.path.join(data_path, args.train_keys)
    args.test_keys = os.path.join(data_path, args.test_keys)
    result_dir = utils.ensure_dir(args.result_dir, args)
    result_file = f"result_matching_{args.test_keys[9:-4]}.csv"

    with open(args.test_keys, "rb") as fp:
        test_keys = pickle.load(fp)
        # Only use isomorphism subgraphs for mapping testing
        test_keys = list(filter(lambda x: x.endswith("iso_test"), test_keys))

    print(f"Number of test data: {len(test_keys)}")

    model = GLeMaNet(args)
    print(
        "Number of parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.initialize_model(model, device, load_save_file=args.ckpt)

    test_dataset = BaseDataset(
        test_keys, args.data_path, embedding_dim=args.embedding_dim
    )
    test_dataloader = DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Starting evaluation
    test_true_mapping = []
    test_pred_mapping = []
    list_results = []

    model.eval()
    st_eval = time.time()

    for sample in tqdm(test_dataloader):
        model.zero_grad()
        H, A1, A2, M, S, Y, V, _ = sample
        H, A1, A2, M, S, Y, V = (
            H.to(device),
            A1.to(device),
            A2.to(device),
            M.to(device),
            S.to(device),
            Y.to(device),
            V.to(device),
        )

        # Test neural network
        pred = model.get_refined_adjs2((H, A1, A2, V))

        # Collect true label and predicted label
        test_true_mapping = M.data.cpu().numpy()
        test_pred_mapping = pred.data.cpu().numpy()

        for mapping_true, mapping_pred in zip(test_true_mapping, test_pred_mapping):
            gt_mapping = {}
            x_coord, y_coord = np.where(mapping_true > 0)
            for x, y in zip(x_coord, y_coord):
                if x < y:
                    gt_mapping[x] = [y]  # Subgraph node: Graph node

            pred_mapping = defaultdict(lambda: {})
            x_coord, y_coord = np.where(mapping_pred > 0)

            # TODO pred_mapping shoud be sorted by probability

            for x, y in zip(x_coord, y_coord):
                if x < y:
                    if y in pred_mapping[x]:
                        pred_mapping[x][y] = (
                            pred_mapping[x][y] + mapping_pred[x][y]
                        ) / 2
                    else:
                        pred_mapping[x][y] = mapping_pred[
                            x, y
                        ]  # Subgraph node: Graph node
                else:
                    if x in pred_mapping[y]:
                        pred_mapping[y][x] = (
                            pred_mapping[y][x] + mapping_pred[x][y]
                        ) / 2
                    else:
                        pred_mapping[y][x] = mapping_pred[
                            x, y
                        ]  # Subgraph node: Graph node

            sorted_predict_mapping = defaultdict(lambda: [])
            sorted_predict_mapping.update(
                {
                    k: [
                        y[0]
                        for y in sorted(
                            [(n, prob) for n, prob in v.items()],
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    ]
                    for k, v in pred_mapping.items()
                }
            )

            results = eval_mapping(
                gt_mapping, sorted_predict_mapping, pred_mapping)
            list_results.append(results)

    end = time.time()

    list_results = np.array(list_results)
    avg_results = np.mean(list_results, axis=0)
    print("Test time: ", end - st_eval)
    print("Top1-Top10 Accuracy, MRR")
    print(avg_results)

    with open(
        os.path.join(result_dir, result_file),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Time,Top1-Acc,Top2-Acc,Top3-Acc,Top4-Acc,Top5-Acc,Top6-Acc,Top7-Acc,Top8-Acc,Top9-Acc,Top10-Acc,MRR\n"
        )
        f.write("%f," % (end - st_eval))
        f.write(",".join([str(x) for x in avg_results]))
        f.write("\n")


if __name__ == "__main__":
    args = utils.parse_args()
    print(args)

    evaluate(args)
