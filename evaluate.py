import os
import pickle
import time

import numpy as np
import torch
import utils
from dataset import BaseDataset, collate_fn
from model import GLeMaNet
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def main(args):
    # hyper parameters
    data_path = os.path.join(args.data_path, args.dataset)
    if args.directed:
        data_path += "_directed"
    result_dir = utils.ensure_dir(args.result_dir, args)
    result_file = f"result{args.test_keys[9:-4]}.csv"
    args.train_keys = os.path.join(data_path, args.train_keys)
    args.test_keys = os.path.join(data_path, args.test_keys)

    with open(args.test_keys, "rb") as fp:
        test_keys = pickle.load(fp)

    print(f"Number of test data: {len(test_keys)}")

    # Initialize model
    model = GLeMaNet(args)
    print(
        "Number of parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.initialize_model(model, device, load_save_file=args.ckpt)

    test_dataset = BaseDataset(test_keys, data_path, embedding_dim=args.embedding_dim)
    test_dataloader = DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Starting evaluation
    test_true = []
    test_pred = []

    model.eval()
    st_eval = time.time()

    for sample in tqdm(test_dataloader):
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
        with torch.no_grad():
            pred = model((H, A1, A2, V))

        # Collect true label and predicted label
        test_true.append(Y.data.cpu().numpy())
        test_pred.append(pred.data.cpu().numpy())

    end = time.time()

    test_pred = np.concatenate(test_pred, 0)
    test_true = np.concatenate(test_true, 0)
    result_rows = []

    for conf_step in [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.91,
        0.92,
        0.93,
        0.94,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
    ]:
        test_pred_by_conf = test_pred.copy()
        test_pred_by_conf[test_pred_by_conf < conf_step] = 0
        test_pred_by_conf[test_pred_by_conf > 0] = 1

        test_roc = roc_auc_score(test_true, test_pred_by_conf)
        test_acc = accuracy_score(test_true, test_pred_by_conf)
        test_pre = precision_score(test_true, test_pred_by_conf)
        test_rec = recall_score(test_true, test_pred_by_conf)
        test_f1s = f1_score(test_true, test_pred_by_conf)
        test_prc = average_precision_score(test_true, test_pred_by_conf)
        test_time = (end - st_eval) / len(test_dataset)

        result_rows.append(
            [
                conf_step,
                test_time,
                test_roc,
                test_prc,
                test_pre,
                test_rec,
                test_f1s,
                test_acc,
            ]
        )

    with open(os.path.join(result_dir, result_file), "w", encoding="utf-8") as f:
        f.write(
            "Confident,Execution Time,ROC AUC,PR AUC,Precision,Recall,F1-Score,Accuracy\n"
        )
        for row in result_rows:
            f.write(",".join([str(x) for x in row]))
            f.write("\n")


if __name__ == "__main__":
    args = utils.parse_args()
    print(args)

    main(args)
