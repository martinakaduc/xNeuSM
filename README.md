# xNeuSM: Explainable Neural Subgraph Matching with Graph Learnable Multi-hop Attention Networks

This repository contains the source code and datasets for our paper:
"xNeuSM: Explainable Neural Subgraph Matching with Graph Learnable Multi-hop Attention Networks".

# Requirements

We use conda to manage the environment. Please install the dependencies by running:

```
conda env create -f environment.yml
```

# Prepare the datasets

We experiment on real-world datasets. Before running the experiments, please follow the below steps to prepare the datasets.

## Real-world datasets

We have prepared the real-world datasets in the [data_real](data_real/datasets) folder. To generate the datasets, please run:

```
cd data_real
python make_datasets.py --ds [DATASET_NAME]
python generate_data_v1.py --config configs/[DATASET_NAME].json
cd ..
python process_data.py [DATASET_NAME] real
```

# Run the experiments

Here is the command to run the experiments:

```
python train.py [--seed SEED] 
                [--lr LR] 
                [--epoch EPOCH] 
                [--ngpu NGPU] 
                [--dataset DATASET] 
                [--batch_size BATCH_SIZE] 
                [--num_workers NUM_WORKERS] 
                [--embedding_dim EMBEDDING_DIM] 
                [--tatic {static,cont,jump}]
                [--nhop NHOP] 
                [--n_graph_layer N_GRAPH_LAYER] 
                [--d_graph_layer D_GRAPH_LAYER] 
                [--n_FC_layer N_FC_LAYER] 
                [--d_FC_layer D_FC_LAYER] 
                [--data_path DATA_PATH] 
                [--save_dir SAVE_DIR]
                [--log_dir LOG_DIR] 
                [--dropout_rate DROPOUT_RATE] 
                [--al_scale AL_SCALE] 
                [--ckpt CKPT] 
                [--train_keys TRAIN_KEYS] 
                [--test_keys TEST_KEYS]

optional arguments:
  --seed SEED           random seed
  --lr LR               learning rate
  --epoch EPOCH         epoch
  --ngpu NGPU           number of gpu
  --dataset DATASET     dataset
  --batch_size BATCH_SIZE
                        batch_size
  --num_workers NUM_WORKERS
                        number of workers
  --embedding_dim EMBEDDING_DIM
                        node embedding dim aka number of distinct node label
  --tatic {static,cont,jump}
                        tactic of defining number of hops
  --nhop NHOP           number of hops
  --n_graph_layer N_GRAPH_LAYER
                        number of GNN layer
  --d_graph_layer D_GRAPH_LAYER
                        dimension of GNN layer
  --n_FC_layer N_FC_LAYER
                        number of FC layer
  --d_FC_layer D_FC_LAYER
                        dimension of FC layer
  --data_path DATA_PATH
                        path to the data
  --save_dir SAVE_DIR   save directory of model parameter
  --log_dir LOG_DIR     logging directory
  --dropout_rate DROPOUT_RATE
                        dropout_rate
  --al_scale AL_SCALE   attn_loss scale
  --ckpt CKPT           Load ckpt file
  --train_keys TRAIN_KEYS
                        train keys
  --test_keys TEST_KEYS
                        test keys
```

Additionally, we have prepared the scripts to run the experiments using real datasets in the [scripts](scripts) folder. To run the experiments, please execute the following command:

```
bash scripts/[DATASET_NAME].sh
```

# Citation

If you find this repository useful in your research, please cite our paper:

```
@article{Nguyen2024Explainable,
  author={Nguyen, Duc Q. and Toan Nguyen, Thanh and Jo, Jun and Poux, Florent and Anirban, Shikha and Quan, Tho T.},
  journal={IEEE Access}, 
  title={{Explainable Neural Subgraph Matching With Learnable Multi-Hop Attention}}, 
  year={2024},
  volume={12},
  number={},
  pages={130474-130492},
  keywords={Spread spectrum communication;Pattern matching;Multitasking;Adaptation models;Graph neural networks;Object recognition;Explainability;graph neural networks;learnable multi-hop attention;subgraph matching},
  doi={10.1109/ACCESS.2024.3458050}
}
```
