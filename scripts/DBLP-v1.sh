python train.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 39

python train.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --directed \
    --embedding_dim 39

# End2end evaluation
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 39 \
    --ckpt save/DBLP-v1_jump/best_model.pt

python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 39 \
    --directed \
    --ckpt save/DBLP-v1_jump_directed/best_model.pt

# Scalability evaluation
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 39 \
    --ckpt save/DBLP-v1_jump/best_model.pt \
    --test_keys test_keys_dense_0_20.pkl

python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 39 \
    --ckpt save/DBLP-v1_jump/best_model.pt \
    --test_keys test_keys_dense_20_40.pkl

python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 39 \
    --ckpt save/DBLP-v1_jump/best_model.pt \
    --test_keys test_keys_nondense_0_20.pkl

python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 39 \
    --ckpt save/DBLP-v1_jump/best_model.pt \
    --test_keys test_keys_nondense_20_40.pkl

# Explainability evaluation
python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 39 \
    --ckpt save/DBLP-v1_jump/best_model.pt

python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 39 \
    --directed \
    --ckpt save/DBLP-v1_jump/best_model.pt

# Abalation study
python train.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 39 \
    --branch left
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 39 \
    --branch left \
    --ckpt save/DBLP-v1_jump_left/best_model.pt
python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 39 \
    --branch left \
    --ckpt save/DBLP-v1_jump_left/best_model.pt

python train.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 39 \
    --branch left
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 39 \
    --branch left \
    --ckpt save/DBLP-v1_cont_left/best_model.pt
python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic cont \
    --embedding_dim 39 \
    --branch left \
    --ckpt save/DBLP-v1_cont_left/best_model.pt

python train.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 39 \
    --branch left
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 39 \
    --branch left \
    --ckpt save/DBLP-v1_static1_left/best_model.pt
python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic static \
    --embedding_dim 39  \
    --branch left \
    --ckpt save/DBLP-v1_static1_left/best_model.pt

python train.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 39 \
    --branch right
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 39 \
    --branch right \
    --ckpt save/DBLP-v1_jump_right/best_model.pt
python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 39 \
    --branch right \
    --ckpt save/DBLP-v1_jump_right/best_model.pt

python train.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 39 \
    --branch right
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 39 \
    --branch right \
    --ckpt save/DBLP-v1_cont_right/best_model.pt
python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic cont \
    --embedding_dim 39  \
    --branch right \
    --ckpt save/DBLP-v1_cont_right/best_model.pt

python train.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 39 \
    --branch right
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 39 \
    --branch right \
    --ckpt save/DBLP-v1_static1_right/best_model.pt
python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic static \
    --embedding_dim 39  \
    --branch right \
    --ckpt save/DBLP-v1_static1_right/best_model.pt

python train.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 39
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 39 \
    --ckpt save/DBLP-v1_cont/best_model.pt
python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic cont \
    --embedding_dim 39 \
    --ckpt save/DBLP-v1_cont/best_model.pt

python train.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 39
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 39 \
    --ckpt save/DBLP-v1_static1/best_model.pt
python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic static \
    --embedding_dim 39 \
    --ckpt save/DBLP-v1_static1/best_model.pt

python train.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 39 \
    --nhead 2
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 39 \
    --nhead 2 \
    --ckpt save/DBLP-v1_jump_nhead2/best_model.pt
python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 39 \
    --nhead 2 \
    --ckpt save/DBLP-v1_jump_nhead2/best_model.pt

python train.py \
    --dataset DBLP-v1 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 39 \
    --nhead 4
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 39 \
    --nhead 4 \
    --ckpt save/DBLP-v1_jump_nhead4/best_model.pt
python evaluate_matching.py \
    --dataset DBLP-v1 \
    --batch_size 32 \
    --tatic jump \
    --embedding_dim 39 \
    --nhead 4 \
    --ckpt save/DBLP-v1_jump_nhead4/best_model.pt

# Generalization
python train.py \
    --dataset DBLP-v1 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag cross

python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag DBLP-v1_cross \
    --ckpt save/DBLP-v1_jump_cross/best_model.pt
python evaluate.py \
    --dataset DHFR \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag DBLP-v1_cross \
    --ckpt save/DBLP-v1_jump_cross/best_model.pt
python evaluate.py \
    --dataset COX2_MD \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag DBLP-v1_cross \
    --ckpt save/DBLP-v1_jump_cross/best_model.pt
python evaluate.py \
    --dataset COX2 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag DBLP-v1_cross \
    --ckpt save/DBLP-v1_jump_cross/best_model.pt
python evaluate.py \
    --dataset MSRC-21 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag DBLP-v1_cross \
    --ckpt save/DBLP-v1_jump_cross/best_model.pt