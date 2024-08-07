python train.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 9

python train.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 9 \
    --directed

# End2end evaluation
python evaluate.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 9 \
    --ckpt save/DHFR_jump/best_model.pt

# Directed evaluation
python evaluate.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 9 \
    --directed \
    --ckpt save/DHFR_jump_directed/best_model.pt

# Scalability evaluation
python evaluate.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 9 \
    --ckpt save/DHFR_jump/best_model.pt \
    --test_keys test_keys_nondense_0_20.pkl

python evaluate.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 9 \
    --ckpt save/DHFR_jump/best_model.pt \
    --test_keys test_keys_nondense_20_40.pkl

python evaluate.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 9 \
    --ckpt save/DHFR_jump/best_model.pt \
    --test_keys test_keys_nondense_40_60.pkl

python evaluate.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 9 \
    --ckpt save/DHFR_jump/best_model.pt \
    --test_keys test_keys_nondense_60_.pkl

# Explainability evaluation
python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 9 \
    --ckpt save/DHFR_jump/best_model.pt

python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 9 \
    --directed \
    --ckpt save/DHFR_jump_directed/best_model.pt

# Abalation study
python train.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 9 \
    --branch left
python evaluate.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 9 \
    --branch left \
    --ckpt save/DHFR_jump_left/best_model.pt
python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 9 \
    --branch left \
    --ckpt save/DHFR_jump_left/best_model.pt

python train.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 9 \
    --branch left
python evaluate.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 9 \
    --branch left \
    --ckpt save/DHFR_cont_left/best_model.pt
python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic cont \
    --embedding_dim 9 \
    --branch left \
    --ckpt save/DHFR_cont_left/best_model.pt

python train.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 9 \
    --branch left
python evaluate.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 9 \
    --branch left \
    --ckpt save/DHFR_static1_left/best_model.pt
python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic static \
    --embedding_dim 9  \
    --branch left \
    --ckpt save/DHFR_static1_left/best_model.pt

python train.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 9 \
    --branch right
python evaluate.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 9 \
    --branch right \
    --ckpt save/DHFR_jump_right/best_model.pt
python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 9 \
    --branch right \
    --ckpt save/DHFR_jump_right/best_model.pt

python train.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 9 \
    --branch right
python evaluate.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 9 \
    --branch right \
    --ckpt save/DHFR_cont_right/best_model.pt
python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic cont \
    --embedding_dim 9  \
    --branch right \
    --ckpt save/DHFR_cont_right/best_model.pt

python train.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 9 \
    --branch right
python evaluate.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 9 \
    --branch right \
    --ckpt save/DHFR_static1_right/best_model.pt
python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic static \
    --embedding_dim 9  \
    --branch right \
    --ckpt save/DHFR_static1_right/best_model.pt

python train.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 9
python evaluate.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 9 \
    --ckpt save/DHFR_cont/best_model.pt
python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic cont \
    --embedding_dim 9 \
    --ckpt save/DHFR_cont/best_model.pt

python train.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 9
python evaluate.py \
    --dataset DHFR \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 9 \
    --ckpt save/DHFR_static1/best_model.pt
python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic static \
    --embedding_dim 9 \
    --ckpt save/DHFR_static1/best_model.pt

python train.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 9 \
    --nhead 2
python evaluate.py \
    --dataset DHFR \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 9 \
    --nhead 2 \
    --ckpt save/DHFR_jump_nhead2/best_model.pt
python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 9 \
    --nhead 2 \
    --ckpt save/DHFR_jump_nhead2/best_model.pt

python train.py \
    --dataset DHFR \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 9 \
    --nhead 4
python evaluate.py \
    --dataset DHFR \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 9 \
    --nhead 4 \
    --ckpt save/DHFR_jump_nhead4/best_model.pt
python evaluate_matching.py \
    --dataset DHFR \
    --batch_size 32 \
    --tatic jump \
    --embedding_dim 9 \
    --nhead 4 \
    --ckpt save/DHFR_jump_nhead4/best_model.pt

# Generalization
python train.py \
    --dataset DHFR \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag cross

python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag DHFR_cross \
    --ckpt save/DHFR_jump_cross/best_model.pt
python evaluate.py \
    --dataset COX2 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag DHFR_cross \
    --ckpt save/DHFR_jump_cross/best_model.pt
python evaluate.py \
    --dataset COX2_MD \
    --batch_size 32 \
    --tatic jump \
    --embedding_dim 190 \
    --tag DHFR_cross \
    --ckpt save/DHFR_jump_cross/best_model.pt
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag DHFR_cross \
    --ckpt save/DHFR_jump_cross/best_model.pt
python evaluate.py \
    --dataset MSRC-21 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag DHFR_cross \
    --ckpt save/DHFR_jump_cross/best_model.pt