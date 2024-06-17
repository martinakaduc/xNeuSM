python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90

python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --directed

# End2end evaluation
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt

python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --directed \
    --ckpt save/KKI_jump_directed/best_model.pt

# Scalability evaluation
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt \
    --test_keys test_keys_dense_0_20.pkl

python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt \
    --test_keys test_keys_dense_20_40.pkl

python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt \
    --test_keys test_keys_dense_40_60.pkl

python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt \
    --test_keys test_keys_dense_60_.pkl

python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt \
    --test_keys test_keys_nondense_0_20.pkl

python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt \
    --test_keys test_keys_nondense_20_40.pkl

python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt \
    --test_keys test_keys_nondense_40_60.pkl

python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt \
    --test_keys test_keys_nondense_60_.pkl

# Explainability evaluation
python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt

python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --directed \
    --ckpt save/KKI_jump_directed/best_model.pt

python viz_matching.py  \
    --dataset KKI_test \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt \
    --data_path data_real/datasets \
    --source 49 \
    --query 0 \
    --iso \
    --mapping_threshold 0.5

python viz_matching.py  \
    --dataset KKI_test \
    --tatic jump \
    --embedding_dim 90 \
    --ckpt save/KKI_jump/best_model.pt \
    --data_path data_real/datasets \
    --source 49 \
    --query 4 \
    --mapping_threshold 0.5

# Abalation study
python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --branch left
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --branch left \
    --ckpt save/KKI_jump_left/best_model.pt
python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --branch left \
    --ckpt save/KKI_jump_left/best_model.pt

python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic cont \
    --embedding_dim 90 \
    --branch left
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic cont \
    --embedding_dim 90 \
    --branch left \
    --ckpt save/KKI_cont_left/best_model.pt
python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic cont \
    --embedding_dim 90 \
    --branch left \
    --ckpt save/KKI_cont_left/best_model.pt

python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic static \
    --embedding_dim 90 \
    --branch left
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic static \
    --embedding_dim 90 \
    --branch left \
    --ckpt save/KKI_static1_left/best_model.pt
python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic static \
    --embedding_dim 90  \
    --branch left \
    --ckpt save/KKI_static1_left/best_model.pt

python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --branch right
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --branch right \
    --ckpt save/KKI_jump_right/best_model.pt
python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --branch right \
    --ckpt save/KKI_jump_right/best_model.pt

python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic cont \
    --embedding_dim 90 \
    --branch right
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic cont \
    --embedding_dim 90 \
    --branch right \
    --ckpt save/KKI_cont_right/best_model.pt
python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic cont \
    --embedding_dim 90  \
    --branch right \
    --ckpt save/KKI_cont_right/best_model.pt

python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic static \
    --embedding_dim 90 \
    --branch right
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic static \
    --embedding_dim 90 \
    --branch right \
    --ckpt save/KKI_static1_right/best_model.pt
python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic static \
    --embedding_dim 90  \
    --branch right \
    --ckpt save/KKI_static1_right/best_model.pt

python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic cont \
    --embedding_dim 90
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic cont \
    --embedding_dim 90 \
    --ckpt save/KKI_cont/best_model.pt
python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic cont \
    --embedding_dim 90 \
    --ckpt save/KKI_cont/best_model.pt

python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic static \
    --embedding_dim 90
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic static \
    --embedding_dim 90 \
    --ckpt save/KKI_static1/best_model.pt
python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic static \
    --embedding_dim 90 \
    --ckpt save/KKI_static1/best_model.pt

python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --nhead 2
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --nhead 2 \
    --ckpt save/KKI_jump_nhead2/best_model.pt
python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --nhead 2 \
    --ckpt save/KKI_jump_nhead2/best_model.pt

python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --nhead 4
python evaluate.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --nhead 4 \
    --ckpt save/KKI_jump_nhead4/best_model.pt
python evaluate_matching.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --nhead 4 \
    --ckpt save/KKI_jump_nhead4/best_model.pt

# Generalization
python train.py \
    --dataset KKI \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --tag cross

python evaluate.py \
    --dataset COX2 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --tag KKI_cross \
    --ckpt save/KKI_jump_cross/best_model.pt
python evaluate.py \
    --dataset COX2_MD \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --tag KKI_cross \
    --ckpt save/KKI_jump_cross/best_model.pt
python evaluate.py \
    --dataset DHFR \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --tag KKI_cross \
    --ckpt save/KKI_jump_cross/best_model.pt
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --tag KKI_cross \
    --ckpt save/KKI_jump_cross/best_model.pt
python evaluate.py \
    --dataset MSRC-21 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 90 \
    --tag KKI_cross \
    --ckpt save/KKI_jump_cross/best_model.pt