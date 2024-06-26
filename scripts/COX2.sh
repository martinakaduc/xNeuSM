python train.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 8

python train.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 8 \
    --directed

# End2end evaluation
python evaluate.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 8 \
    --ckpt save/COX2_jump/best_model.pt

# Directed evaluation
python evaluate.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 8 \
    --directed \
    --ckpt save/COX2_jump_directed/best_model.pt

# Scalability evaluation
python evaluate.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 8 \
    --ckpt save/COX2_jump/best_model.pt \
    --test_keys test_keys_nondense_0_20.pkl

python evaluate.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 8 \
    --ckpt save/COX2_jump/best_model.pt \
    --test_keys test_keys_nondense_20_40.pkl

python evaluate.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 8 \
    --ckpt save/COX2_jump/best_model.pt \
    --test_keys test_keys_nondense_40_60.pkl

# Explainability evaluation
python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 8 \
    --ckpt save/COX2_jump/best_model.pt

python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 8 \
    --directed \
    --ckpt save/COX2_jump_directed/best_model.pt

# Abalation study
python train.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 8 \
    --branch left
python evaluate.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 8 \
    --branch left \
    --ckpt save/COX2_jump_left/best_model.pt
python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 8 \
    --branch left \
    --ckpt save/COX2_jump_left/best_model.pt

python train.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 8 \
    --branch left
python evaluate.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 8 \
    --branch left \
    --ckpt save/COX2_cont_left/best_model.pt
python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic cont \
    --embedding_dim 8 \
    --branch left \
    --ckpt save/COX2_cont_left/best_model.pt

python train.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 8 \
    --branch left
python evaluate.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 8 \
    --branch left \
    --ckpt save/COX2_static1_left/best_model.pt
python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic static \
    --embedding_dim 8  \
    --branch left \
    --ckpt save/COX2_static1_left/best_model.pt


python train.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 8 \
    --branch right
python evaluate.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic jump \
    --embedding_dim 8 \
    --branch right \
    --ckpt save/COX2_jump_right/best_model.pt
python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 8 \
    --branch right \
    --ckpt save/COX2_jump_right/best_model.pt

python train.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 8 \
    --branch right
python evaluate.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 8 \
    --branch right \
    --ckpt save/COX2_cont_right/best_model.pt
python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic cont \
    --embedding_dim 8  \
    --branch right \
    --ckpt save/COX2_cont_right/best_model.pt

python train.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 8 \
    --branch right
python evaluate.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 8 \
    --branch right \
    --ckpt save/COX2_static1_right/best_model.pt
python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic static \
    --embedding_dim 8  \
    --branch right \
    --ckpt save/COX2_static1_right/best_model.pt


python train.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 8
python evaluate.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic cont \
    --embedding_dim 8 \
    --ckpt save/COX2_cont/best_model.pt
python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic cont \
    --embedding_dim 8 \
    --ckpt save/COX2_cont/best_model.pt

python train.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 8
python evaluate.py \
    --dataset COX2 \
    --batch_size 256 \
    --tatic static \
    --embedding_dim 8 \
    --ckpt save/COX2_static1/best_model.pt
python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic static \
    --embedding_dim 8 \
    --ckpt save/COX2_static1/best_model.pt


python train.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 8 \
    --nhead 2
python evaluate.py \
    --dataset COX2 \
    --batch_size 128 \
    --tatic jump \
    --embedding_dim 8 \
    --nhead 2 \
    --ckpt save/COX2_jump_nhead2/best_model.pt
python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 8 \
    --nhead 2 \
    --ckpt save/COX2_jump_nhead2/best_model.pt

python train.py \
    --dataset COX2 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 8 \
    --nhead 4
python evaluate.py \
    --dataset COX2 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 8 \
    --nhead 4 \
    --ckpt save/COX2_jump_nhead4/best_model.pt
python evaluate_matching.py \
    --dataset COX2 \
    --batch_size 32 \
    --tatic jump \
    --embedding_dim 8 \
    --nhead 4 \
    --ckpt save/COX2_jump_nhead4/best_model.pt

# Generalization
python train.py \
    --dataset COX2 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag cross

python evaluate.py \
    --dataset COX2 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag COX2_cross \
    --ckpt save/COX2_jump_cross/best_model.pt
python evaluate.py \
    --dataset DHFR \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag COX2_cross \
    --ckpt save/COX2_jump_cross/best_model.pt
python evaluate.py \
    --dataset COX2_MD \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag COX2_cross \
    --ckpt save/COX2_jump_cross/best_model.pt
python evaluate.py \
    --dataset DBLP-v1 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag COX2_cross \
    --ckpt save/COX2_jump_cross/best_model.pt
python evaluate.py \
    --dataset MSRC-21 \
    --batch_size 64 \
    --tatic jump \
    --embedding_dim 190 \
    --tag COX2_cross \
    --ckpt save/COX2_jump_cross/best_model.pt

