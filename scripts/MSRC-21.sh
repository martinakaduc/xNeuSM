python train.py --dataset MSRC-21 \
                --batch_size 64 \
                --tatic jump \
                --embedding_dim 7

python train.py --dataset MSRC-21 \
                --batch_size 64 \
                --tatic jump \
                --embedding_dim 7 \
                --directed

# End2end evaluation
python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --directed \
                   --ckpt save/MSRC-21_jump_1_directed/save_29.pt

# Scalability evaluation
python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_dense_0_20.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_dense_20_40.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_dense_40_60.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_dense_60_.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_0_20.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_20_40.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_40_60.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_60_.pkl

# Explainability evaluation
python evaluate_matching.py --dataset MSRC-21 \
                            --batch_size 64 \
                            --tatic jump \
                            --embedding_dim 7 \
                            --ckpt save/MSRC-21_jump_1/save_29.pt

python evaluate_matching.py --dataset MSRC-21 \
                            --batch_size 64 \
                            --tatic jump \
                            --embedding_dim 7 \
                            --directed \
                            --ckpt save/MSRC-21_jump_1_directed/save_29.pt

# Abalation study
python train.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch left
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch left --ckpt save/MSRC-21_jump_1_left/save_29.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch left --ckpt save/MSRC-21_jump_1_left/save_29.pt

python train.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --branch left
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --branch left --ckpt save/MSRC-21_cont_1_left/save_29.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --branch left --ckpt save/MSRC-21_cont_1_left/save_29.pt

python train.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --branch left
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --branch left --ckpt save/MSRC-21_static_1_left/save_29.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7  --branch left --ckpt save/MSRC-21_static_1_left/save_29.pt


python train.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch right
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch right --ckpt save/MSRC-21_jump_1_right/save_29.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch right --ckpt save/MSRC-21_jump_1_right/save_29.pt

python train.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --branch right
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --branch right --ckpt save/MSRC-21_cont_1_right/save_29.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7  --branch right --ckpt save/MSRC-21_cont_1_right/save_29.pt

python train.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --branch right
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --branch right --ckpt save/MSRC-21_static_1_right/save_29.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7  --branch right --ckpt save/MSRC-21_static_1_right/save_29.pt


python train.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --ckpt save/MSRC-21_cont_1/save_29.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --ckpt save/MSRC-21_cont_1/save_29.pt

python train.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --ckpt save/MSRC-21_static_1/save_29.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --ckpt save/MSRC-21_static_1/save_29.pt

# Generalization
python train.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 90 --tag cross

python evaluate.py --dataset DHFR --batch_size 64 --tatic jump --embedding_dim 90 --ckpt save/MSRC-21_jump_1_cross/save_29.pt
python evaluate.py --dataset COX2 --batch_size 64 --tatic jump --embedding_dim 90 --ckpt save/MSRC-21_jump_1_cross/save_29.pt
python evaluate.py --dataset COX2_MD --batch_size 64 --tatic jump --embedding_dim 90 --ckpt save/MSRC-21_jump_1_cross/save_29.pt
python evaluate.py --dataset KKI --batch_size 64 --tatic jump --embedding_dim 90 --ckpt save/MSRC-21_jump_1_cross/save_29.pt
python evaluate.py --dataset DBLP-v1 --batch_size 64 --tatic jump --embedding_dim 90 --ckpt save/MSRC-21_jump_1_cross/save_29.pt