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
                   --ckpt save/MSRC-21_jump/best_model.pt

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --directed \
                   --ckpt save/MSRC-21_jump_directed/best_model.pt

# Scalability evaluation
python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump/best_model.pt \
                   --test_keys test_keys_dense_0_20.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump/best_model.pt \
                   --test_keys test_keys_dense_20_40.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump/best_model.pt \
                   --test_keys test_keys_dense_40_60.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump/best_model.pt \
                   --test_keys test_keys_dense_60_.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump/best_model.pt \
                   --test_keys test_keys_nondense_0_20.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump/best_model.pt \
                   --test_keys test_keys_nondense_20_40.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump/best_model.pt \
                   --test_keys test_keys_nondense_40_60.pkl

python evaluate.py --dataset MSRC-21 \
                   --batch_size 64 \
                   --tatic jump \
                   --embedding_dim 7 \
                   --ckpt save/MSRC-21_jump/best_model.pt \
                   --test_keys test_keys_nondense_60_.pkl

# Explainability evaluation
python evaluate_matching.py --dataset MSRC-21 \
                            --batch_size 64 \
                            --tatic jump \
                            --embedding_dim 7 \
                            --ckpt save/MSRC-21_jump/best_model.pt

python evaluate_matching.py --dataset MSRC-21 \
                            --batch_size 64 \
                            --tatic jump \
                            --embedding_dim 7 \
                            --directed \
                            --ckpt save/MSRC-21_jump_directed/best_model.pt

# Abalation study
python train.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch left
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch left --ckpt save/MSRC-21_jump_left/best_model.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch left --ckpt save/MSRC-21_jump_left/best_model.pt

python train.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --branch left
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --branch left --ckpt save/MSRC-21_cont_left/best_model.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --branch left --ckpt save/MSRC-21_cont_left/best_model.pt

python train.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --branch left
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --branch left --ckpt save/MSRC-21_static1_left/best_model.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7  --branch left --ckpt save/MSRC-21_static1_left/best_model.pt


python train.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch right
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch right --ckpt save/MSRC-21_jump_right/best_model.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 7 --branch right --ckpt save/MSRC-21_jump_right/best_model.pt

python train.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --branch right
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --branch right --ckpt save/MSRC-21_cont_right/best_model.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7  --branch right --ckpt save/MSRC-21_cont_right/best_model.pt

python train.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --branch right
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --branch right --ckpt save/MSRC-21_static1_right/best_model.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7  --branch right --ckpt save/MSRC-21_static1_right/best_model.pt


python train.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --ckpt save/MSRC-21_cont/best_model.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic cont --embedding_dim 7 --ckpt save/MSRC-21_cont/best_model.pt

python train.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7
python evaluate.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --ckpt save/MSRC-21_static1/best_model.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 64 --tatic static --embedding_dim 7 --ckpt save/MSRC-21_static1/best_model.pt


python train.py --dataset MSRC-21 --batch_size 32 --tatic jump --embedding_dim 7 --nhead 2
python evaluate.py --dataset MSRC-21 --batch_size 32 --tatic jump --embedding_dim 7 --nhead 2 --ckpt save/MSRC-21_jump_nhead2/best_model.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 32 --tatic jump --embedding_dim 7 --nhead 2 --ckpt save/MSRC-21_jump_nhead2/best_model.pt

python train.py --dataset MSRC-21 --batch_size 16 --tatic jump --embedding_dim 7 --nhead 4
python evaluate.py --dataset MSRC-21 --batch_size 16 --tatic jump --embedding_dim 7 --nhead 4 --ckpt save/MSRC-21_jump_nhead4/best_model.pt
python evaluate_matching.py --dataset MSRC-21 --batch_size 16 --tatic jump --embedding_dim 7 --nhead 4 --ckpt save/MSRC-21_jump_nhead4/best_model.pt

# Generalization
python train.py --dataset MSRC-21 --batch_size 64 --tatic jump --embedding_dim 90 --tag cross

python evaluate.py --dataset DHFR --batch_size 64 --tatic jump --embedding_dim 90 --tag MSRC-21_cross --ckpt save/MSRC-21_jump_cross/best_model.pt
python evaluate.py --dataset COX2 --batch_size 64 --tatic jump --embedding_dim 90 --tag MSRC-21_cross --ckpt save/MSRC-21_jump_cross/best_model.pt
python evaluate.py --dataset COX2_MD --batch_size 64 --tatic jump --embedding_dim 90 --tag MSRC-21_cross --ckpt save/MSRC-21_jump_cross/best_model.pt
python evaluate.py --dataset KKI --batch_size 64 --tatic jump --embedding_dim 90 --tag MSRC-21_cross --ckpt save/MSRC-21_jump_cross/best_model.pt
python evaluate.py --dataset DBLP-v1 --batch_size 64 --tatic jump --embedding_dim 90 --tag MSRC-21_cross --ckpt save/MSRC-21_jump_cross/best_model.pt