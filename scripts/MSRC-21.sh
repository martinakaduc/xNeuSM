python train.py --ngpu 1 \
                --dataset MSRC-21 \
                --batch_size 64 \
                --epoch 30 \
                --dropout_rate 0.0 \
                --tatic jump \
                --embedding_dim 141
# End2end evaluation
python evaluate.py --ngpu 1 \
                   --dataset MSRC-21 \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 141 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt

# Scalability evaluation
python evaluate.py --ngpu 1 \
                   --dataset MSRC-21 \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 141 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_dense_0_20.pkl

python evaluate.py --ngpu 1 \
                   --dataset MSRC-21 \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 141 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_dense_20_40.pkl

python evaluate.py --ngpu 1 \
                   --dataset MSRC-21 \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 141 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_dense_40_60.pkl

python evaluate.py --ngpu 1 \
                   --dataset MSRC-21 \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 141 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_dense_60_.pkl

python evaluate.py --ngpu 1 \
                   --dataset MSRC-21 \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 141 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_0_20.pkl

python evaluate.py --ngpu 1 \
                   --dataset MSRC-21 \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 141 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_20_40.pkl

python evaluate.py --ngpu 1 \
                   --dataset MSRC-21 \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 141 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_40_60.pkl

python evaluate.py --ngpu 1 \
                   --dataset MSRC-21 \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 141 \
                   --ckpt save/MSRC-21_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_60_.pkl

# Explainability evaluation
python evaluate_matching.py --ngpu 1 \
                            --dataset MSRC-21 \
                            --batch_size 64 \
                            --dropout_rate 0.0 \
                            --tatic jump \
                            --embedding_dim 141 \
                            --ckpt save/MSRC-21_jump_1/save_29.pt

# Generalization
python train.py --ngpu 1 --dataset MSRC-21 --batch_size 64 --epoch 30 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --tag cross

python evaluate.py --ngpu 1 --dataset DHFR --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --ckpt save/MSRC-21_jump_1_cross/save_29.pt
python evaluate.py --ngpu 1 --dataset COX2 --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --ckpt save/MSRC-21_jump_1_cross/save_29.pt
python evaluate.py --ngpu 1 --dataset COX2_MD --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --ckpt save/MSRC-21_jump_1_cross/save_29.pt
python evaluate.py --ngpu 1 --dataset KKI --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --ckpt save/MSRC-21_jump_1_cross/save_29.pt
python evaluate.py --ngpu 1 --dataset DBLP-v1 --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --ckpt save/MSRC-21_jump_1_cross/save_29.pt