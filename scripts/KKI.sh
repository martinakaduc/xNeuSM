python train.py --ngpu 1 \
                --dataset KKI \
                --batch_size 64 \
                --epoch 30 \
                --dropout_rate 0.0 \
                --tatic jump \
                --embedding_dim 190

# End2end evaluation
python evaluate.py --ngpu 1 \
                   --dataset KKI \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 190 \
                   --ckpt save/KKI_jump_1/save_29.pt

# Scalability evaluation
python evaluate.py --ngpu 1 \
                   --dataset KKI \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 190 \
                   --ckpt save/KKI_jump_1/save_29.pt \
                   --test_keys test_keys_dense_0_20.pkl

python evaluate.py --ngpu 1 \
                   --dataset KKI \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 190 \
                   --ckpt save/KKI_jump_1/save_29.pt \
                   --test_keys test_keys_dense_20_40.pkl

python evaluate.py --ngpu 1 \
                   --dataset KKI \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 190 \
                   --ckpt save/KKI_jump_1/save_29.pt \
                   --test_keys test_keys_dense_40_60.pkl

python evaluate.py --ngpu 1 \
                   --dataset KKI \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 190 \
                   --ckpt save/KKI_jump_1/save_29.pt \
                   --test_keys test_keys_dense_60_.pkl

python evaluate.py --ngpu 1 \
                   --dataset KKI \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 190 \
                   --ckpt save/KKI_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_0_20.pkl

python evaluate.py --ngpu 1 \
                   --dataset KKI \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 190 \
                   --ckpt save/KKI_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_20_40.pkl

python evaluate.py --ngpu 1 \
                   --dataset KKI \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 190 \
                   --ckpt save/KKI_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_40_60.pkl

python evaluate.py --ngpu 1 \
                   --dataset KKI \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 190 \
                   --ckpt save/KKI_jump_1/save_29.pt \
                   --test_keys test_keys_nondense_60_.pkl

# Explainability evaluation
python evaluate_matching.py --ngpu 1 \
                            --dataset KKI \
                            --batch_size 64 \
                            --dropout_rate 0.0 \
                            --tatic jump \
                            --embedding_dim 190 \
                            --ckpt save/KKI_jump_1/save_29.pt

python viz_matching.py --ngpu 1 \
                       --dataset KKI_test \
                       --dropout_rate 0.0\
                        --tatic jump \
                        --embedding_dim 190 \
                        --ckpt save/KKI_jump_1/save_29.pt \
                        --data_path data_real/datasets \
                        --source 49 --query 0 --iso \
                        --mapping_threshold 0.5

python viz_matching.py --ngpu 1 \
                       --dataset KKI_test \
                       --dropout_rate 0.0 \
                       --tatic jump \
                       --embedding_dim 190 \
                       --ckpt save/KKI_jump_1/save_29.pt \
                       --data_path data_real/datasets \
                       --source 49 --query 4 \
                       --mapping_threshold 0.5

# Abalation study
python train.py --ngpu 1 --dataset KKI --batch_size 64 --epoch 30 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --branch left
python evaluate.py --ngpu 1 --dataset KKI --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --branch left --ckpt save/KKI_jump_1_left/save_29.pt
python evaluate_matching.py --ngpu 1 --dataset KKI --batch_size 128 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --branch left --ckpt save/KKI_jump_1_left/save_29.pt

python train.py --ngpu 1 --dataset KKI --batch_size 64 --epoch 30 --dropout_rate 0.0 --tatic cont --embedding_dim 190 --branch left
python evaluate.py --ngpu 1 --dataset KKI --batch_size 64 --dropout_rate 0.0 --tatic cont --embedding_dim 190 --branch left --ckpt save/KKI_cont_1_left/save_29.pt
python evaluate_matching.py --ngpu 1 --dataset KKI --batch_size 128 --dropout_rate 0.0 --tatic cont --embedding_dim 190 --branch left --ckpt save/KKI_cont_1_left/save_29.pt

python train.py --ngpu 1 --dataset KKI --batch_size 64 --epoch 30 --dropout_rate 0.0 --tatic static --embedding_dim 190 --branch left
python evaluate.py --ngpu 1 --dataset KKI --batch_size 64 --dropout_rate 0.0 --tatic static --embedding_dim 190 --branch left --ckpt save/KKI_static_1_left/save_29.pt
python evaluate_matching.py --ngpu 1 --dataset KKI --batch_size 128 --dropout_rate 0.0 --tatic static --embedding_dim 190  --branch left --ckpt save/KKI_static_1_left/save_29.pt


python train.py --ngpu 1 --dataset KKI --batch_size 64 --epoch 30 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --branch right
python evaluate.py --ngpu 1 --dataset KKI --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --branch right --ckpt save/KKI_jump_1_right/save_29.pt
python evaluate_matching.py --ngpu 1 --dataset KKI --batch_size 128 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --branch right --ckpt save/KKI_jump_1_right/save_29.pt

python train.py --ngpu 1 --dataset KKI --batch_size 64 --epoch 30 --dropout_rate 0.0 --tatic cont --embedding_dim 190 --branch right
python evaluate.py --ngpu 1 --dataset KKI --batch_size 64 --dropout_rate 0.0 --tatic cont --embedding_dim 190 --branch right --ckpt save/KKI_cont_1_right/save_29.pt
python evaluate_matching.py --ngpu 1 --dataset KKI --batch_size 128 --dropout_rate 0.0 --tatic cont --embedding_dim 190  --branch right --ckpt save/KKI_cont_1_right/save_29.pt

python train.py --ngpu 1 --dataset KKI --batch_size 64 --epoch 30 --dropout_rate 0.0 --tatic static --embedding_dim 190 --branch right
python evaluate.py --ngpu 1 --dataset KKI --batch_size 64 --dropout_rate 0.0 --tatic static --embedding_dim 190 --branch right --ckpt save/KKI_static_1_right/save_29.pt
python evaluate_matching.py --ngpu 1 --dataset KKI --batch_size 128 --dropout_rate 0.0 --tatic static --embedding_dim 190  --branch right --ckpt save/KKI_static_1_right/save_29.pt


python train.py --ngpu 1 --dataset KKI --batch_size 64 --epoch 30 --dropout_rate 0.0 --tatic cont --embedding_dim 190
python evaluate.py --ngpu 1 --dataset KKI --batch_size 64 --dropout_rate 0.0 --tatic cont --embedding_dim 190 --ckpt save/KKI_cont_1/save_29.pt
python evaluate_matching.py --ngpu 1 --dataset KKI --batch_size 128 --dropout_rate 0.0 --tatic cont --embedding_dim 190 --ckpt save/KKI_cont_1/save_29.pt

python train.py --ngpu 1 --dataset KKI --batch_size 64 --epoch 30 --dropout_rate 0.0 --tatic static —nhop 1 --embedding_dim 190
python evaluate.py --ngpu 1 --dataset KKI --batch_size 64 --dropout_rate 0.0 --tatic static --embedding_dim 190 --ckpt save/KKI_static_1/save_29.pt
python evaluate_matching.py --ngpu 1 --dataset KKI --batch_size 128 --dropout_rate 0.0 --tatic static --embedding_dim 190 --ckpt save/KKI_static_1/save_29.pt


# Generalization
python train.py --ngpu 1 --dataset KKI --batch_size 64 --epoch 30 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --tag cross

python evaluate.py --ngpu 1 --dataset COX2 --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --ckpt save/KKI_jump_1_cross/save_29.pt
python evaluate.py --ngpu 1 --dataset COX2_MD --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --ckpt save/KKI_jump_1_cross/save_29.pt
python evaluate.py --ngpu 1 --dataset DHFR --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --ckpt save/KKI_jump_1_cross/save_29.pt
python evaluate.py --ngpu 1 --dataset DBLP-v1 --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --ckpt save/KKI_jump_1_cross/save_29.pt
python evaluate.py --ngpu 1 --dataset MSRC-21 --batch_size 64 --dropout_rate 0.0 --tatic jump --embedding_dim 190 --ckpt save/KKI_jump_1_cross/save_29.pt