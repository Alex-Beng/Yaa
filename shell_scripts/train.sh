python3 act_imitate_learning.py \
--task_name nazuchi_beach_friendship \
--ckpt_dir ./models \
--kl_weight 10 --chunk_size 100 --hidden_dim 64 --batch_size 4 --dim_feedforward 1280 \
--num_epochs 2000  --lr 1e-5 \
--seed 0
