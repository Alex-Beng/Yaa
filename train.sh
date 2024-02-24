python3 act_imitate_learning.py \
--task_name test \
--ckpt_dir ./models \
--kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0
