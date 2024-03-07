python act_infer.py \
--onscreen_render \
--ckpt_dir ./models --ckpt_name policy_best.ckpt \
--task_name nazuchi_beach_friendship \
--seed 1000 \
--chunk_size 40 --hidden_dim 512 --dim_feedforward 3200 \
--temporal_agg 