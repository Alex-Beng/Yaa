# mlp
python mlp_infer.py `
--onscreen_render `
--ckpt_dir ./models/models_mlp --ckpt_name policy_epoch_200_seed_0.ckpt `
--task_name nazuchi_beach_friendship `
--seed 1000 `
--chunk_size 100 --hidden_dim 512 --dim_feedforward 128 




# local small

python act_infer.py `
--onscreen_render `
--ckpt_dir ./models/models_small --ckpt_name policy_epoch_600_seed_0.ckpt `
--task_name nazuchi_beach_friendship `
--seed 1000 `
--chunk_size 100 --hidden_dim 64 --dim_feedforward 128 


# local 增大一倍

python act_infer.py `
--onscreen_render `
--ckpt_dir ./models/models_128_256 --ckpt_name policy_epoch_500_seed_0.ckpt `
--task_name nazuchi_beach_friendship `
--seed 1000 `
--chunk_size 100 --hidden_dim 128 --dim_feedforward 256 



# autodl models_smaller

python act_infer.py `
--onscreen_render `
--ckpt_dir ./models/autodl/models_smaller --ckpt_name policy_epoch_1500_seed_0.ckpt `
--task_name nazuchi_beach_friendship `
--seed 1000 `
--chunk_size 20 --hidden_dim 32 --dim_feedforward 64 

# autodl models_small

python act_infer.py `
--onscreen_render `
--ckpt_dir ./models/autodl/models_small --ckpt_name policy_best.ckpt `
--task_name nazuchi_beach_friendship `
--seed 1000 `
--chunk_size 20 --hidden_dim 64 --dim_feedforward 128 


# autodl models

python act_infer.py `
--onscreen_render `
--ckpt_dir ./models/autodl/models --ckpt_name policy_best.ckpt `
--task_name nazuchi_beach_friendship `
--seed 1000 `
--chunk_size 20 --hidden_dim 768 --dim_feedforward 3200 


