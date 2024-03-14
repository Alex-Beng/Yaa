# MLP
python mlp_imitate_learning.py `
--task_name nazuchi_beach_friendship `
--ckpt_dir ./models/models_mlp `
--kl_weight 100 --chunk_size 20 --hidden_dim 512 --batch_size 16 --dim_feedforward 2048 `
--num_epochs 2000  --lr 1e-4 `
--seed 0


# 512 2048
python act_imitate_learning.py `
--task_name nazuchi_beach_friendship `
--ckpt_dir ./models/models_512_2048 `
--kl_weight 100 --chunk_size 20 --hidden_dim 512 --batch_size 4 --dim_feedforward 2048 `
--num_epochs 2000  --lr 1e-4 `
--seed 0



# small
python act_imitate_learning.py `
--task_name nazuchi_beach_friendship `
--ckpt_dir ./models/models_small `
--kl_weight 10 --chunk_size 100 --hidden_dim 64 --batch_size 4 --dim_feedforward 128 `
--num_epochs 2000  --lr 1e-3 `
--seed 0

# 增大一倍
python act_imitate_learning.py `
--task_name nazuchi_beach_friendship `
--ckpt_dir ./models/models_128_256 `
--kl_weight 10 --chunk_size 100 --hidden_dim 128 --batch_size 4 --dim_feedforward 256 `
--num_epochs 2000  --lr 1e-3 `
--seed 0

# 仅增大hidden_dim
python act_imitate_learning.py `
--task_name nazuchi_beach_friendship `
--ckpt_dir ./models/models_128_128 `
--kl_weight 10 --chunk_size 100 --hidden_dim 128 --batch_size 4 --dim_feedforward 128 `
--num_epochs 2000  --lr 1e-3 `
--seed 0

# smaller
python act_imitate_learning.py `
--task_name nazuchi_beach_friendship `
--ckpt_dir ./models/models_smaller `
--kl_weight 10 --chunk_size 100 --hidden_dim 32 --batch_size 4 --dim_feedforward 64 `
--num_epochs 2000  --lr 1e-5 `
--seed 0


python act_imitate_learning.py `
--task_name nazuchi_beach_friendship `
--ckpt_dir ./models `
--kl_weight 10 --chunk_size 100 --hidden_dim 64 --batch_size 4 --dim_feedforward 1280 `
--num_epochs 2000  --lr 1e-5 `
--seed 0