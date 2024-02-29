# shell 调用如下命令从 0 -> $1
# py visualize_hdf5.py --dataset_dir ./datasets/nazuchi_beach_friendship/ --episode_idx 1

# 循环
for i in `seq 0 $1`
do
    echo "visualize episode $i"
    python visualize_hdf5.py --dataset_dir ./datasets/nazuchi_beach_friendship/ --episode_idx $i
done
