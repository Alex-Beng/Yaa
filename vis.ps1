# PowerShell 调用如下命令从 0 -> $args[0]
# py visualize_hdf5.py --dataset_dir ./datasets/nazuchi_beach_friendship/ --episode_idx 1

# 循环
for ($i=0; $i -le $args[0]; $i++){
    Write-Host "visualize episode $i"
    python visualize_hdf5.py --dataset_dir ./datasets/nazuchi_beach_friendship/ --episode_idx $i
}