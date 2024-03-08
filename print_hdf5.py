import h5py
from constants import STATE_DIM

min_values = [0] * (STATE_DIM-3)
max_values = [0] * (STATE_DIM-3)


def print_structure(hdf_file):
    def print_group(group, indent=''):
        global min_values, max_values
        print(indent, group.name)
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                print_group(value, indent + '  ')
            else:
                print(indent + '  ', key)
                # 打印数据集的大小
                if isinstance(value, h5py.Dataset):
                    # 计算value中数值范围
                    if key == 'action':
                        # 统计 action 每个state dim 里的最大最小值
                        for ai in range(STATE_DIM-3):
                            min_values[ai] = min(min_values[ai], value[:, ai].min())
                            max_values[ai] = max(max_values[ai], value[:, ai].max())
                        print(f'{min_values}\n {max_values}')
                    print(indent + '  ', value.shape)

    with h5py.File(hdf_file, 'r') as f:
        print_group(f)
def print_dataset_sizes(hdf_file):
    with h5py.File(hdf_file, 'r') as f:
        for name, dataset in f.items():
            if isinstance(dataset, h5py.Dataset):
                print(f'Dataset {name} contains {dataset.size} elements')

for i in range(50):
    hdf5_paht = f'./datasets/nazuchi_beach_friendship/{i}.hdf5'
    print_structure(hdf5_paht)
    print_dataset_sizes(hdf5_paht)
    print(min_values)
    print(max_values)