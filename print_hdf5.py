import h5py

def print_structure(hdf_file):
    def print_group(group, indent=''):
        print(indent, group.name)
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                print_group(value, indent + '  ')
            else:
                print(indent + '  ', key)
                # 打印数据集的大小
                if isinstance(value, h5py.Dataset):
                    print(indent + '  ', value.shape)

    with h5py.File(hdf_file, 'r') as f:
        print_group(f)
def print_dataset_sizes(hdf_file):
    with h5py.File(hdf_file, 'r') as f:
        for name, dataset in f.items():
            if isinstance(dataset, h5py.Dataset):
                print(f'Dataset {name} contains {dataset.size} elements')

hdf5_paht = './build/test/1.hdf5'
print_structure(hdf5_paht)
print_dataset_sizes(hdf5_paht)