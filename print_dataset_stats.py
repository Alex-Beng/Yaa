# 打印出dataset的norm stats
# 单纯好奇均值和标准差是多少

import pickle

import IPython
e = IPython.embed

if __name__ == "__main__":
    pkl_path = './models/dataset_stats.pkl'
    with open(pkl_path, 'rb') as f:
        stats = pickle.load(f)
        print(stats.keys())
        # e()
        for k in stats:
            print(k)
            print(stats[k].shape)
            if k != 'example_state':
                print(stats[k])