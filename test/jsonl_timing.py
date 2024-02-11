# 读取一个jsonl文件
# 每一行json里的timestamp提取出来进行分析

import json

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def main():
    # 第一个命令行参数是file path
    import sys
    if len(sys.argv) < 2:
        print('Usage: python jsonl_timing.py <file_path>')
        return
    file_path = sys.argv[1]
    tss = []
    for data in read_jsonl(file_path):
        if 'timestamp' in data:
            if 'type' in data and data['type'] == 'mouse' and data['event_type'] == 0:
                # continue
                pass
            tss.append(data['timestamp'])
    # print(tss)
    # ns -> ms
    tss = [ts / 1000000 for ts in tss]
    diffs = [tss[i] - tss[i-1] for i in range(1, len(tss))]
    # print(diffs)

    # print(tss)
    min_diff = min(diffs)
    print(min_diff)
    # 换算Hz
    print(1 / min_diff * 1000)

    max_diff = max(diffs)
    print(max_diff)
    print(1 / max_diff * 1000)

    # 平均频率
    print(len(diffs) / sum(diffs) * 1000)
    
    # 可视化diff
    import matplotlib.pyplot as plt
    plt.xlabel('index')
    plt.ylabel('time / ms')
    plt.title('ms kb cap diff')
    plt.plot(diffs)

    plt.show()

if __name__ == '__main__':
    main()
    # 