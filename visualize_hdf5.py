# 从hdf5中可视化数据
# 先简单的cv imshow + print state&action

import os
import random
import numpy as np
import argparse

import cv2
import h5py
import matplotlib.pyplot as plt



from constants import DT, STATE_DIM, SN_idx2key


def load_hdf5(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        state = f['/obs/state'][()]
        action = f['/action'][()]
        image_dict = dict()
        for cam_name in f['/obs/images/'].keys():
            image_dict[cam_name] = f[f'/obs/images/{cam_name}'][()]

    return state, action, image_dict

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    if episode_idx is None:
        episode_idx = 0
    hdf5_path = os.path.join(dataset_dir, f'{episode_idx}.hdf5')
    states, actions, image_dict = load_hdf5(hdf5_path)

    plot_dstb(actions)

    save_videos(image_dict, states, actions, DT, os.path.join(dataset_dir, f'{episode_idx}_hdf5.mp4'))

def save_videos(image_dict, states, actions, dt, video_path):
    from resample_record import state_to_str

    # act 里面还做了list和dict的判断，这里直接assert！只用dict
    assert isinstance(image_dict, dict)
    cam_names = list(image_dict.keys())
    all_cam_videos = []
    for cam_name in cam_names:
        all_cam_videos.append(image_dict[cam_name])
    # batch, height, width, channel
    # concat in width
    all_cam_videos = np.concatenate(all_cam_videos, axis=2)
    n_frames, h, w, _ = all_cam_videos.shape
    print(h, w)
    
    fps = int(1 / dt)
    out = cv2.VideoWriter(f'{video_path}', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for i in range(n_frames):
        frame = all_cam_videos[i]
        # TODO:
        # 在图像上绘制鼠标的dx dy
        # 以及打印key state
        print(f'{states[i, :]}')
        print(f'{actions[i, :]}')

        state = states[i]
        action = actions[i]
        
        # 不关心鼠标的roll dx dy
        kb_events = []
        for state_id in range(STATE_DIM-2):
            if state[state_id] != action[state_id]:
                # 说明是press or release
                is_press = True if action[state_id] == 1 else False
                key_name = SN_idx2key[state_id]
                key_name = 'sp' if key_name == ' ' else key_name
                kb_events.append(f"{key_name} {'down' if is_press else 'up'}")
        kb_events_str = ','.join(kb_events)
        # 在frame 上显示 str
        frame = cv2.putText(frame, kb_events_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
        # 在frame 中间绘制一个箭头，表示dx dy
        dx, dy = action[-2], action[-1]
        dx, dy = int(dx), int(dy)
        # print(dx, dy)
        frame = cv2.arrowedLine(frame, (w//2, h//2), (w//2+dx, h//2+dy), (0, 255, 0), 2)


        cv2.imshow('frame', frame)
        print(f'frame {i}/{n_frames}')

        wt = 0 if len(kb_events) else 1
        wt = 1
        if cv2.waitKey(wt) & 0xFF == ord('q'):
            break

        out.write(frame)

    out.release()


def plot_dstb(actions):
    # record_len, state_dim
    # print(actions.shape)
    record_len, _ = actions.shape

    # 统计各dim是否全部为0
    is_all_zero = [0] * STATE_DIM
    for i in range(STATE_DIM):
        if np.min(actions[:, i]) == np.max(actions[:, i]):
            print(i, SN_idx2key[i])
            is_all_zero[i] = 1
    
    # plot time to dim
    for i in range(STATE_DIM):
        if is_all_zero[i]:
            continue
        plt.figure(f"{SN_idx2key[i]}")
        data = actions[:, i]
        plt.plot(data)

    # plot dx, dy 散点图
    dx = actions[:, -2]
    dy = actions[:, -1]
    plt.figure("dx dy")
    plt.scatter(dx, dy)
    # print mean and std of dx dy
    print(f'mean dx {np.mean(dx)} std dx {np.std(dx)}')
    print(f'mean dy {np.mean(dy)} std dy {np.std(dy)}')

    plt.show(block=True)
    # exit()
    # 绘制action的分布
    # 检查
def plot_test():
    # 绘制测试
    # 绘制一维数据随时间的变化
    # 绘制二维散点图

    # 1. 一维数据随时间的变化
    n_frames = 100
    time = np.arange(n_frames)
    data = np.random.randn(n_frames)
    plt.plot(time, data)

    # 2. 二维散点图
    n_points = 100
    x = np.random.randn(n_points)
    y = np.random.randn(n_points)
    plt.figure()
    plt.scatter(x, y)

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))

