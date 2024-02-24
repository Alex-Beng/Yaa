# 从hdf5中可视化数据
# 先简单的cv imshow + print state&action

import os
import numpy as np
import cv2
import h5py
import argparse

from constants import DT


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
    state, action, image_dict = load_hdf5(hdf5_path)

    save_videos(image_dict, state, action, DT, os.path.join(dataset_dir, f'{episode_idx}_hdf5.mp4'))

def save_videos(image_dict, state, action, dt, video_path):
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
    
    keyboard_state = [0] * 14
    fps = int(1 / dt)
    out = cv2.VideoWriter(f'{video_path}', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for i in range(n_frames):
        frame = all_cam_videos[i]
        # TODO:
        # 在图像上绘制鼠标的dx dy
        # 以及打印key state
        print(f'{state[i, :]}')
        print(f'{action[i, :]}')
        cv2.imshow('frame', frame)
        print(f'frame {i}/{n_frames}')

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

        out.write(frame)

    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))

