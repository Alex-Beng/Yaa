# 从hdf5中可视化数据
# 先简单的cv imshow + print state&action

import os
import numpy as np
import cv2
import h5py
import argparse

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
        for state_id in range(STATE_DIM-3):
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
        if cv2.waitKey(wt) & 0xFF == ord('q'):
            break

        out.write(frame)

    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))

