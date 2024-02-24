# 用于从原始录制中，生成 obs, action 对
"""
For each timestep:
observations
- images
    - each_cam_name     (480, 640, 3) 'uint8'
- qpos                  (14,)         'float64'
- qvel                  (14,)         'float64'

action                  (14,)         'float64'
"""
# HERE in Yaa
"""
For each timestep:
observations
- images
    - full_gray_view    (480, 640, 3) 'uint8'
    - full_alpha_view   (480, 640, 3) 'uint8'
- mskb_status           (19,)         'float64'

mskb_status             (19,)         'float64'
"""
import os
import json
import time
from copy import deepcopy

import cv2
import h5py

from constants import DT, STATE_DIM, SC_sc2idx, SC_idx2sc, SN_idx2key, SN, CAMERA_NAMES

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def test_resize(video_path: str):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(video_path)
    frame_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_cnt += 1
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('frame', frame)
        # 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    print(frame_cnt)

def test_sample_video(video_path: str, video_samp_idxs: list):
    cap = cv2.VideoCapture(video_path)
    frame_cnt = 0
    sleep_dt_in_ms = DT * 1000
    sleep_dt_in_ms = int(sleep_dt_in_ms)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_cnt in video_samp_idxs:
            cv2.imshow('frame', frame)
        if cv2.waitKey(sleep_dt_in_ms) & 0xFF == ord('q'):
            break
        frame_cnt += 1
    cap.release()

def check_record_idx(record_folder, idx):
    # 检查
    # {idx}_alpha.mp4, {idx}_mskb.jsonl, {idx}_video.jsonl, {idx}.mp4
    # 是否存在
    if not os.path.exists(os.path.join(record_folder, f'{idx}_alpha.mp4')):
        return False
    if not os.path.exists(os.path.join(record_folder, f'{idx}_mskb.jsonl')):
        return False
    if not os.path.exists(os.path.join(record_folder, f'{idx}_video.jsonl')):
        return False
    if not os.path.exists(os.path.join(record_folder, f'{idx}.mp4')):
        return False
    return True

def get_sample_timestamps(record_folder, idx):
    # 读取jsonl，返回需要采样的时间点
    video_jsonl_path = os.path.join(record_folder, f'{idx}_video.jsonl')
    mskb_jsonl_path  = os.path.join(record_folder, f'{idx}_mskb.jsonl')
    
    video_timestamps = [line['timestamp'] for line in read_jsonl(video_jsonl_path)]
    mskb_timestamps = [line['timestamp'] for line in read_jsonl(mskb_jsonl_path)]

    # 选择起始点
    start_ts = max(video_timestamps[0], mskb_timestamps[0])
    # 选择结束点
    end_ts = min(video_timestamps[-1], mskb_timestamps[-1])

    # 生成采样点
    dt_in_ns = int(DT * 1e9)
    sample_timestamps = [ts for ts in range(start_ts, end_ts, dt_in_ns)]

    video_sample_idx = []
    t_v_idx = 0
    for ts in sample_timestamps:
        while video_timestamps[t_v_idx] < ts:
            t_v_idx += 1
    
        video_sample_idx.append(t_v_idx - 1)
    assert len(sample_timestamps) == len(video_sample_idx)
    return sample_timestamps, video_sample_idx


def do_sample(record_folder, idx, tss, video_samp_idxs):
    # for video, just sample the nearest frame
    # for mskb, accumulate the events in the gap
    sampled_video_frames = []
    sampled_video_frames_idx = []
    # 分别对应/obs/state和/action
    sampled_mskb_states = []
    sampled_mskb_events = []

    # sample the frames first
    print(f"Sampling video {idx}...")
    t0 = time.time()
    rgb_video_path = os.path.join(record_folder, f'{idx}.mp4')
    alpha_video_path = os.path.join(record_folder, f'{idx}_alpha.mp4')
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    alpha_cap = cv2.VideoCapture(alpha_video_path)
    frame_cnt = -1
    
    # 由于set会去重，所以维护一个head作为samp_idxs的索引
    video_samped_head = 0
    debug = False
    if debug:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow('alpha', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = rgb_cap.read()
        ret_alpha, alpha_frame = alpha_cap.read()
        if not ret or not ret_alpha:
            break
        frame_cnt += 1
        # print(f'frame_cnt: {frame_cnt}, video_samped_head: {video_samped_head}, video_samp_idxs[video_samped_head]: {video_samp_idxs[video_samped_head]}')
        if frame_cnt < video_samp_idxs[video_samped_head]:
            continue
        elif frame_cnt == video_samp_idxs[video_samped_head]:
            if debug:
                cv2.imshow('frame', frame)
                cv2.imshow('alpha', alpha_frame)
            frame = cv2.resize(frame, (640, 480))
            alpha_frame = cv2.resize(alpha_frame, (640, 480))
            # 转换alpha为rgb
            # alpha_frame = cv2.cvtColor(alpha_frame, cv2.COLOR_GRAY2BGR)
            assert frame.shape == alpha_frame.shape
            # 因为可能存在多个采样点对应同一帧，所以这里用while
            while video_samped_head < len(video_samp_idxs) and \
                frame_cnt == video_samp_idxs[video_samped_head]:

                sampled_video_frames.append((frame, alpha_frame))
                sampled_video_frames_idx.append(frame_cnt)
                video_samped_head += 1
        
        if video_samped_head >= len(video_samp_idxs):
            # 实际无需判断，在生成video_samp_idxs时已经保证了
            break
        
        if debug and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cost time
    print(f'video cost time: {time.time() - t0}')
    # replay the video
    if debug:
        for frame, alpha_frame in sampled_video_frames:
            cv2.imshow('frame', frame)
            cv2.imshow('alpha', alpha_frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break


    # then sample mskb 
    print(f"Sampling mskb {idx}...")
    t0 = time.time()
    # 首先抛弃掉tts[0]时刻之前的msbk事件camera_names = ['rgb', 'alpha
    t_mskb_idx = 0
    mskb_jsonl_path  = os.path.join(record_folder, f'{idx}_mskb.jsonl')
    all_mskb_events = list(read_jsonl(mskb_jsonl_path))
    while all_mskb_events[t_mskb_idx]['timestamp'] < tss[0]:
        t_mskb_idx += 1
    print(f'len tss: {len(tss)}')
    # 然后开始采样
    for ts in tss:
        curr_events = []
        while all_mskb_events[t_mskb_idx]['timestamp'] < ts:
            curr_events.append(all_mskb_events[t_mskb_idx])
            t_mskb_idx += 1
        # accumulate the events to one
        # 经过测试，在关闭提高鼠标精确度的设置后，鼠标的移动满足可叠加性
        # 所以直接叠加
        # 引入假设，在O中叠加性也是成立的
            
        # 初值不应为0，应维持上一次的状态
        if len(sampled_mskb_events) == 0:
            event_state = [0] * STATE_DIM
        else:
            event_state = deepcopy(sampled_mskb_events[-1])
            # 设置鼠标的移动为0
            event_state[SN['Mdx']] = 0
            event_state[SN['Mdy']] = 0
            event_state[SN['MRo']] = 0
        # 所以目前的event_state就是当前frame的状态
        sampled_mskb_states.append(deepcopy(event_state))

        click_cnt = 0
        for event in curr_events:
            if event["type"] == 'keyboard':
                # 如果是KOI
                if event['scancode'] in SC_sc2idx:
                    # 如果是摁下
                    if event['event_type'] == 0:
                        event_state[SC_sc2idx[event['scancode']]] = 1
                    else:
                        # 如果是松开，print看看
                        if event_state[SC_sc2idx[event['scancode']]] == 1:
                            click_cnt += 1
                            print(f'event click key: {SN_idx2key[SC_sc2idx[event["scancode"]]]}')
                        event_state[SC_sc2idx[event['scancode']]] = 0
            elif event["type"] == 'mouse':
                # 如果是鼠标移动
                if event['event_type'] == 0:
                    event_state[SN['Mdx']] += event['dx']
                    event_state[SN['Mdy']] += event['dy']
                # 如果是左键摁下 / 松开
                elif event['event_type'] == 1:
                    event_state[SN['ML']] = 1
                elif event['event_type'] == 2:
                    if event_state[SN['ML']] == 1:
                        click_cnt += 1
                        print(f'event click left mouse')
                    event_state[SN['ML']] = 0
                # 如果是右键摁下 / 松开
                # 转换为Lshift的摁下 / 松开
                elif event['event_type'] == 4:
                    event_state[SN['LS']] = 1
                elif event['event_type'] == 8:
                    if event_state[SN['LS']] == 1:
                        click_cnt += 1
                        print(f'event click right mouse')
                    event_state[SN['LS']] = 0
                # 如果是滚轮滚动
                elif event['event_type'] == 1024:
                    event_state[SN['MRo']] += event['dy']//120
                # middle button not in KOI
        sampled_mskb_events.append(event_state)
        # print(f'event_state: {event_state}')
        if click_cnt > 0:
            print(f'click_cnt: {click_cnt}')
    # cost time
    print(f'mskb cost time: {time.time() - t0}')
    # print sampel frames & actions len
    print(f'len(sampled_video_frames): {len(sampled_video_frames)}')
    print(f'len(sampled_mskb_states): {len(sampled_mskb_states)}')
    print(f'len(sampled_mskb_events): {len(sampled_mskb_events)}')
    print(sampled_mskb_events[:5])

    # replay the video
    if debug:
        repaly_sampled_video_and_mskb(sampled_video_frames, sampled_mskb_events)
        
    # save to hdf5 like act
    save_to_hdf5(sampled_video_frames, sampled_mskb_states, sampled_mskb_events, record_folder, idx)


def state_to_str(state, former_state):
    # for mouse, show dx dy ml mro
    mouse_str = f'Mdx: {state[SN["Mdx"]]}, Mdy: {state[SN["Mdy"]]}, ML: {state[SN["ML"]]}, MRo: {state[SN["MRo"]]}'
    # for keyboard, show the change-state keys
    key_str = ''
    for i in range(14):
        if state[i] != former_state[i]:
            key_str += f'{SN_idx2key[i]} is {"press" if state[i] == 1 else "release"}; '

    return mouse_str, key_str


def repaly_sampled_video_and_mskb(sampled_video_frames, sampled_mskb_events):
    assert len(sampled_video_frames) == len(sampled_mskb_events)
    replay_len = len(sampled_video_frames)
    # replay the video
    keyboard_state = [0] * 14
    for i in range(replay_len):
        frame, alpha_frame = sampled_video_frames[i]
        state = sampled_mskb_events[i]
        
        # 分两行字显示鼠标和键盘事件
        mouse_str, key_str = state_to_str(state, keyboard_state)
        keyboard_state = state
        # cv2.putText(frame, mouse_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, key_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        dx, dy = state[SN['Mdx']], state[SN['Mdy']]
        # 根据dx，dy在图像中间绘制一个箭头
        cv2.arrowedLine(frame, (320, 240), (320+dx, 240+dy), (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        if key_str != "":
            cv2.waitKey(0)
            continue
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    pass

def save_to_hdf5(sampled_video_frames, 
                sampled_mskb_states, sampled_mskb_events, 
                record_folder, idx):
    # 直接和record放一个文件夹力，命名为{idx}.hdf5
    data_dict = {
        '/obs/state': [],
        '/action': []
    }
    for cam_name in CAMERA_NAMES:
        data_dict[f'/obs/images/{cam_name}'] = []
    
    assert len(sampled_video_frames) == len(sampled_mskb_events)
    assert len(sampled_mskb_states) == len(sampled_mskb_events)
    max_timestamp = len(sampled_video_frames)
    while sampled_video_frames:
        frames = sampled_video_frames.pop(0)
        state = sampled_mskb_states.pop(0)
        action = sampled_mskb_events.pop(0)
        data_dict['/obs/state'].append(state)
        data_dict['/action'].append(action)
        for cam_idx, cam_name in enumerate(CAMERA_NAMES):
            data_dict[f'/obs/images/{cam_name}'].append(frames[cam_idx])

    # HDF5
    t0 = time.time()
    hdf5_path = os.path.join(record_folder, f'{idx}.hdf5')
    with h5py.File(hdf5_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        obs = root.create_group('obs')
        image = obs.create_group('images')
        for cam_name in CAMERA_NAMES:
            image.create_dataset(cam_name, (max_timestamp, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
        obs.create_dataset('state', (max_timestamp, 19), dtype='float64')
        act = root.create_dataset('action', (max_timestamp, 19), dtype='float64')

        for name, array in data_dict.items():
            print(name)
            root[name][...] = array
    print(f'save hdf5 cost time: {time.time() - t0}')


def main(output_path: str, task_name: str):
    # 自适应查找各个录制。
    record_folder = os.path.join(output_path, task_name)
    if not os.path.exists(record_folder):
        print(f'Error: {record_folder} not exists.')
        return
    # trying idx in [i, +inf)
    idx = 1
    while True:
        if not check_record_idx(record_folder, idx):
            print(f'Error: {record_folder}/{idx} not exists.')   
            break
        # 重采样到20Hz
        #   -------------
        # =============
        # |  |  |  |  |
        
        # | means sample video/mskb frame
        # <space> means sample gaps
        
        samp_tts, video_samp_idxs = get_sample_timestamps(record_folder, idx)
        # print(f'len(samp_tts): {len(samp_tts)}')
        # video_samp_idxs = set(video_samp_idxs)
        # print(f'len(video_samp_idxs): {len(video_samp_idxs)}')
        # 草，set之后变短了，说明有重复的

        do_sample(record_folder, idx, samp_tts, video_samp_idxs)

        # showing the video samp frames
        # video_path = os.path.join(record_folder, f'{idx}.mp4')
        # test_sample_video(video_path, video_samp_idxs)

        idx += 1


if __name__ == '__main__':
    # test_resize('./build/test/0.mp4')
    main(output_path='./build', task_name='test')    # test_resize('./build/test/1_alpha.mp4')
