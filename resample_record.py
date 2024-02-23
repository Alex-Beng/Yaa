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

import cv2

from constants import DT

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
        
    return sample_timestamps, video_sample_idx


def do_sample(record_folder, idx, tss, video_samp_idxs):
    
    # for video, just sample the nearest frame
    # for mskb, accumulate the events in the gap
    sampled_video_frames = []
    sampled_mskb_events = []

    # sample the frames first
    print(f"Sampling video {idx}...")
    rgb_video_path = os.path.join(record_folder, f'{idx}.mp4')
    alpha_video_path = os.path.join(record_folder, f'{idx}_alpha.mp4')
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    alpha_cap = cv2.VideoCapture(alpha_video_path)
    frame_cnt = 0
    debug = True
    if debug:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow('alpha', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = rgb_cap.read()
        ret_alpha, alpha_frame = alpha_cap.read()
        if not ret or not ret_alpha:
            break
        if frame_cnt in video_samp_idxs:
            if debug:
                cv2.imshow('frame', frame)
                cv2.imshow('alpha', alpha_frame)
            frame = cv2.resize(frame, (640, 480))
            alpha_frame = cv2.resize(alpha_frame, (640, 480))
            sampled_video_frames.append((
                frame, alpha_frame
            ))

        if debug and cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_cnt += 1
    
    # then sample mskb 
    print(f"Sampling mskb {idx}...")
    # 首先抛弃掉tts[0]时刻之前的msbk事件
    t_mskb_idx = 0
    mskb_jsonl_path  = os.path.join(record_folder, f'{idx}_mskb.jsonl')
    all_mskb_events = list(read_jsonl(mskb_jsonl_path))
    while all_mskb_events[t_mskb_idx]['timestamp'] < tss[0]:
        t_mskb_idx += 1
    # 然后开始采样
    for ts in tss:
        
        
    
    
    
    pass

def main(output_path: str, task_name: str):
    # 自适应查找各个录制。
    record_folder = os.path.join(output_path, task_name)
    if not os.path.exists(record_folder):
        print(f'Error: {record_folder} not exists.')
        return
    # trying idx in [i, +inf)
    idx = 0
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
        video_samp_idxs = set(video_samp_idxs)

        do_sample(record_folder, idx, samp_tts, video_samp_idxs)

        # showing the video samp frames
        # video_path = os.path.join(record_folder, f'{idx}.mp4')
        # test_sample_video(video_path, video_samp_idxs)

        idx += 1


if __name__ == '__main__':
    # test_resize('./build/test/0.mp4')
    main(output_path='./build', task_name='test')    # test_resize('./build/test/1_alpha.mp4')
