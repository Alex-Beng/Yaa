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
- qpos                  (14,)         'float64'
- qvel                  (14,)         'float64'

action                  (14,)         'float64'
"""

def test_resize(video_path: str):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('frame', frame)
        # 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def main(record_root: str, episode_idx: int):
    


if __name__ == '__main__':
    test_resize('./build/test/1_alpha.mp4')