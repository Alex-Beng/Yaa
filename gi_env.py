# GI sim & real env
# here sim is only step in data, WITHOUT any iteration
# so the 'sim' just for visualization & testing infering speed & etc.
# real is with GI iteration

# JUST for aligning with act 

from threading import Thread, Lock
from queue import Queue # copilot said it is thread safe

# TODO: 做一套send/post message的接口，或者从GIA抄过来
# 现在先用 pynput + pydirectinput 凑活用吧
from pynput.keyboard import Controller
import pydirectinput
pydirectinput.PAUSE = 0

from constants import TASK_CONFIG, STATE_DIM, SN_idx2key
from utils import load_data_test, capture

# windows only, translate from yap
from win32gui import FindWindow

def find_window_local():
    class_name = "UnityWndClass"
    window_name = "原神"
    hwnd = FindWindow(class_name, window_name)
    return hwnd

# srds, bitblt + hwnd 的截不到云原神
def find_window_cloud():
    window_name = "云·原神"
    hwnd = FindWindow(None, window_name)
    return hwnd

# copy from PySmartCubeGenshin
keyboard = Controller() # use like keyboard.press('a'); keyboard.release('a')
def move_mouse(x, y):
    pydirectinput.moveTo(x, y, relative=True)


import IPython
e = IPython.embed

class GIDataEnv:
    def __init__(self, config) -> None:
        self.ckpt_dir = config['ckpt_dir']
        self.task_name = config['task_name']

        task_config = TASK_CONFIG[self.task_name]
        # get task parameters
        self.dataset_dir = task_config['dataset_dir']
        self.num_episodes = task_config['num_episodes']
        self.camera_names = task_config['camera_names']
        print(self.num_episodes, self.camera_names)
        
        # count to num_episodes
        # if end of episode, just return flag for caller to reset
        self.episode_id = 0
        # 
        self.step_id = 0
        
        self.dataset, self.norm_stats = load_data_test(self.dataset_dir, self.ckpt_dir, self.episode_id, self.camera_names)
        # print(len(self.dataloader))

        self.preprocess = lambda action: (action - self.norm_stats['action_mean']) / self.norm_stats['action_std']
        self.postprocess = lambda action: action * self.norm_stats['action_std'] + self.norm_stats['action_mean']
        
        # image, action,
        # torch.Size([1, 2, 3, 480, 640]) torch.Size([1, 19])

    def reset(self):
        self.episode_id += 1
        # 如果已经到了最后一个episode
        if self.episode_id >= self.num_episodes:
            return False
        self.dataset, self.norm_stats = load_data_test(self.dataset_dir, self.ckpt_dir, self.episode_id, self.camera_names)
        # print(len(self.dataset))
        print(f'episode {self.episode_id} reseted')

        self.step_id = 0
        return True
        
    def render(self):
        # 实际是取出当前step的图像
        image_tensors = self.dataset[self.step_id][0]
        image_dict = {}
        for i in range(len(self.camera_names)):
            # make it numpy in shape (480, 640, 3)
            curr_image = image_tensors[i].numpy()
            image_dict[self.camera_names[i]] = curr_image
        return image_dict
        
    def step(self, action):
        # action is a tensor
        self.step_id += 1

        if self.step_id >= len(self.dataset):
            return False
        else:
            return True
    
    def observation(self):
        action = self.dataset[self.step_id][1]
        image_dict = self.render()
        return image_dict, action

latest_capture = None
capture_lock = Lock()

class CaptureThread(Thread):
    def __init__(self, hwnd) -> None:
        super().__init__()
        self.hwnd = hwnd
        self.running = True

    def run(self):
        global latest_capture
        while self.running:
            with capture_lock:
                latest_capture = capture(self.hwnd)
    def stop(self):
        self.running = False

class GIRealEnv:
    # 0. 获取窗口的hwnd，用于截屏和send message
    # 1. 一个持续截屏的线程
    # 2. reset重置hwnd和协程
    # 3. render从队列中取出当前的图像
    # 4. step发送action到窗口
    
    def __init__(self, config) -> None:
        # get hwnd
        self.hwnd = find_window_local()
        if self.hwnd == 0:
            self.hwnd = find_window_cloud()
        if self.hwnd == 0:
            raise ValueError('Cannot find window')

        # SetForegroundWindow
        from win32gui import SetForegroundWindow
        SetForegroundWindow(self.hwnd)
        
        self.capture_thread = CaptureThread(self.hwnd)
        self.capture_thread.start()

        # 0 for up, 1 for down
        self.keyboard_state = [0]*(STATE_DIM-2)

        # press threshold
        self.press_threshold = config['kb_threshold'] if 'kb_threshold' in config else 0.5
        assert 0 < self.press_threshold < 1
        
    def reset(self):
        # reset the keyboard state
        self.keyboard_state = [0]*(STATE_DIM-2)
    
    def render(self):
        global latest_capture
        with capture_lock:
            return latest_capture
    
    def step(self, action):
        # sigmoid should be done in caller
        for a_id in range(STATE_DIM-2):
            if action[a_id] > self.press_threshold:
                if self.keyboard_state[a_id] == 0:
                    print(f'press {SN_idx2key[a_id]}', end=' ')
                    keyboard.press(SN_idx2key[a_id].lower())
                    self.keyboard_state[a_id] = 1
            elif action[a_id] < 1 - self.press_threshold:
                if self.keyboard_state[a_id] == 1:
                    print(f'release {SN_idx2key[a_id]}', end=' ')
                    keyboard.release(SN_idx2key[a_id].lower())
                    self.keyboard_state[a_id] = 0
        # for mouse, input dx, dyww
        dx, dy = action[-2], action[-1]
        print(f'move mouse {dx}, {dy}')
        move_mouse(dx, dy)
        return True

    def observation(self):
        return self.render(), None

if __name__ == '__main__':
    keyboard.press('w')
    keyboard.release('w')

    import cv2
    ret = find_window_local()
    print(ret)
    ret = find_window_cloud()
    print(ret)
    
    env = GIRealEnv({'kb_threshold': 0.5})

    # for i in range(100):
    #     img = env.render()
    #     # print(img.shape)
    #     cv2.imshow('img', img)
    #     cv2.waitKey(1)
    #     # if not env.step(None):
    #     #     env.reset()
    from constants import SN
    from time import sleep
    action = [0]*STATE_DIM
    action[SN['W']] = 1
    env.step(action)
    sleep(10)
    action[SN['W']] = 0
    env.step(action)
    sleep(2)
    action[SN['Mdx']] = 100
    env.step(action)
    sleep(2)
    action[SN['Mdx']] = -100
    env.step(action)
    sleep(2)

    env.capture_thread.stop()
    del env.capture_thread
    exit()
    config = {
        'task_name': 'nazuchi_beach_friendship',
        'ckpt_dir': './models',
    }

    env = GIDataEnv(config)
    # e()
    for i in range(1000):
        env.render()
        if not env.step(None):
            env.reset()
    