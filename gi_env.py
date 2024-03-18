# GI sim & real env
# here sim is only step in data, WITHOUT any iteration
# so the 'sim' just for visualization & testing infering speed & etc.
# real is with GI iteration

# JUST for aligning with act 

from constants import TASK_CONFIG
from utils import load_data_test

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

class GIRealEnv:
    # TODO: implement GIRealEnv
    def __init__(self, config) -> None:
        raise NotImplementedError
        # image, action,
        # torch.Size([1, 2, 3, 480, 640]) torch.Size([1, 19])

    def reset(self):
        pass        
    def render(self):
        pass
    def step(self, action):
        pass

if __name__ == '__main__':
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
    