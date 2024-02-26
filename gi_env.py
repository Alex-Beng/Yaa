# GI sim & real env
# here sim is only step in data, WITHOUT any iteration
# so the 'sim' just for visualization & testing infering speed & etc.
# real is with GI iteration

# JUST for aligning with act 

from constants import TASK_CONFIG
from utils import load_data_test

class GIDataEnv:
    def __init__(self, config) -> None:
        self.task_name = config['task_name']
        task_config = TASK_CONFIG[self.task_name]
        # get task parameters
        self.dataset_dir = task_config['dataset_dir']
        self.num_episodes = task_config['num_episodes']
        self.camera_names = task_config['camera_names']

        # 读取数据集
        self.dataloader, self.norm_stats = load_data_test(self.dataset_dir, self.num_episodes, self.camera_names, 1)
        print(len(self.dataloader))

    def reset(self):
        
        pass

    def render(self):
        pass

if __name__ == '__main__':
    config = {
        'task_name': 'nazuchi_beach_friendship',
    }

    env = GIDataEnv(config)
    