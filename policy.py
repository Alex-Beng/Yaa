# copy from act shamelessly.
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.last_l2 = None
        
        # TODO: rm this hack
        self.kl_weight = args_override['kl_weight'] if 'kl_weight' in args_override else 10
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        # 应该只要norm到-0.5, 0.5就行了 here？
        # 因为游戏的图像分布不来自真实世界，imagenet的mean和std不适用
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        image = normalize(image)
        if actions is not None: # training time

            # mouse action & keyboard action
            mouse_actions = actions[:, :, -2:]
            keyboard_actions = actions[:, :, :-2]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            # action shape batch_size, chunk_size, action_dim
            mouse_a_hat = a_hat[:, :, -2:]
            keyboard_a_hat = a_hat[:, :, :-2]

            # mouse 使用 l1 loss，keyboard 使用 binary cross entropy
            mouse_l1 = F.l1_loss(mouse_a_hat, mouse_actions, reduction='none')
            # mouse_l1 = F.mse_loss(mouse_a_hat, mouse_actions, reduction='none')
            # mouse_l1 *= 100
            # 使用l2 loss，但是需要进行clip
            # mouse_l1 = F.mse_loss(mouse_a_hat, mouse_actions, reduction='none')
    
            keyboard_ce = F.binary_cross_entropy_with_logits(keyboard_a_hat, keyboard_actions, reduction='none')
            # 为什么action的范围不在0-1之间？
            # 因为傻逼norm不是0-1，而是 N(0, 1) 
            # min_action = keyboard_actions.min()
            # max_action = keyboard_actions.max()
            # print(f'Min action {min_action}, Max action {max_action}')
            # keyboard_ce = F.l1_loss(keyboard_a_hat, keyboard_actions, reduction='none')
            
            # mask掉pad的部分
            mouse_l1 = (mouse_l1 * ~is_pad.unsqueeze(-1)).mean()
            keyboard_ce = (keyboard_ce * ~is_pad.unsqueeze(-1)).mean()

            # if self.last_l2 is None:
            #     self.last_l2 = mouse_l1
            # else:
            #     if mouse_l1 > self.last_l2 * 1.5:
            #         print(f'clip mouse l1 {mouse_l1} to {self.last_l2}')
            #         mouse_l1 = self.last_l2 * 0.8


            # print(f'Mouse L1 {mouse_l1}, Keyboard CE {keyboard_ce}')
            keyboard_ce *= 10
            # mouse_l1 *= 0.8
            action_loss_sum = (mouse_l1 + keyboard_ce) / 2
            # action_loss_sum = keyboard_ce 
            
            loss_dict['action_loss_sum'] = action_loss_sum
            loss_dict['mouse_l1'] = mouse_l1
            loss_dict['keyboard_ce'] = keyboard_ce
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['action_loss_sum'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
    
    def load_can_load(self, ckpt_path):
        # TODO: copy from yap-train
        # 其实如果是网络结构全变了，strict=False应该就够用了
        # 只能对同名不同shape的参数做处理
        try:
            self.load_state_dict(torch.load(ckpt_path), strict=True)
            print(f'Loaded {ckpt_path}')
        except:
            state_dict = torch.load(ckpt_path)
            model_dict = self.state_dict()
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    model_dict[k] = v
                elif k in model_dict and v.shape != model_dict[k].shape:
                    print(f'Shape mismatch {k} {v.shape} {model_dict[k].shape}')
                else:
                    print(f'Key not found {k}')
            self.load_state_dict(model_dict, strict=False)

            print(f'Loaded {ckpt_path}')
        
# 这个MLP是没有 chunk_size 的纯血 BC
class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            # mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            # 使用类似的键盘bce + 鼠标l1 的loss
            mouse_actions = actions[:, -2:]
            keyboard_actions = actions[:, :-2]
            
            mouse_a_hat = a_hat[:, -2:]
            keyboard_a_hat = a_hat[:, :-2]
            
            mouse_l1 = F.l1_loss(mouse_a_hat, mouse_actions, reduction='none')
            keyboard_ce = F.binary_cross_entropy_with_logits(keyboard_a_hat, keyboard_actions, reduction='none')

            mouse_l1 = mouse_l1.sum()
            keyboard_ce = keyboard_ce.sum()

            action_loss = (mouse_l1 + keyboard_ce) / 2   
            # 先只学键盘，后面再学鼠标
            # action_loss = keyboard_ce     

            loss_dict['mouse_l1'] = mouse_l1
            loss_dict['keyboard_ce'] = keyboard_ce
            loss_dict['loss'] = action_loss

            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
    
    def load_can_load(self, ckpt_path):
        # TODO: copy from yap-train
        # 其实如果是网络结构全变了，strict=False应该就够用了
        # 只能对同名不同shape的参数做处理
        try:
            self.load_state_dict(torch.load(ckpt_path), strict=True)
            print(f'Loaded {ckpt_path}')
        except:
            state_dict = torch.load(ckpt_path)
            model_dict = self.state_dict()
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    model_dict[k] = v
                elif k in model_dict and v.shape != model_dict[k].shape:
                    print(f'Shape mismatch {k} {v.shape} {model_dict[k].shape}')
                else:
                    print(f'Key not found {k}')
            self.load_state_dict(model_dict, strict=False)

            print(f'Loaded {ckpt_path}')

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
