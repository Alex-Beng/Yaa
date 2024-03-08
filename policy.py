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
        
        # TODO: rm this hack
        self.kl_weight = args_override['kl_weight'] if 'kl_weight' in args_override else 10
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        # norm from ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # mouse action & keyboard action
            mouse_actions = actions[:, :, -3:]
            keyboard_actions = actions[:, :, :-3]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            # action shape batch_size, chunk_size, action_dim
            mouse_a_hat = a_hat[:, :, -3:]
            keyboard_a_hat = a_hat[:, :, :-3]

            # mouse 使用 l1 loss，keyboard 使用 binary cross entropy
            mouse_l1 = F.l1_loss(mouse_actions, mouse_a_hat, reduction='none')
            keyboard_ce = F.binary_cross_entropy_with_logits(keyboard_a_hat, keyboard_actions, reduction='none')
            # keyboard_ce = F.l1_loss(keyboard_a_hat, keyboard_actions, reduction='none')
            
            # mask掉pad的部分
            mouse_l1 = (mouse_l1 * ~is_pad.unsqueeze(-1)).mean()
            keyboard_ce = (keyboard_ce * ~is_pad.unsqueeze(-1)).mean()
            print(f'Mouse L1 {mouse_l1}, Keyboard CE {keyboard_ce}')
            keyboard_ce *= 1.2
            mouse_l1 *= 0.8
            l1 = (mouse_l1 + keyboard_ce) / 2
            l1 = keyboard_ce
            
            if False:
                all_l1 = F.l1_loss(actions, a_hat, reduction='none')
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                # print(f'L1 {l1}')
                
            
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
    
    def load_can_loadd(self, ckpt_path):
        # TODO: copy from yap-train
        # 其实如果是网络结构全变了，strict=False应该就够用了
        # yap train 是只改最后的fc层，所以可以通过idx进一步填充
        # 这里大概是chunk size和action dim会变？
        raise NotImplementedError
        

# IN Yaa, useless
class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

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
