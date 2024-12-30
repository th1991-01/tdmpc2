import os
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"

from go2_env import Go2Env, get_cfgs
import genesis as gs

import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
    assert torch.cuda.is_available()
    
    gs.init(logging_level="warning")
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env = Go2Env(
        num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg#, show_viewer=True
    )

    print("cfg",cfg)
    assert cfg.steps > 0, 'Must train for at least 1 step.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
    cfg.obs_shape = {'state': (obs_cfg["num_obs"],)}
    cfg.action_dim = env_cfg["num_actions"]
    cfg.episode_length = env.max_episode_length

    print("cfg.multitask",cfg.multitask)
    trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer

    trainer = trainer_cls(
        cfg=cfg,
        #env=make_env(cfg),
        env=env,
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    trainer.train()
train()
