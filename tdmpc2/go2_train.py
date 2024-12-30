from go2_env import Go2Env, get_cfgs

import genesis as gs

import torch


def main():

    gs.init(logging_level="warning")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    env = Go2Env(
        num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True
    )

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = torch.zeros(1,env_cfg["num_actions"], device="cuda:0")
            obs, _, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
