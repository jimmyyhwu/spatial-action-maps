# Prevent numpy from using up all cpu
import os
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position
import argparse
from pathlib import Path
import numpy as np
import utils

def _run_eval(cfg, num_episodes=20):
    env = utils.get_env_from_cfg(cfg, random_seed=0)
    policy = utils.get_policy_from_cfg(cfg, env.get_action_space(), random_seed=0)
    data = [[] for _ in range(num_episodes)]
    episode_count = 0
    state = env.reset()
    while True:
        action, _ = policy.step(state)
        state, _, done, info = env.step(action)
        data[episode_count].append({'distance': info['cumulative_distance'], 'cubes': info['cumulative_cubes']})
        if done:
            state = env.reset()
            episode_count += 1
            print('Completed {}/{} episodes'.format(episode_count, num_episodes))
            if episode_count >= num_episodes:
                break
    return data

def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        print('Please provide a config path')
        return
    cfg = utils.read_config(config_path)

    eval_dir = Path(cfg.logs_dir).parent / 'eval'
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)

    eval_path = eval_dir / '{}.npy'.format(cfg.run_name)
    data = _run_eval(cfg)
    np.save(eval_path, data)
    print(eval_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    main(parser.parse_args())
