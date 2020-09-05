import argparse
import utils

def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        print('Please provide a config path')
        return
    cfg = utils.read_config(config_path)
    env = utils.get_env_from_cfg(cfg, use_gui=True)
    policy = utils.get_policy_from_cfg(cfg, env.get_action_space())
    state = env.reset()
    while True:
        action, _ = policy.step(state)
        state, _, done, _ = env.step(action)
        if done:
            state = env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    main(parser.parse_args())
