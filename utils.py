from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from matplotlib import cm
from munch import Munch
from prompt_toolkit.shortcuts import radiolist_dialog

import environment
import policies


################################################################################
# Experiment management

def read_config(config_path):
    with open(config_path, 'r') as f:
        cfg = Munch.fromYAML(f)
    return cfg

def write_config(cfg, config_path):
    with open(config_path, 'w') as f:
        f.write(cfg.toYAML())

def setup_run(config_path):
    cfg = read_config(config_path)

    if cfg.log_dir is not None:
        # Run has already been set up
        return config_path

    # Set up run_name, log_dir, and checkpoint_dir
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S%f')
    cfg.run_name = '{}-{}'.format(timestamp, cfg.experiment_name)
    log_dir = Path(cfg.logs_dir) / cfg.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(log_dir)
    cfg.checkpoint_dir = str(Path(cfg.checkpoints_dir) / cfg.run_name)

    # Save config file for the new run
    config_path = log_dir / 'config.yml'
    write_config(cfg, config_path)

    return config_path

def select_run():
    logs_dir = Path('logs')
    log_dirs = [path for path in sorted(logs_dir.iterdir()) if path.is_dir()]
    if len(log_dirs) == 0:
        return None

    grouped_config_paths = {}
    for log_dir in log_dirs:
        parts = log_dir.name.split('-')
        experiment_name = '-'.join(parts[1:])
        if experiment_name not in grouped_config_paths:
            grouped_config_paths[experiment_name] = []
        grouped_config_paths[experiment_name].append(log_dir / 'config.yml')

    if len(grouped_config_paths) > 1:
        config_paths = radiolist_dialog(
            values=[(value, key) for key, value in sorted(grouped_config_paths.items())],
            text='Please select an experiment:').run()
        if config_paths is None:
            return None
    else:
        config_paths = next(iter(grouped_config_paths.values()))

    selected_config_path = radiolist_dialog(
        values=[(path, path.parent.name) for path in config_paths],
        text='Please select a run:').run()
    if selected_config_path is None:
        return None

    return selected_config_path

def get_env_from_cfg(cfg, real_env=False, **kwargs):
    kwarg_list = [
        'room_length', 'room_width', 'num_cubes', 'obstacle_config',
        'use_distance_to_receptacle_channel', 'distance_to_receptacle_channel_scale',
        'use_shortest_path_to_receptacle_channel', 'use_shortest_path_channel', 'shortest_path_channel_scale',
        'use_position_channel', 'position_channel_scale',
        'partial_rewards_scale', 'use_shortest_path_partial_rewards', 'collision_penalty', 'nonmovement_penalty',
        'use_shortest_path_movement', 'fixed_step_size', 'use_steering_commands', 'steering_commands_num_turns',
        'ministep_size', 'inactivity_cutoff', 'random_seed',
    ]
    original_kwargs = {}
    for kwarg_name in kwarg_list:
        original_kwargs[kwarg_name] = cfg[kwarg_name]
    original_kwargs.update(kwargs)
    if real_env:
        return environment.RealEnvironment(**original_kwargs)
    return environment.Environment(**original_kwargs)

def get_policy_from_cfg(cfg, action_space, **kwargs):
    if cfg.policy_type == 'steering_commands':
        return policies.SteeringCommandsPolicy(cfg, action_space, **kwargs)
    if cfg.policy_type == 'dense_action_space':
        return policies.DenseActionSpacePolicy(cfg, action_space, **kwargs)
    raise Exception

################################################################################
# Visualization

JET = np.array([list(cm.jet(i)[:3]) for i in range(256)])

def scale_min_max(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def to_uint8_image(image):
    return np.round(255.0 * image).astype(np.uint8)

def get_state_visualization(state):
    if state.shape[2] == 2:
        return np.stack([state[:, :, 1], state[:, :, 0], state[:, :, 0]], axis=2)  # (robot_state_channel, overhead_map, overhead_map)
    return np.stack([state[:, :, 1], state[:, :, 0], state[:, :, -1]], axis=2)  # (robot_state_channel, overhead_map, distance_channel)

def get_overhead_image(state):
    return np.stack([state[:, :, 0], state[:, :, 0], state[:, :, 0]], axis=2)

def get_reward_img(reward, ministeps, img_height, state_width):
    reward_img = np.zeros((img_height, state_width, 3), dtype=np.float32)
    text = '{:.02f}/{:+.02f}'.format(ministeps, reward)
    cv2.putText(reward_img, text, (state_width - 5 * len(text), 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (1, 1, 1))
    return reward_img

def get_output_visualization(overhead_image, output):
    alpha = 0.5
    return (1 - alpha) * overhead_image + alpha * JET[output, :]

def get_state_and_output_visualization(state, output):
    output = to_uint8_image(scale_min_max(output))
    return np.concatenate((get_state_visualization(state), get_output_visualization(get_overhead_image(state), output)), axis=1)
