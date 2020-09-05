# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import argparse
import random
import sys
import time
from collections import namedtuple
from pathlib import Path

# Prevent numpy from using up all cpu
import os
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position

import torch
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils


torch.backends.cudnn.benchmark = True

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'ministeps', 'next_state'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)

def train(cfg, policy_net, target_net, optimizer, batch, transform_func):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_batch = torch.cat([transform_func(s) for s in batch.state]).to(device)  # (32, 3, 96, 96)
    action_batch = torch.tensor(batch.action, dtype=torch.long).to(device)  # (32,)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)  # (32,)
    ministeps_batch = torch.tensor(batch.ministeps, dtype=torch.float32).to(device)  # (32,)
    non_final_next_states = torch.cat([transform_func(s) for s in batch.next_state if s is not None]).to(device, non_blocking=True)  # (?32, 3, 96, 96)

    output = policy_net(state_batch)  # (32, 2, 96, 96)
    state_action_values = output.view(cfg.batch_size, -1).gather(1, action_batch.unsqueeze(1)).squeeze(1)  # (32,)
    next_state_values = torch.zeros(cfg.batch_size, dtype=torch.float32, device=device)  # (32,)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=device)  # (32,)

    if cfg.use_double_dqn:
        with torch.no_grad():
            best_action = policy_net(non_final_next_states).view(non_final_next_states.size(0), -1).max(1)[1].view(non_final_next_states.size(0), 1)  # (32?, 1)
            next_state_values[non_final_mask] = target_net(non_final_next_states).view(non_final_next_states.size(0), -1).gather(1, best_action).view(-1)  # (32?,)
    else:
        next_state_values[non_final_mask] = target_net(non_final_next_states).view(non_final_next_states.size(0), -1).max(1)[0].detach()  # (32,)

    expected_state_action_values = (reward_batch + torch.pow(cfg.discount_factor, ministeps_batch) * next_state_values)  # (32,)
    td_error = torch.abs(state_action_values - expected_state_action_values).detach()  # (32,)
    loss = smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    if cfg.grad_norm_clipping is not None:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.grad_norm_clipping)
    optimizer.step()

    train_info = {}
    train_info['q_value_min'] = output.min().item()
    train_info['q_value_max'] = output.max().item()
    train_info['td_error'] = td_error.mean()
    train_info['loss'] = loss

    return train_info

def main(cfg):
    # Set up logging and checkpointing
    log_dir = Path(cfg.log_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    print('log_dir: {}'.format(log_dir))
    print('checkpoint_dir: {}'.format(checkpoint_dir))

    # Create environment
    kwargs = {}
    if sys.platform == 'darwin':
        kwargs['use_gui'] = True
    env = utils.get_env_from_cfg(cfg, **kwargs)

    # Policy
    policy = utils.get_policy_from_cfg(cfg, env.get_action_space(), train=True)

    # Optimizer
    optimizer = optim.SGD(policy.policy_net.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)

    # Replay buffer
    replay_buffer = ReplayBuffer(cfg.replay_buffer_size)

    # Resume if applicable
    start_timestep = 0
    episode = 0
    if cfg.checkpoint_path is not None:
        checkpoint = torch.load(cfg.checkpoint_path)
        start_timestep = checkpoint['timestep']
        episode = checkpoint['episode']
        optimizer.load_state_dict(checkpoint['optimizer'])
        replay_buffer = checkpoint['replay_buffer']
        print("=> loaded checkpoint '{}' (timestep {})".format(cfg.checkpoint_path, start_timestep))

    # Target net
    target_net = policy.build_network()
    target_net.load_state_dict(policy.policy_net.state_dict())
    target_net.eval()

    # Logging
    train_summary_writer = SummaryWriter(log_dir=str(log_dir / 'train'))
    visualization_summary_writer = SummaryWriter(log_dir=str(log_dir / 'visualization'))
    meters = Meters()

    state = env.reset()
    total_timesteps_with_warm_up = cfg.learning_starts + cfg.total_timesteps
    for timestep in tqdm(range(start_timestep, total_timesteps_with_warm_up),
                         initial=start_timestep, total=total_timesteps_with_warm_up, file=sys.stdout):

        start_time = time.time()

        # Select an action
        if cfg.exploration_timesteps > 0:
            exploration_eps = 1 - min(max(timestep - cfg.learning_starts, 0) / cfg.exploration_timesteps, 1) * (1 - cfg.final_exploration)
        else:
            exploration_eps = cfg.final_exploration
        action, _ = policy.step(state, exploration_eps=exploration_eps)

        # Step the simulation
        next_state, reward, done, info = env.step(action)
        ministeps = info['ministeps']

        # Store in buffer
        replay_buffer.push(state, action, reward, ministeps, next_state)
        state = next_state

        # Reset if episode ended
        if done:
            state = env.reset()
            episode += 1

        # Train network
        if timestep >= cfg.learning_starts:
            batch = replay_buffer.sample(cfg.batch_size)
            train_info = train(cfg, policy.policy_net, target_net, optimizer, batch, policy.apply_transform)

        # Update target network
        if (timestep + 1) % cfg.target_update_freq == 0:
            target_net.load_state_dict(policy.policy_net.state_dict())

        step_time = time.time() - start_time

        ################################################################################
        # Logging

        # Meters
        meters.update('step_time', step_time)
        if timestep >= cfg.learning_starts:
            for name, val in train_info.items():
                meters.update(name, val)

        if done:
            for name in meters.get_names():
                train_summary_writer.add_scalar(name, meters.avg(name), timestep + 1)
            eta_seconds = meters.avg('step_time') * (total_timesteps_with_warm_up - timestep)
            meters.reset()

            train_summary_writer.add_scalar('episodes', episode, timestep + 1)
            train_summary_writer.add_scalar('eta_hours', eta_seconds / 3600, timestep + 1)

            for name in ['cumulative_cubes', 'cumulative_distance', 'cumulative_reward']:
                train_summary_writer.add_scalar(name, info[name], timestep + 1)

            # Visualize Q-network outputs
            if timestep >= cfg.learning_starts and not cfg.use_steering_commands:
                random_state = random.choice(replay_buffer.buffer).state
                _, info = policy.step(random_state, debug=True)
                output = info['output'].cpu().numpy()
                visualization = utils.get_state_and_output_visualization(random_state, output).transpose((2, 0, 1))
                visualization_summary_writer.add_image('output', visualization, timestep + 1)

        ################################################################################
        # Checkpointing

        if (timestep + 1) % cfg.checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warm_up:
            # Save model
            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model_name = 'model_{:08d}.pth.tar'.format(timestep + 1)
            torch.save({
                'timestep': timestep + 1,
                'state_dict': policy.policy_net.state_dict(),
            }, str(checkpoint_dir / model_name))

            # Save checkpoint
            checkpoint_name = 'checkpoint_{:08d}.pth.tar'.format(timestep + 1)
            torch.save({
                'timestep': timestep + 1,
                'episode': episode,
                'optimizer': optimizer.state_dict(),
                'replay_buffer': replay_buffer,
            }, str(checkpoint_dir / checkpoint_name))

            # Save updated config file
            cfg.model_path = str(checkpoint_dir / model_name)
            cfg.checkpoint_path = str(checkpoint_dir / checkpoint_name)
            utils.write_config(cfg, log_dir / 'config.yml')

            # Remove old checkpoint
            old_checkpoint_path = checkpoint_dir / 'checkpoint_{:08d}.pth.tar'.format((timestep + 1) - cfg.checkpoint_freq)
            if old_checkpoint_path.exists():
                old_checkpoint_path.unlink()

    env.close()

    # Create file to indicate training completed
    (log_dir / 'success').touch()

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Meters:
    def __init__(self):
        self.meters = {}

    def get_names(self):
        return self.meters.keys()

    def reset(self):
        for _, meter in self.meters.items():
            meter.reset()

    def update(self, name, val):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(val)

    def avg(self, name):
        return self.meters[name].avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    config_path = parser.parse_args().config_path
    config_path = utils.setup_run(config_path)
    main(utils.read_config(config_path))
