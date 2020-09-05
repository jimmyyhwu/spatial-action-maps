from pathlib import Path
import utils

def generate_experiment(experiment_name, template_experiment_name, modify_cfg_fn, output_dir, template_dir='config/experiments/base'):
    # Ensure output dir exists
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Read template config
    cfg = utils.read_config(Path(template_dir) / '{}.yml'.format(template_experiment_name))

    # Apply modifications
    cfg.experiment_name = experiment_name
    num_fields = len(cfg)
    modify_cfg_fn(cfg)
    assert num_fields == len(cfg), experiment_name  # New fields should not have been added

    # Save new config
    utils.write_config(cfg, output_dir / '{}.yml'.format(experiment_name))

def main():
    # Different obstacle configs
    obstacle_configs = ['small_empty', 'small_columns', 'large_columns', 'large_divider']
    def modify_cfg_obstacle_config(cfg, obstacle_config):
        cfg.obstacle_config = obstacle_config
        if obstacle_config.startswith('large'):
            cfg.num_cubes = 20
            cfg.room_width = 1.0
    for obstacle_config in obstacle_configs:
        generate_experiment(obstacle_config, 'small_empty', lambda cfg: modify_cfg_obstacle_config(cfg, obstacle_config), 'config/experiments/base', template_dir='config/templates')

    # Steering commands
    def modify_cfg_steering_commands(cfg):
        cfg.policy_type = 'steering_commands'
        cfg.use_steering_commands = True
        cfg.steering_commands_num_turns = 18
        cfg.fixed_step_size = 0.25
        cfg.use_position_channel = True
        cfg.num_input_channels += 2
    for obstacle_config in obstacle_configs:
        experiment_name = '-'.join([obstacle_config, 'steering_commands'])
        generate_experiment(experiment_name, obstacle_config, modify_cfg_steering_commands, 'config/experiments/steering-commands')

    # Fixed step size
    def modify_cfg_fixed_step_size(cfg):
        cfg.fixed_step_size = 0.25
    for obstacle_config in obstacle_configs:
        experiment_name = '-'.join([obstacle_config, 'fixed_step_size'])
        generate_experiment(experiment_name, obstacle_config, modify_cfg_fixed_step_size, 'config/experiments/fixed-step-size')

    # Shortest path ablations
    def modify_cfg_sp_ablation(cfg, ablation):
        if ablation == 'no_sp_movement':
            cfg.use_shortest_path_movement = False
        elif ablation == 'no_sp_from_agent':
            cfg.num_input_channels -= 1
            cfg.use_shortest_path_channel = False
        elif ablation == 'no_sp_to_receptacle':
            cfg.use_distance_to_receptacle_channel = True
            cfg.use_shortest_path_to_receptacle_channel = False
        elif ablation == 'no_sp_in_rewards':
            cfg.use_shortest_path_partial_rewards = False
        elif ablation == 'no_sp_components':
            cfg.use_shortest_path_movement = False
            cfg.num_input_channels -= 1
            cfg.use_shortest_path_channel = False
            cfg.use_distance_to_receptacle_channel = True
            cfg.use_shortest_path_to_receptacle_channel = False
            cfg.use_shortest_path_partial_rewards = False
    for obstacle_config in obstacle_configs:
        for ablation in ['no_sp_movement', 'no_sp_from_agent', 'no_sp_to_receptacle', 'no_sp_in_rewards', 'no_sp_components']:
            if obstacle_config == 'small_empty' and ablation in ['no_sp_in_rewards', 'no_sp_movement', 'no_sp_to_receptacle']:
                continue
            experiment_name = '-'.join([obstacle_config, ablation])
            generate_experiment(experiment_name, obstacle_config, lambda cfg: modify_cfg_sp_ablation(cfg, ablation), 'config/experiments/shortest-path-ablations')

    # Steering commands with no shortest path components
    def modify_cfg_steering_commands_no_sp_components(cfg):
        modify_cfg_steering_commands(cfg)
        modify_cfg_sp_ablation(cfg, 'no_sp_components')
    for obstacle_config in obstacle_configs:
        experiment_name = '-'.join([obstacle_config, 'steering_commands', 'no_sp_components'])
        generate_experiment(experiment_name, obstacle_config, modify_cfg_steering_commands_no_sp_components, 'config/experiments/steering-commands')

    # No partial rewards
    def modify_cfg_no_partial_rewards(cfg):
        cfg.partial_rewards_scale = 0
    for obstacle_config in obstacle_configs:
        experiment_name = '-'.join([obstacle_config, 'no_partial_rewards'])
        generate_experiment(experiment_name, obstacle_config, modify_cfg_no_partial_rewards, 'config/experiments/no-partial-rewards')

    # For development on local machine
    def modify_cfg_to_local(cfg):
        cfg.batch_size = 4
        cfg.replay_buffer_size = 1000
        cfg.learning_starts = 10
        cfg.inactivity_cutoff = 5
    generate_experiment('small_empty-local', 'small_empty', modify_cfg_to_local, 'config/local')

main()
