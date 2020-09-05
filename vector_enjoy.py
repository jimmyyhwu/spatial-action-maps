import argparse
from multiprocessing.connection import Client

import cv2

import utils
import vector_action_executor


def main(args):
    # Connect to aruco server for pose estimates
    try:
        conn = Client(('localhost', 6000), authkey=b'secret password')
    except ConnectionRefusedError:
        print('Could not connect to aruco server for pose estimates')
        return

    # Create action executor for the physical robot
    action_executor = vector_action_executor.VectorActionExecutor(args.robot_index)

    # Create env
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        print('Please provide a config path')
        return
    cfg = utils.read_config(config_path)
    kwargs = {'num_cubes': args.num_cubes}
    if args.debug:
        kwargs['use_gui'] = True
    cube_indices = list(range(args.num_cubes))
    env = utils.get_env_from_cfg(cfg, physical_env=True, robot_index=action_executor.robot_index, cube_indices=cube_indices, **kwargs)
    env.reset()

    # Create policy
    policy = utils.get_policy_from_cfg(cfg, env.get_action_space())

    # Debug visualization
    if args.debug:
        cv2.namedWindow('out', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('out', 960, 480)

    try:
        while True:
            # Get new pose estimates
            poses = None
            while conn.poll():  # ensure up-to-date data
                poses = conn.recv()
            if poses is None:
                continue

            # Update poses in the simulation
            env.update_poses(poses)

            # Get new action
            state = env.get_state()
            if action_executor.is_action_completed() and args.debug:
                action, info = policy.step(state, debug=True)
                # Visualize
                assert not cfg.use_steering_commands
                output = info['output'].cpu().numpy()
                cv2.imshow('out', utils.get_state_and_output_visualization(state, output)[:, :, ::-1])
                cv2.waitKey(1)
            else:
                action, _ = policy.step(state)

            # Run selected action through simulation
            try_action_result = env.try_action(action)

            if action_executor.is_action_completed():
                # Update action executor
                action_executor.update_try_action_result(try_action_result)

            # Run action executor
            action_executor.step(poses)

    finally:
        action_executor.disconnect()

parser = argparse.ArgumentParser()
parser.add_argument('--config-path')
parser.add_argument('--robot-index', type=int)
parser.add_argument('--num-cubes', type=int, default=10)
parser.add_argument('--debug', action='store_true')
parser.set_defaults(debug=False)
main(parser.parse_args())
