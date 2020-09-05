import argparse
from multiprocessing.connection import Client

import cv2

import environment
import utils
import vector_action_executor


class ClickAgent:
    def __init__(self, conn, env, action_executor):
        self.conn = conn
        self.env = env
        self.action_executor = action_executor
        self.window_name = 'window'
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
        #cv2.resizeWindow(self.window_name, 960, 960)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.selected_action = None

    def mouse_callback(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_action = (y, x)

    def run(self):
        try:
            while True:
                # Get new pose estimates
                poses = None
                while self.conn.poll():  # Discard all but the latest data
                    poses = self.conn.recv()
                if poses is None:
                    continue

                # Update poses in the simulation
                self.env.update_poses(poses)

                # Visualize state representation
                state = self.env.get_state()
                cv2.imshow(self.window_name, utils.get_state_visualization(state)[:, :, ::-1])
                cv2.waitKey(1)

                if self.selected_action is not None:
                    action = self.selected_action[0] * self.env.get_state_width() + self.selected_action[1]
                    self.selected_action = None

                    # Run selected action through simulation
                    try_action_result = self.env.try_action(action)

                    # Update action executor
                    self.action_executor.update_try_action_result(try_action_result)

                # Run action executor
                self.action_executor.step(poses)
        finally:
            self.action_executor.disconnect()

def main(args):
    # Connect to aruco server for pose estimates
    try:
        conn = Client(('localhost', 6000), authkey=b'secret password')
    except ConnectionRefusedError:
        print('Could not connect to aruco server for pose estimates')
        return

    # Create action executor for the physical robot
    action_executor = vector_action_executor.VectorActionExecutor(args.robot_index)

    # Create pybullet environment to mirror real-world
    kwargs = {'num_cubes': args.num_cubes, 'use_gui': True}
    cube_indices = list(range(args.num_cubes))
    env = environment.RealEnvironment(robot_index=action_executor.robot_index, cube_indices=cube_indices, **kwargs)
    env.reset()

    # Run the agent
    agent = ClickAgent(conn, env, action_executor)
    agent.run()

parser = argparse.ArgumentParser()
parser.add_argument('--robot-index', type=int)
parser.add_argument('--num-cubes', type=int, default=10)
main(parser.parse_args())
