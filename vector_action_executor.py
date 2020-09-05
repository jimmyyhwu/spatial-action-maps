import time
import numpy as np
import vector_controller
import vector_utils as utils

class VectorActionExecutor:
    def __init__(self, robot_index):
        if robot_index is None:
            robot_name = utils.get_first_available_robot()
            if robot_name is None:
                print('No robots found')
            robot_index = utils.get_robot_indices()[robot_name]

        self.robot_index = robot_index
        self.controller = vector_controller.VectorController(self.robot_index)

        self.last_update_time = time.time()
        self.last_robot_position = None
        self.last_robot_heading = None
        self.stuck_count = 0
        self.unstuck_count = 0
        self.jittering = False

        self.target_end_effector_position = None

    def update_try_action_result(self, try_action_result):
        # Simulation results
        self.target_end_effector_position = try_action_result['target_end_effector_position']
        self.last_update_time = time.time()

    def step(self, poses):
        if self.target_end_effector_position is None:
            return

        robot_poses = poses['robots']
        if robot_poses is None or self.robot_index not in robot_poses:
            return

        robot_position = robot_poses[self.robot_index]['position']
        robot_heading = robot_poses[self.robot_index]['heading']
        info = {
            'last_robot_position': self.last_robot_position,
            'last_robot_heading': self.last_robot_heading,
            'robot_position': robot_position,
            'robot_heading': robot_heading,
            'target_end_effector_position': self.target_end_effector_position,
        }
        if self.jittering:
            info['robot_heading'] += np.random.uniform(-np.pi, np.pi)
        self.last_robot_position = robot_position
        self.last_robot_heading = robot_heading

        # Update the controller
        self.controller.step(info)

        # Handle robot getting stuck
        if self.controller.is_stuck():
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.unstuck_count += 1
        if self.stuck_count > 30:
            self.jittering = True
            self.unstuck_count = 0
        if self.jittering and self.unstuck_count > 5:
            self.jittering = False

    def is_action_completed(self, timeout=10):
        if any([self.controller.is_idle(), self.stuck_count > 10, time.time() - self.last_update_time > timeout]):
            return True
        return False

    def disconnect(self):
        self.controller.disconnect()
