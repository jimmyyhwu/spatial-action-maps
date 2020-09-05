import anki_vector
import numpy as np
import vector_utils
from environment import ROBOT_RADIUS

MIN_TURN_SPEED = 10
MAX_TURN_SPEED = 150
MIN_MOVE_SPEED = 10
MAX_MOVE_SPEED = 150
#PROPORTIONAL_GAIN = 65
PROPORTIONAL_GAIN = 55
TURN_ERROR_THRESHOLD = np.radians(5)
MOVE_ERROR_THRESHOLD = np.radians(30)
TARGET_DIST_THRESHOLD = 0.010
SLOWING_DIST_THRESHOLD = 0.100

class VectorController:
    def __init__(self, robot_index):
        self.robot_name = vector_utils.get_robot_names()[robot_index]
        serial = vector_utils.get_robot_serials()[self.robot_name]
        #self.robot = anki_vector.AsyncRobot(serial=serial, default_logging=False)
        self.robot = anki_vector.AsyncRobot(serial=serial, default_logging=False, behavior_control_level=anki_vector.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY)
        self.robot.connect()
        print('Connected to {}'.format(self.robot_name))

        self._reset_motors()
        self.robot_state = 'idle'
        self.stuck = False

    def step(self, info):
        last_robot_position, last_robot_heading = info['last_robot_position'], info['last_robot_heading']
        robot_position, robot_heading = info['robot_position'], info['robot_heading']
        target_end_effector_position = info['target_end_effector_position']

        # Compute target heading
        x_movement = target_end_effector_position[0] - robot_position[0]
        y_movement = target_end_effector_position[1] - robot_position[1]
        dist = np.sqrt(x_movement**2 + y_movement**2) - ROBOT_RADIUS
        move_sign = np.sign(dist)
        dist = abs(dist)
        target_heading = np.arctan2(y_movement, x_movement)

        # Compute heading error
        heading_error = target_heading - robot_heading
        heading_error = np.mod(heading_error + np.pi, 2 * np.pi) - np.pi

        # Set wheel motors
        if self.robot_state == 'idle':
            if dist > TARGET_DIST_THRESHOLD or abs(heading_error) > TURN_ERROR_THRESHOLD:
                self.robot_state = 'turning'
            else:
                self.robot.motors.set_wheel_motors(0, 0)

        elif self.robot_state == 'turning':
            if abs(heading_error) < TURN_ERROR_THRESHOLD:
                self.robot_state = 'moving'
                self.robot.motors.set_wheel_motors(0, 0)
            else:
                signed_turn_speed = PROPORTIONAL_GAIN * heading_error
                signed_turn_speed = np.clip(signed_turn_speed, -MAX_TURN_SPEED, MAX_TURN_SPEED)
                signed_turn_speed = max(1, MIN_TURN_SPEED / abs(signed_turn_speed)) * signed_turn_speed
                self.robot.motors.set_wheel_motors(-signed_turn_speed, signed_turn_speed)

        elif self.robot_state == 'moving':
            if abs(heading_error) > MOVE_ERROR_THRESHOLD:  # Veered too far off path
                self.robot.motors.set_wheel_motors(0, 0)
                self.robot_state = 'turning'
            elif dist < TARGET_DIST_THRESHOLD:
                self.robot.motors.set_wheel_motors(0, 0)
                self.robot_state = 'idle'
            else:
                signed_turn_speed = PROPORTIONAL_GAIN * heading_error
                signed_turn_speed = np.clip(signed_turn_speed, -MAX_TURN_SPEED, MAX_TURN_SPEED)
                move_speed = min(MAX_MOVE_SPEED, max(MIN_MOVE_SPEED, 1000 * dist)) if dist < SLOWING_DIST_THRESHOLD else MAX_MOVE_SPEED
                self.robot.motors.set_wheel_motors(move_sign * move_speed - signed_turn_speed, move_sign * move_speed + signed_turn_speed)

        # Check if stuck
        self.stuck = False
        if last_robot_position is not None and self.robot_state in ('moving', 'turning'):
            position_diff = np.linalg.norm(np.asarray(robot_position) - np.asarray(last_robot_position))
            heading_diff = robot_heading - last_robot_heading
            heading_diff = np.mod(heading_diff + np.pi, 2 * np.pi) - np.pi
            if position_diff < 0.003 and heading_diff < np.radians(3):
                self.stuck = True

    def is_idle(self):
        return self.robot_state == 'idle'

    def is_stuck(self):
        return self.stuck

    def _reset_motors(self):
        self.robot.behavior.set_head_angle(anki_vector.util.degrees(0))
        self.robot.motors.set_wheel_motors(0, 0)

    def disconnect(self):
        self._reset_motors()
        self.robot.disconnect()
        print('Disconnected from {}'.format(self.robot_name))
