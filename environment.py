import pprint
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.ndimage import rotate as rotate_image
from scipy.ndimage.morphology import distance_transform_edt
from shapely.geometry import box
from shapely.ops import unary_union
from skimage.draw import line
from skimage.measure import approximate_polygon
from skimage.morphology import binary_dilation
from skimage.morphology.selem import disk

import spfa


# Room, walls, and objects
GLOBAL_SCALING = 0.1
ROBOT_HALF_WIDTH = 0.03
ROBOT_RADIUS = (ROBOT_HALF_WIDTH**2 + 0.0565**2)**0.5
ROBOT_HEIGHT = 0.07
CUBE_WIDTH = 0.044
CUBE_MASS = 0.01
CUBE_COLOR = (237.0 / 255, 201.0 / 255, 72.0 / 255, 1)
WALL_HEIGHT = 0.1
WALL_THICKNESS = 1.4
RECEPTACLE_WIDTH = 0.15
RECEPTACLE_COLOR = (1, 87.0 / 255, 89.0 / 255, 1)
ROUNDED_CORNER_WIDTH = 0.1006834873
OBSTACLE_COLOR = (0.9, 0.9, 0.9, 1)
DEBUG_LINE_COLOR = (78.0 / 255, 121.0 / 255, 167.0 / 255)

# Movement
MOVE_STEP_SIZE = 0.005  # 5 mm
TURN_STEP_SIZE = np.radians(15)  # 15 degrees
MOVEMENT_MAX_FORCE = 10
NOT_MOVING_THRESHOLD = 0.0005  # 0.5 mm
NOT_TURNING_THRESHOLD = np.radians(1)  # 1 degree
NONMOVEMENT_DIST_THRESHOLD = 0.005
NONMOVEMENT_TURN_THRESHOLD = np.radians(1)
STEP_LIMIT = 800

# Camera
CAMERA_HEIGHT = ROBOT_HEIGHT
CAMERA_PITCH = -30
CAMERA_FOV = 60
CAMERA_NEAR = 0.01
CAMERA_FAR = 10
FLOOR_SEG_INDEX = 1
OBSTACLE_SEG_INDEX = 2
RECEPTACLE_SEG_INDEX = 3
CUBE_SEG_INDEX = 4
ROBOT_SEG_INDEX = 5
MAX_SEG_INDEX = 8

# Overhead map
LOCAL_MAP_PIXEL_WIDTH = 96
LOCAL_MAP_WIDTH = 1  # 1 meter
LOCAL_MAP_PIXELS_PER_METER = LOCAL_MAP_PIXEL_WIDTH / LOCAL_MAP_WIDTH
MAP_UPDATE_STEPS = 250

class Environment:
    def __init__(
        # pylint: disable=bad-continuation
        # This comment is here to make code folding work
            self, room_length=1.0, room_width=0.5, num_cubes=10, obstacle_config='small_empty',
            use_distance_to_receptacle_channel=True, distance_to_receptacle_channel_scale=0.25,
            use_shortest_path_to_receptacle_channel=False, use_shortest_path_channel=False, shortest_path_channel_scale=0.25,
            use_position_channel=False, position_channel_scale=0.25,
            partial_rewards_scale=2.0, use_shortest_path_partial_rewards=False, collision_penalty=0.25, nonmovement_penalty=0.25,
            use_shortest_path_movement=False, fixed_step_size=None, use_steering_commands=False, steering_commands_num_turns=4,
            ministep_size=0.25, inactivity_cutoff=100, random_seed=None,
            use_gui=False, show_debug_annotations=False, show_occupancy_map=False,
        ):

        ################################################################################
        # Store arguments

        # Room config
        self.room_length = room_length
        self.room_width = room_width
        self.num_cubes = num_cubes
        self.obstacle_config = obstacle_config

        # State representation
        self.use_distance_to_receptacle_channel = use_distance_to_receptacle_channel
        self.distance_to_receptacle_channel_scale = distance_to_receptacle_channel_scale
        self.use_shortest_path_to_receptacle_channel = use_shortest_path_to_receptacle_channel
        self.use_shortest_path_channel = use_shortest_path_channel
        self.shortest_path_channel_scale = shortest_path_channel_scale
        self.use_position_channel = use_position_channel
        self.position_channel_scale = position_channel_scale

        # Rewards
        self.partial_rewards_scale = partial_rewards_scale
        self.use_shortest_path_partial_rewards = use_shortest_path_partial_rewards
        self.collision_penalty = collision_penalty
        self.nonmovement_penalty = nonmovement_penalty

        # Movement
        self.use_shortest_path_movement = use_shortest_path_movement
        self.fixed_step_size = fixed_step_size
        self.use_steering_commands = use_steering_commands
        self.steering_commands_num_turns = steering_commands_num_turns

        # Misc
        self.ministep_size = ministep_size
        self.inactivity_cutoff = inactivity_cutoff
        self.random_seed = random_seed
        self.use_gui = use_gui
        self.show_debug_annotations = show_debug_annotations
        self.show_occupancy_map = show_occupancy_map

        pprint.PrettyPrinter(indent=4).pprint(self.__dict__)

        ################################################################################
        # Room and objects

        assert self.num_cubes > 0
        assert self.room_length >= self.room_width

        # Random placement of robots, cubes, and obstacles
        self.random_state = np.random.RandomState(self.random_seed)

        # Trash receptacle
        self.receptacle_position = [self.room_length / 2 - RECEPTACLE_WIDTH / 2, self.room_width / 2 - RECEPTACLE_WIDTH / 2, 0]
        self.receptacle_id = None

        # Obstacles
        self.obstacle_ids = None
        self.min_obstacle_id = None
        self.max_obstacle_id = None

        # Robot
        self.robot_id = None
        self.robot_cid = None
        self.robot_position = None
        self.robot_heading = None
        self.robot_cumulative_cubes = None
        self.robot_cumulative_distance = None
        self.robot_cumulative_reward = None

        # Cubes
        self.cube_ids = None
        self.min_cube_id = None
        self.max_cube_id = None
        self.available_cube_ids_set = None
        self.removed_cube_ids_set = None

        # Used to determine whether to end episode
        self.inactivity_counter = None

        ################################################################################
        # State representation

        # Forward-facing camera
        self.camera_image_pixel_height = int(1.63 * LOCAL_MAP_PIXEL_WIDTH)
        self.camera_aspect = 16 / 9
        self.camera_image_pixel_width = int(self.camera_aspect * self.camera_image_pixel_height)
        self.projection_matrix = p.computeProjectionMatrixFOV(CAMERA_FOV, self.camera_aspect, CAMERA_NEAR, CAMERA_FAR)

        # Robot state
        robot_pixel_width = int(2 * ROBOT_RADIUS * LOCAL_MAP_PIXELS_PER_METER)
        self.robot_state_channel = np.zeros((LOCAL_MAP_PIXEL_WIDTH, LOCAL_MAP_PIXEL_WIDTH), dtype=np.float32)
        start = int(np.floor(LOCAL_MAP_PIXEL_WIDTH / 2 - robot_pixel_width / 2))
        for i in range(start, start + robot_pixel_width):
            for j in range(start, start + robot_pixel_width):
                # Circular robot mask
                if (((i + 0.5) - LOCAL_MAP_PIXEL_WIDTH / 2)**2 + ((j + 0.5) - LOCAL_MAP_PIXEL_WIDTH / 2)**2)**0.5 < robot_pixel_width / 2:
                    self.robot_state_channel[i, j] = 1

        # Distance channel can be precomputed
        if self.use_distance_to_receptacle_channel:
            self.global_distance_to_receptacle_map = self._create_global_distance_to_receptacle_map()

        # Used to mask out the wall pixels when updating occupancy map
        room_mask = self._create_padded_room_zeros()
        room_width_pixels = int(2 * np.ceil(((self.room_width - 2 * ROBOT_HALF_WIDTH) * LOCAL_MAP_PIXELS_PER_METER) / 2))
        room_length_pixels = int(2 * np.ceil(((self.room_length - 2 * ROBOT_HALF_WIDTH) * LOCAL_MAP_PIXELS_PER_METER) / 2))
        start_i, start_j = int(room_mask.shape[0] / 2 - room_width_pixels / 2), int(room_mask.shape[1] / 2 - room_length_pixels / 2)
        room_mask[start_i:start_i + room_width_pixels, start_j:start_j + room_length_pixels] = 1
        self.wall_map = 1 - room_mask

        self.global_overhead_map = None
        self.configuration_space = None
        self.configuration_space_thin = None
        self.closest_cspace_indices = None
        self.occupancy_map = None

        if self.show_occupancy_map:
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
            self.plt = plt
            self.plt.figure(figsize=(4, 4 * self.room_width / self.room_length))
            self.plt.ion()
            self.free_space_map = None

        # Position channel can be precomputed
        if self.use_position_channel:
            self.local_position_map_x, self.local_position_map_y = self._create_local_position_map()

        ################################################################################
        # pybullet

        if self.use_gui:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p.connect(p.DIRECT)

        p.resetDebugVisualizerCamera(
            0.47 + (5.25 - 0.47) / (10 - 0.7) * (self.room_length - 0.7), 0, -70,
            (0, -(0.07 + (1.5 - 0.07) / (10 - 0.7) * (self.room_width - 0.7)), 0)
        )

        self.assets_dir = Path(__file__).parent / 'assets'

        ################################################################################
        # Misc

        if self.use_steering_commands:
            increment = 360 / self.steering_commands_num_turns
            self.simple_action_space_turn_angles = [np.radians(i * increment) for i in range(self.steering_commands_num_turns)]

    def reset(self):
        ################################################################################
        # Room and objects

        # Reset pybullet
        p.resetSimulation()
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -9.8)

        # Create the environment
        self._create_environment()
        self._reset_poses()
        self._step_simulation_until_still()

        ################################################################################
        # State representation

        self.global_overhead_map = self._create_padded_room_zeros()
        self.occupancy_map = self._create_padded_room_zeros()
        if self.show_occupancy_map:
            self.free_space_map = self._create_padded_room_zeros()

        self.robot_position, self.robot_heading = self._get_robot_pose()
        self._update_state()
        if self.show_occupancy_map:
            self._update_occupancy_map_visualization()

        ################################################################################
        # Variables

        # Sets to track cubes
        self.available_cube_ids_set = set(self.cube_ids)  # Cubes that have not been removed
        self.removed_cube_ids_set = set()

        # Counters
        self.inactivity_counter = 0
        self.robot_cumulative_cubes = 0
        self.robot_cumulative_distance = 0
        self.robot_cumulative_reward = 0

        ################################################################################

        return self.get_state()

    def step(self, action):
        return self._step(action)

    def _step(self, action, dry_run=False):
        ################################################################################
        # Setup

        # Store new action
        if self.use_steering_commands:
            robot_action = action
        else:
            robot_action = np.unravel_index(action, (LOCAL_MAP_PIXEL_WIDTH, LOCAL_MAP_PIXEL_WIDTH))

        robot_cubes = 0
        robot_reward = 0
        robot_hit_obstacle = False

        # Initial pose
        robot_initial_position, robot_initial_heading = self._get_robot_pose()

        # Compute target end effector position
        if self.use_steering_commands:
            straight_line_dist = self.fixed_step_size + ROBOT_RADIUS
            turn_angle = np.radians(90) - self.simple_action_space_turn_angles[robot_action]
        else:
            # Compute distance from front of robot (not robot center), which is used to find the
            # robot position and heading needed in order to place end effector over specified location.
            x_movement = -LOCAL_MAP_WIDTH / 2 + float(robot_action[1]) / LOCAL_MAP_PIXELS_PER_METER
            y_movement = LOCAL_MAP_WIDTH / 2 - float(robot_action[0]) / LOCAL_MAP_PIXELS_PER_METER
            if self.fixed_step_size is not None:
                straight_line_dist = self.fixed_step_size + ROBOT_RADIUS
            else:
                straight_line_dist = np.sqrt(x_movement**2 + y_movement**2)
            turn_angle = np.arctan2(-x_movement, y_movement)
        straight_line_heading = restrict_heading_range(robot_initial_heading + turn_angle)

        robot_target_end_effector_position = [
            robot_initial_position[0] + straight_line_dist * np.cos(straight_line_heading),
            robot_initial_position[1] + straight_line_dist * np.sin(straight_line_heading),
            0
        ]

        # Do not allow going outside the room
        diff = np.asarray(robot_target_end_effector_position) - np.asarray(robot_initial_position)
        ratio_x, ratio_y = (1, 1)
        bound_x = np.sign(robot_target_end_effector_position[0]) * self.room_length / 2
        bound_y = np.sign(robot_target_end_effector_position[1]) * self.room_width / 2
        if abs(robot_target_end_effector_position[0]) > abs(bound_x):
            ratio_x = (bound_x - robot_initial_position[0]) / (robot_target_end_effector_position[0] - robot_initial_position[0])
        if abs(robot_target_end_effector_position[1]) > abs(bound_y):
            ratio_y = (bound_y - robot_initial_position[1]) / (robot_target_end_effector_position[1] - robot_initial_position[1])
        ratio = min(ratio_x, ratio_y)
        robot_target_end_effector_position = (np.asarray(robot_initial_position) + ratio * diff).tolist()
        if dry_run:
            # Used in physical environment
            return robot_target_end_effector_position

        # Compute waypoint positions
        if self.use_shortest_path_movement:
            robot_waypoint_positions = self._shortest_path(robot_initial_position, robot_target_end_effector_position, check_straight=True)
        else:
            robot_waypoint_positions = [robot_initial_position, robot_target_end_effector_position]

        # Compute waypoint headings
        robot_waypoint_headings = [None]
        for i in range(1, len(robot_waypoint_positions)):
            x_diff = robot_waypoint_positions[i][0] - robot_waypoint_positions[i - 1][0]
            y_diff = robot_waypoint_positions[i][1] - robot_waypoint_positions[i - 1][1]
            waypoint_heading = restrict_heading_range(np.arctan2(y_diff, x_diff))
            robot_waypoint_headings.append(waypoint_heading)

        # Compute movement from final waypoint to the target and apply ROBOT_RADIUS offset to the final waypoint
        dist_to_target_end_effector_position = distance(robot_waypoint_positions[-2], robot_waypoint_positions[-1])
        signed_dist = dist_to_target_end_effector_position - ROBOT_RADIUS

        robot_move_sign = np.sign(signed_dist)  # Whether to move backwards to get to final position
        robot_target_heading = robot_waypoint_headings[-1]
        robot_target_position = [
            robot_waypoint_positions[-2][0] + signed_dist * np.cos(robot_target_heading),
            robot_waypoint_positions[-2][1] + signed_dist * np.sin(robot_target_heading),
            0
        ]
        robot_waypoint_positions[-1] = robot_target_position

        # Avoid awkward backing up to reach the last waypoint
        if len(robot_waypoint_positions) > 2 and signed_dist < 0:
            robot_waypoint_positions[-2] = robot_waypoint_positions[-1]
            x_diff = robot_waypoint_positions[-2][0] - robot_waypoint_positions[-3][0]
            y_diff = robot_waypoint_positions[-2][1] - robot_waypoint_positions[-3][1]
            waypoint_heading = restrict_heading_range(np.arctan2(y_diff, x_diff))
            robot_waypoint_headings[-2] = waypoint_heading
            robot_move_sign = 1

        # Store the initial configuration space at the start of the step (for partial reward calculation)
        initial_configuration_space = self.configuration_space.copy()

        # Store initial cube distances for partial reward calculation
        initial_cube_distances = {}
        cube_ids_to_track = list(self.cube_ids)
        for cube_id in cube_ids_to_track:
            cube_position, _ = p.getBasePositionAndOrientation(cube_id)
            if self.use_shortest_path_partial_rewards:
                dist = self._shortest_path_distance(cube_position, self.receptacle_position)
            else:
                dist = distance(cube_position, self.receptacle_position)
            initial_cube_distances[cube_id] = dist

        ################################################################################
        # Movement

        self.robot_position = robot_initial_position.copy()
        self.robot_heading = robot_initial_heading
        robot_is_moving = True
        robot_distance = 0
        robot_waypoint_index = 1
        robot_prev_waypoint_position = robot_waypoint_positions[robot_waypoint_index - 1]
        robot_waypoint_position = robot_waypoint_positions[robot_waypoint_index]
        robot_waypoint_heading = robot_waypoint_headings[robot_waypoint_index]

        sim_steps = 0
        while True:
            if not robot_is_moving:
                break

            # Store pose to determine distance moved during simulation step
            robot_prev_position = self.robot_position.copy()
            robot_prev_heading = self.robot_heading

            # Compute robot pose for new constraint
            robot_new_position = self.robot_position.copy()
            robot_new_heading = self.robot_heading
            heading_diff = heading_difference(self.robot_heading, robot_waypoint_heading)
            if np.abs(heading_diff) > TURN_STEP_SIZE:
                # Turn towards next waypoint first
                robot_new_heading += np.sign(heading_diff) * TURN_STEP_SIZE
            else:
                dx = robot_waypoint_position[0] - self.robot_position[0]
                dy = robot_waypoint_position[1] - self.robot_position[1]
                if distance(self.robot_position, robot_waypoint_position) < MOVE_STEP_SIZE:
                    robot_new_position = robot_waypoint_position
                else:
                    if robot_waypoint_index == len(robot_waypoint_positions) - 1:
                        move_sign = robot_move_sign
                    else:
                        move_sign = 1
                    robot_new_heading = np.arctan2(move_sign * dy, move_sign * dx)
                    robot_new_position[0] += move_sign * MOVE_STEP_SIZE * np.cos(robot_new_heading)
                    robot_new_position[1] += move_sign * MOVE_STEP_SIZE * np.sin(robot_new_heading)

            # Set new constraint to move the robot to new pose
            p.changeConstraint(self.robot_cid, jointChildPivot=robot_new_position, jointChildFrameOrientation=p.getQuaternionFromEuler([0, 0, robot_new_heading]), maxForce=MOVEMENT_MAX_FORCE)

            p.stepSimulation()

            # Get new robot pose
            self.robot_position, self.robot_heading = self._get_robot_pose()
            self.robot_position[2] = 0

            # Stop moving if robot collided with obstacle
            if distance(robot_prev_waypoint_position, self.robot_position) > MOVE_STEP_SIZE:
                contact_points = p.getContactPoints(self.robot_id)
                if len(contact_points) > 0:
                    for contact_point in contact_points:
                        if contact_point[2] in self.obstacle_ids + [self.robot_id]:
                            robot_is_moving = False
                            robot_hit_obstacle = True
                            break  # Note: self.robot_distance does not get not updated

            # Robot no longer turning or moving
            if (distance(self.robot_position, robot_prev_position) < NOT_MOVING_THRESHOLD
                    and np.abs(self.robot_heading - robot_prev_heading) < NOT_TURNING_THRESHOLD):

                # Update distance moved
                robot_distance += distance(robot_prev_waypoint_position, self.robot_position)

                if self.show_debug_annotations:
                    p.addUserDebugLine(robot_prev_waypoint_position[:2] + [0.001], self.robot_position[:2] + [0.001], DEBUG_LINE_COLOR)

                # Increment waypoint index, or stop moving if done
                if robot_waypoint_index == len(robot_waypoint_positions) - 1:
                    robot_is_moving = False
                else:
                    robot_waypoint_index += 1
                    robot_prev_waypoint_position = robot_waypoint_positions[robot_waypoint_index - 1]
                    robot_waypoint_position = robot_waypoint_positions[robot_waypoint_index]
                    robot_waypoint_heading = robot_waypoint_headings[robot_waypoint_index]

            # Break if robot is stuck
            sim_steps += 1
            if sim_steps > STEP_LIMIT:
                break  # Note: self.robot_distance does not get not updated

            if sim_steps % MAP_UPDATE_STEPS == 0:
                self._update_state()
                if self.show_occupancy_map:
                    self._update_occupancy_map_visualization(robot_waypoint_positions, robot_target_end_effector_position)

        # Step the simulation until everything is still
        self._step_simulation_until_still()

        ################################################################################
        # Process cubes

        # Store final cube positions and remove cubes that are above the wall
        to_remove = []
        final_cube_positions = {}
        for cube_id in self.available_cube_ids_set:
            cube_position, _ = p.getBasePositionAndOrientation(cube_id)
            final_cube_positions[cube_id] = cube_position
            if cube_position[2] > WALL_HEIGHT + 0.49 * CUBE_WIDTH or cube_position[2] < CUBE_WIDTH / 4:
                self._remove_cube(cube_id, out_of_bounds=True)
                to_remove.append(cube_id)
        for cube_id in to_remove:
            self.available_cube_ids_set.remove(cube_id)

        # Give partial rewards
        for cube_id in self.available_cube_ids_set:
            cube_position = final_cube_positions[cube_id]
            if self.use_shortest_path_partial_rewards:
                dist = self._shortest_path_distance(cube_position, self.receptacle_position, configuration_space=initial_configuration_space)
            else:
                dist = distance(cube_position, self.receptacle_position)
            dist_moved = initial_cube_distances[cube_id] - dist
            robot_reward += self.partial_rewards_scale * dist_moved

        # Give rewards for cubes in receptacle (and remove the cubes)
        to_remove = []
        for cube_id in self.available_cube_ids_set:
            cube_position = final_cube_positions[cube_id]
            if self._cube_position_in_receptacle(cube_position):
                to_remove.append(cube_id)
                self._remove_cube(cube_id)
                robot_cubes += 1
                robot_reward += 1
        for cube_id in to_remove:
            self.available_cube_ids_set.remove(cube_id)

        ################################################################################
        # Update state representation

        self.robot_position, self.robot_heading = self._get_robot_pose()
        self._update_state()
        if self.show_occupancy_map:
            self._update_occupancy_map_visualization(robot_waypoint_positions, robot_target_end_effector_position)

        ################################################################################
        # Compute stats

        # Get final pose
        self.robot_position, self.robot_heading = self._get_robot_pose()

        # Add distance traveled to cumulative distance
        self.robot_cumulative_distance += robot_distance

        # Calculate amount turned to check if robot turned this step
        robot_turn_angle = heading_difference(robot_initial_heading, self.robot_heading)

        # Increment inactivity counter, which measures steps elapsed since the previous cube was stashed
        if robot_cubes == 0:
            self.inactivity_counter += 1

        # Determine whether episode is done
        done = False
        if len(self.removed_cube_ids_set) == self.num_cubes or self.inactivity_counter >= self.inactivity_cutoff:
            done = True

        # Compute reward for the step
        if robot_hit_obstacle:
            robot_reward -= self.collision_penalty
        if robot_distance < NONMOVEMENT_DIST_THRESHOLD and abs(robot_turn_angle) < NONMOVEMENT_TURN_THRESHOLD:
            robot_reward -= self.nonmovement_penalty
        self.robot_cumulative_cubes += robot_cubes
        self.robot_cumulative_reward += robot_reward

        # Compute items to return
        state = self.get_state(done=done)
        reward = robot_reward
        ministeps = robot_distance / self.ministep_size
        info = {
            'ministeps': ministeps,
            'inactivity': self.inactivity_counter,
            'cumulative_cubes': self.robot_cumulative_cubes,
            'cumulative_distance': self.robot_cumulative_distance,
            'cumulative_reward': self.robot_cumulative_reward,
        }

        return state, reward, done, info

    @staticmethod
    def close():
        p.disconnect()

    @staticmethod
    def get_state_width():
        return LOCAL_MAP_PIXEL_WIDTH

    def get_action_space(self):
        if self.use_steering_commands:
            return self.steering_commands_num_turns
        return LOCAL_MAP_PIXEL_WIDTH * LOCAL_MAP_PIXEL_WIDTH

    def get_camera_image(self, image_width=1024, image_height=768):
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.use_gui else p.ER_TINY_RENDERER
        _, _, rgb, _, _ = p.getCameraImage(image_width, image_height, flags=p.ER_NO_SEGMENTATION_MASK, renderer=renderer)
        return rgb

    @staticmethod
    def start_video_logging(video_path):
        return p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)

    @staticmethod
    def stop_video_logging(log_id):
        p.stopStateLogging(log_id)

    def _create_environment(self):
        # Create floor
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            # Create custom obj and urdf for current room size
            room_length_with_walls = self.room_length + 2 * WALL_THICKNESS
            room_width_with_walls = self.room_width + 2 * WALL_THICKNESS
            plane_obj_path = str(Path(tmp_dir_name) / 'plane.obj')
            with open(self.assets_dir / 'plane.obj.template') as f1:
                with open(plane_obj_path, 'w') as f2:
                    f2.write(f1.read().replace('HALFLENGTH', str(room_length_with_walls / GLOBAL_SCALING / 2)).replace('HALFWIDTH', str(room_width_with_walls / GLOBAL_SCALING / 2)))
            plane_urdf_path = str(Path(tmp_dir_name) / 'plane.urdf')
            with open(self.assets_dir / 'plane.urdf.template') as f1:
                with open(plane_urdf_path, 'w') as f2:
                    f2.write(f1.read().replace('LENGTH', str(room_length_with_walls / GLOBAL_SCALING)).replace('WIDTH', str(room_width_with_walls / GLOBAL_SCALING)))
            p.loadURDF(plane_urdf_path, globalScaling=GLOBAL_SCALING)

        # Create obstacles (including walls)
        self.obstacle_ids = self._create_obstacles()
        self.min_obstacle_id = min(self.obstacle_ids)
        self.max_obstacle_id = max(self.obstacle_ids)

        # Create trash receptacle
        receptacle_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0, 0, 0])
        receptacle_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[RECEPTACLE_WIDTH / 2, RECEPTACLE_WIDTH / 2, 0], rgbaColor=RECEPTACLE_COLOR, visualFramePosition=[0, 0, 0.0001])
        self.receptacle_id = p.createMultiBody(0.01, receptacle_collision_shape_id, receptacle_visual_shape_id, self.receptacle_position)

        # Create cubes
        cube_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=(3 * [CUBE_WIDTH / 2]))
        cube_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=(3 * [CUBE_WIDTH / 2]), rgbaColor=CUBE_COLOR)
        self.cube_ids = []
        for _ in range(self.num_cubes):
            self.cube_ids.append(p.createMultiBody(CUBE_MASS, cube_collision_shape_id, cube_visual_shape_id))
        self.min_cube_id = min(self.cube_ids)
        self.max_cube_id = max(self.cube_ids)

        # Create robot and initialize contraint
        self.robot_id = p.loadURDF(str(self.assets_dir / 'robot.urdf'))
        self.robot_cid = p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])

    def _create_obstacles(self):
        obstacles = []

        # Create walls
        for x, y, length, width in [
                (-self.room_length / 2 - WALL_THICKNESS / 2, 0, WALL_THICKNESS, self.room_width),
                (self.room_length / 2 + WALL_THICKNESS / 2, 0, WALL_THICKNESS, self.room_width),
                (0, -self.room_width / 2 - WALL_THICKNESS / 2, self.room_length + 2 * WALL_THICKNESS, WALL_THICKNESS),
                (0, self.room_width / 2 + WALL_THICKNESS / 2, self.room_length + 2 * WALL_THICKNESS, WALL_THICKNESS)
            ]:
            obstacles.append({'type': 'wall', 'position': (x, y), 'heading': 0, 'length': length, 'width': width})

        def get_obstacle_box(obstacle, buffer_width=0.08):
            x, y = obstacle['position']
            length, width = obstacle['length'], obstacle['width']
            b = box(x - length / 2, y - width / 2, x + length / 2, y + width / 2)
            if buffer_width > 0:
                b = b.buffer(buffer_width)
            return b

        def get_receptacle_box():
            obstacle = {'position': self.receptacle_position[:2], 'heading': 0, 'length': RECEPTACLE_WIDTH, 'width': RECEPTACLE_WIDTH}
            return get_obstacle_box(obstacle, buffer_width=0)

        def add_random_columns(obstacles, max_num_columns):
            num_columns = self.random_state.randint(max_num_columns) + 1
            column_length = 0.1
            column_width = 0.1
            buffer_width = 0.08
            polygons = [get_receptacle_box()] + [get_obstacle_box(obstacle) for obstacle in obstacles]

            for _ in range(10):
                new_obstacles = []
                new_polygons = []
                polygon_union = unary_union(polygons)
                for _ in range(num_columns):
                    for _ in range(100):
                        x = self.random_state.uniform(
                            -self.room_length / 2 + 2 * buffer_width + column_length / 2,
                            self.room_length / 2 - 2 * buffer_width - column_length / 2
                        )
                        y = self.random_state.uniform(
                            -self.room_width / 2 + 2 * buffer_width + column_width / 2,
                            self.room_width / 2 - 2 * buffer_width - column_width / 2
                        )
                        obstacle = {'type': 'column', 'position': (x, y), 'heading': 0, 'length': column_length, 'width': column_width}
                        b = get_obstacle_box(obstacle)
                        if not polygon_union.intersects(b):
                            new_obstacles.append(obstacle)
                            new_polygons.append(b)
                            polygon_union = unary_union(polygons + new_polygons)
                            break
                if len(new_obstacles) == num_columns:
                    break
            return new_obstacles

        def add_random_horiz_divider():
            divider_length = 0.8
            divider_width = 0.05
            buffer_width = (2 + np.sqrt(2)) * ROUNDED_CORNER_WIDTH
            polygons = unary_union([get_receptacle_box()])

            for _ in range(10):
                new_obstacles = []
                for _ in range(100):
                    x = self.room_length / 2 - divider_length / 2
                    y = self.random_state.uniform(
                        -self.room_width / 2 + buffer_width + divider_width / 2,
                        self.room_width / 2 - buffer_width - divider_width / 2
                    )
                    obstacle = {'type': 'divider', 'position': [x, y], 'heading': 0, 'length': divider_length, 'width': divider_width}
                    b_no_inflate = get_obstacle_box(obstacle, buffer_width=0)
                    if not polygons.intersects(b_no_inflate):
                        new_obstacles.append(obstacle)
                        break
                if len(new_obstacles) == 1:
                    break
            return new_obstacles

        # Create obstacles
        if self.obstacle_config == 'small_empty':
            pass
        elif self.obstacle_config == 'small_columns':
            obstacles.extend(add_random_columns(obstacles, 3))
        elif self.obstacle_config == 'large_columns':
            obstacles.extend(add_random_columns(obstacles, 8))
        elif self.obstacle_config == 'large_divider':
            obstacles.extend(add_random_horiz_divider())
        else:
            raise Exception(self.obstacle_config)

        # Create room corners
        for i, (x, y) in enumerate([
                (-self.room_length / 2, self.room_width / 2),
                (self.room_length / 2, self.room_width / 2),
                (self.room_length / 2, -self.room_width / 2),
                (-self.room_length / 2, -self.room_width / 2)
            ]):
            if i == 1:  # Skip the receptacle corner
                continue
            heading = -np.radians(i * 90)
            offset = ROUNDED_CORNER_WIDTH / np.sqrt(2)
            adjusted_position = (x + offset * np.cos(heading - np.radians(45)), y + offset * np.sin(heading - np.radians(45)))
            obstacles.append({'type': 'corner', 'position': adjusted_position, 'heading': heading})

        # Create additional corners for the divider
        new_obstacles = []
        for obstacle in obstacles:
            if obstacle['type'] == 'divider':
                (x, y), length, width = obstacle['position'], obstacle['length'], obstacle['width']
                corner_positions = [(self.room_length / 2, y - width / 2), (self.room_length / 2, y + width / 2)]
                corner_headings = [-90, 180]
                for position, heading in zip(corner_positions, corner_headings):
                    heading = np.radians(heading)
                    offset = ROUNDED_CORNER_WIDTH / np.sqrt(2)
                    adjusted_position = (position[0] + offset * np.cos(heading - np.radians(45)), position[1] + offset * np.sin(heading - np.radians(45)))
                    obstacles.append({'type': 'corner', 'position': adjusted_position, 'heading': heading})
        obstacles.extend(new_obstacles)

        # Add obstacles to pybullet
        obstacle_ids = []
        for obstacle in obstacles:
            if obstacle['type'] == 'corner':
                obstacle_collision_shape_id = p.createCollisionShape(p.GEOM_MESH, fileName=str(self.assets_dir / 'corner.obj'))
                obstacle_visual_shape_id = p.createVisualShape(p.GEOM_MESH, fileName=str(self.assets_dir / 'corner.obj'), rgbaColor=OBSTACLE_COLOR)
            else:
                obstacle_half_extents = [obstacle['length'] / 2, obstacle['width'] / 2, WALL_HEIGHT / 2]
                obstacle_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle_half_extents)
                obstacle_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=OBSTACLE_COLOR)
            obstacle_id = p.createMultiBody(
                0, obstacle_collision_shape_id, obstacle_visual_shape_id,
                [obstacle['position'][0], obstacle['position'][1], WALL_HEIGHT / 2], p.getQuaternionFromEuler([0, 0, obstacle['heading']])
            )
            obstacle_ids.append(obstacle_id)

        return obstacle_ids

    def _reset_poses(self):
        # Random robot pose
        robot_positions_x, robot_positions_y = self._random_position(ROBOT_RADIUS, 1)
        robot_positions = np.stack([robot_positions_x, robot_positions_y, np.tile(0, 1)], axis=1)
        robot_headings = self.random_state.uniform(-np.pi, np.pi, 1)
        p.resetBasePositionAndOrientation(self.robot_id, robot_positions[0], p.getQuaternionFromEuler([0, 0, robot_headings[0]]))
        p.changeConstraint(self.robot_cid, jointChildPivot=robot_positions[0], jointChildFrameOrientation=p.getQuaternionFromEuler([0, 0, robot_headings[0]]), maxForce=MOVEMENT_MAX_FORCE)

        # Random cube poses
        for cube_id in self.cube_ids:
            cube_heading = self.random_state.uniform(-np.pi, np.pi)

            # Only spawn cubes outside of the receptacle
            done = False
            while not done:
                cube_position_x, cube_position_y = self._random_position(CUBE_WIDTH / 2)
                cube_position = [cube_position_x, cube_position_y]
                if not self._cube_position_in_receptacle(cube_position):
                    done = True
            cube_position.append(CUBE_WIDTH / 2)
            p.resetBasePositionAndOrientation(cube_id, cube_position, p.getQuaternionFromEuler([0, 0, cube_heading]))

    def _random_position(self, radius, size=None):
        position_x = self.random_state.uniform(-self.room_length / 2 + radius, self.room_length / 2 - radius, size)
        position_y = self.random_state.uniform(-self.room_width / 2 + radius, self.room_width / 2 - radius, size)
        return position_x, position_y

    def _step_simulation_until_still(self):
        # Kick-start gravity
        for _ in range(2):
            p.stepSimulation()

        prev_positions = []
        sim_steps = 0
        done = False
        while not done:
            # Check whether anything moved since last step
            positions = []
            for body_id in self.cube_ids + [self.robot_id]:
                position, _ = p.getBasePositionAndOrientation(body_id)
                positions.append(position)
            if len(prev_positions) > 0:
                done = True
                for i, position in enumerate(positions):
                    # Ignore cubes that are in free fall
                    if position[2] > 0 and np.linalg.norm(np.asarray(prev_positions[i]) - np.asarray(position)) > NOT_MOVING_THRESHOLD:
                        done = False
                        break
            prev_positions = positions

            p.stepSimulation()

            # If robot is stacked on top of a cube, reset its position
            self.robot_position, self.robot_heading = self._get_robot_pose()
            if np.abs(self.robot_position[2]) > ROBOT_HEIGHT / 4:
                done = False
                robot_position_x, robot_position_y = self._random_position(ROBOT_RADIUS)
                self.robot_position = [robot_position_x, robot_position_y]
                self.robot_position.append(0)
            p.changeConstraint(self.robot_cid, jointChildPivot=self.robot_position, jointChildFrameOrientation=p.getQuaternionFromEuler([0, 0, self.robot_heading]), maxForce=500)

            # Break if stuck
            sim_steps += 1
            if sim_steps > STEP_LIMIT:
                break

    def _get_robot_pose(self):
        robot_position, robot_orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_position = list(robot_position)
        robot_heading = orientation_to_heading(robot_orientation)
        return robot_position, robot_heading

    def _remove_cube(self, cube_id, out_of_bounds=False):
        # Hide cubes 1000 m below
        p.resetBasePositionAndOrientation(cube_id, [0, 0, -1000], [0, 0, 0, 1])
        self.removed_cube_ids_set.add(cube_id)
        if not out_of_bounds:
            self.inactivity_counter = 0

    def _cube_position_in_receptacle(self, cube_position):
        cube_position_copy = list(cube_position)

        # Sometimes the cube gets squished against the wall
        if np.abs(cube_position_copy[0]) + CUBE_WIDTH / 2 > self.room_length / 2:
            cube_position_copy[0] = np.sign(cube_position_copy[0]) * (self.room_length / 2 - CUBE_WIDTH)
        if np.abs(cube_position_copy[1]) + CUBE_WIDTH / 2 > self.room_width / 2:
            cube_position_copy[1] = np.sign(cube_position_copy[1]) * (self.room_width / 2 - CUBE_WIDTH)

        vec = np.asarray(cube_position_copy[:2]) - np.asarray(self.receptacle_position[:2])
        rotated_vec = [np.cos(-0) * vec[0] - np.sin(-0) * vec[1], np.sin(-0) * vec[0] + np.cos(0) * vec[1]]
        if (np.abs(rotated_vec[0]) < (RECEPTACLE_WIDTH - CUBE_WIDTH) / 2 and np.abs(rotated_vec[1]) < (RECEPTACLE_WIDTH - CUBE_WIDTH) / 2):
            return True

        return False

    def _update_occupancy_map_visualization(self, robot_waypoint_positions=None, robot_target_end_effector_position=None):
        occupancy_map_vis = self._create_padded_room_zeros() + 0.5
        occupancy_map_vis[np.isclose(self.free_space_map, 1)] = 1
        occupancy_map_vis[np.isclose(self.occupancy_map, 1)] = 0
        height, width = occupancy_map_vis.shape
        height, width = height / LOCAL_MAP_PIXELS_PER_METER, width / LOCAL_MAP_PIXELS_PER_METER
        self.plt.clf()
        self.plt.axis('off')
        self.plt.axis([
            -self.room_length / 2 - ROBOT_RADIUS, self.room_length / 2 + ROBOT_RADIUS,
            -self.room_width / 2 - ROBOT_RADIUS, self.room_width / 2 + ROBOT_RADIUS
        ])
        self.plt.imshow(255 * occupancy_map_vis, extent=(-width / 2, width / 2, -height / 2, height / 2), cmap='gray', vmin=0, vmax=255)
        if robot_waypoint_positions is not None:
            self.plt.plot(np.asarray(robot_waypoint_positions)[:, 0], np.asarray(robot_waypoint_positions)[:, 1], color='r', marker='.')
        if robot_target_end_effector_position is not None:
            self.plt.plot(robot_target_end_effector_position[0], robot_target_end_effector_position[1], color='r', marker='x')
        self.plt.pause(0.001)  # Update display

    def _create_padded_room_zeros(self):
        return np.zeros((
            int(2 * np.ceil((self.room_width * LOCAL_MAP_PIXELS_PER_METER + LOCAL_MAP_PIXEL_WIDTH * np.sqrt(2)) / 2)),  # Ensure even
            int(2 * np.ceil((self.room_length * LOCAL_MAP_PIXELS_PER_METER + LOCAL_MAP_PIXEL_WIDTH * np.sqrt(2)) / 2))
        ), dtype=np.float32)

    def _create_global_distance_to_receptacle_map(self):
        global_map = self._create_padded_room_zeros()
        for i in range(global_map.shape[0]):
            for j in range(global_map.shape[1]):
                position_x = ((j + 1) - global_map.shape[1] / 2) / LOCAL_MAP_PIXELS_PER_METER
                position_y = -((i + 1) - global_map.shape[0] / 2) / LOCAL_MAP_PIXELS_PER_METER
                global_map[i, j] = ((self.receptacle_position[0] - position_x)**2 + (self.receptacle_position[1] - position_y)**2)**0.5
        global_map /= (np.sqrt(2) * LOCAL_MAP_PIXEL_WIDTH) / LOCAL_MAP_PIXELS_PER_METER
        global_map *= self.distance_to_receptacle_channel_scale
        return global_map

    def _create_local_position_map(self):
        local_position_map_x = np.zeros((LOCAL_MAP_PIXEL_WIDTH, LOCAL_MAP_PIXEL_WIDTH), dtype=np.float32)
        local_position_map_y = np.zeros((LOCAL_MAP_PIXEL_WIDTH, LOCAL_MAP_PIXEL_WIDTH), dtype=np.float32)
        for i in range(local_position_map_x.shape[0]):
            for j in range(local_position_map_x.shape[1]):
                position_x = ((j + 1) - local_position_map_x.shape[1] / 2) / LOCAL_MAP_PIXELS_PER_METER
                position_y = -((i + 1) - local_position_map_x.shape[0] / 2) / LOCAL_MAP_PIXELS_PER_METER
                local_position_map_x[i][j] = position_x
                local_position_map_y[i][j] = position_y
        local_position_map_x *= self.position_channel_scale
        local_position_map_y *= self.position_channel_scale
        return local_position_map_x, local_position_map_y

    def _create_global_shortest_path_to_receptacle_map(self):
        global_map = self._create_padded_room_zeros() + np.inf
        pixel_i, pixel_j = position_to_pixel_indices(self.receptacle_position[0], self.receptacle_position[1], self.configuration_space.shape)
        pixel_i, pixel_j = self._closest_valid_cspace_indices(pixel_i, pixel_j)
        shortest_path_image, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))
        shortest_path_image /= LOCAL_MAP_PIXELS_PER_METER
        global_map = np.minimum(global_map, shortest_path_image)
        global_map /= (np.sqrt(2) * LOCAL_MAP_PIXEL_WIDTH) / LOCAL_MAP_PIXELS_PER_METER
        global_map *= self.shortest_path_channel_scale
        return global_map

    def _create_global_shortest_path_map(self, robot_position):
        pixel_i, pixel_j = position_to_pixel_indices(robot_position[0], robot_position[1], self.configuration_space.shape)
        pixel_i, pixel_j = self._closest_valid_cspace_indices(pixel_i, pixel_j)
        global_map, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))
        global_map /= LOCAL_MAP_PIXELS_PER_METER
        global_map /= (np.sqrt(2) * LOCAL_MAP_PIXEL_WIDTH) / LOCAL_MAP_PIXELS_PER_METER
        global_map *= self.shortest_path_channel_scale
        return global_map

    def _get_new_observation(self):
        # Capture images from forward-facing camera
        camera_position = self.robot_position[:2] + [CAMERA_HEIGHT]
        camera_target = [
            camera_position[0] + CAMERA_HEIGHT * np.tan(np.radians(90 + CAMERA_PITCH)) * np.cos(self.robot_heading),
            camera_position[1] + CAMERA_HEIGHT * np.tan(np.radians(90 + CAMERA_PITCH)) * np.sin(self.robot_heading),
            0
        ]
        camera_up = [
            np.cos(np.radians(90 + CAMERA_PITCH)) * np.cos(self.robot_heading),
            np.cos(np.radians(90 + CAMERA_PITCH)) * np.sin(self.robot_heading),
            np.sin(np.radians(90 + CAMERA_PITCH))
        ]
        view_matrix = p.computeViewMatrix(camera_position, camera_target, camera_up)
        images = p.getCameraImage(self.camera_image_pixel_width, self.camera_image_pixel_height, view_matrix, self.projection_matrix)  # tinyrenderer

        # Compute depth
        depth_buffer = np.reshape(images[3], (self.camera_image_pixel_height, self.camera_image_pixel_width))
        depth = CAMERA_FAR * CAMERA_NEAR / (CAMERA_FAR - (CAMERA_FAR - CAMERA_NEAR) * depth_buffer)

        # Construct point cloud
        principal = np.asarray(camera_target) - np.asarray(camera_position)
        principal = principal / np.linalg.norm(principal)
        camera_up = np.asarray(camera_up)
        up = camera_up - np.dot(camera_up, principal) * principal
        up = up / np.linalg.norm(up)
        right = np.cross(principal, up)
        right = right / np.linalg.norm(right)
        points = np.broadcast_to(camera_position, (self.camera_image_pixel_height, self.camera_image_pixel_width, 3))
        limit_y = np.tan(np.radians(CAMERA_FOV / 2))
        limit_x = limit_y * self.camera_aspect
        pixel_x, pixel_y = np.meshgrid(np.linspace(-limit_x, limit_x, self.camera_image_pixel_width), np.linspace(limit_y, -limit_y, self.camera_image_pixel_height))
        points = points + depth[:, :, np.newaxis] * (principal + pixel_y[:, :, np.newaxis] * up + pixel_x[:, :, np.newaxis] * right)

        # Get segmentation
        seg_raw = np.reshape(images[4], (self.camera_image_pixel_height, self.camera_image_pixel_width))
        seg = np.zeros_like(seg_raw, dtype=np.float32)
        seg += FLOOR_SEG_INDEX * (seg_raw == 0)
        seg += OBSTACLE_SEG_INDEX * (seg_raw >= self.min_obstacle_id) * (seg_raw <= self.max_obstacle_id)
        seg += RECEPTACLE_SEG_INDEX * (seg_raw == self.receptacle_id)
        seg += ROBOT_SEG_INDEX * (seg_raw == self.robot_id)
        seg += CUBE_SEG_INDEX * (seg_raw >= self.min_cube_id) * (seg_raw <= self.max_cube_id)
        seg /= MAX_SEG_INDEX

        return points, seg

    def _update_state(self):
        points, seg = self._get_new_observation()

        # Update occupancy map
        augmented_points = np.concatenate((points, np.isclose(seg[:, :, np.newaxis], OBSTACLE_SEG_INDEX / MAX_SEG_INDEX)), axis=2).reshape(-1, 4)
        obstacle_points = augmented_points[np.isclose(augmented_points[:, 3], 1)]
        pixel_i, pixel_j = position_to_pixel_indices(obstacle_points[:, 0], obstacle_points[:, 1], self.occupancy_map.shape)
        self.occupancy_map[pixel_i, pixel_j] = 1
        if self.show_occupancy_map:
            free_space_points = augmented_points[np.isclose(augmented_points[:, 3], 0)]
            pixel_i, pixel_j = position_to_pixel_indices(free_space_points[:, 0], free_space_points[:, 1], self.free_space_map.shape)
            self.free_space_map[pixel_i, pixel_j] = 1

        # Update overhead map
        augmented_points = np.concatenate((points, seg[:, :, np.newaxis]), axis=2).reshape(-1, 4)
        augmented_points = augmented_points[np.argsort(-augmented_points[:, 2])]
        pixel_i, pixel_j = position_to_pixel_indices(augmented_points[:, 0], augmented_points[:, 1], self.global_overhead_map.shape)
        self.global_overhead_map[pixel_i, pixel_j] = augmented_points[:, 3]

        # Update configuration space
        selem = disk(np.floor(ROBOT_RADIUS * LOCAL_MAP_PIXELS_PER_METER))
        self.configuration_space = 1 - np.maximum(self.wall_map, binary_dilation(self.occupancy_map, selem).astype(np.uint8))
        selem_thin = disk(np.floor(ROBOT_HALF_WIDTH * LOCAL_MAP_PIXELS_PER_METER))
        self.configuration_space_thin = 1 - binary_dilation(np.minimum(1 - self.wall_map, self.occupancy_map), selem_thin).astype(np.uint8)
        self.closest_cspace_indices = distance_transform_edt(1 - self.configuration_space, return_distances=False, return_indices=True)

    def _get_local_overhead_map(self):
        # Note: Can just use _get_local_map. Keeping this here only for reproducibility since it gives slightly different outputs.
        rotation_angle = np.degrees(self.robot_heading) - 90
        pos_y = int(np.floor(self.global_overhead_map.shape[0] / 2 - self.robot_position[1] * LOCAL_MAP_PIXELS_PER_METER))
        pos_x = int(np.floor(self.global_overhead_map.shape[1] / 2 + self.robot_position[0] * LOCAL_MAP_PIXELS_PER_METER))
        mask = rotate_image(np.zeros((LOCAL_MAP_PIXEL_WIDTH, LOCAL_MAP_PIXEL_WIDTH), dtype=np.float32), rotation_angle, order=0)
        y_start = pos_y - int(mask.shape[0] / 2)
        y_end = y_start + mask.shape[0]
        x_start = pos_x - int(mask.shape[1] / 2)
        x_end = x_start + mask.shape[1]
        crop = self.global_overhead_map[y_start:y_end, x_start:x_end]
        crop = rotate_image(crop, -rotation_angle, order=0)
        y_start = int(crop.shape[0] / 2 - LOCAL_MAP_PIXEL_WIDTH / 2)
        y_end = y_start + LOCAL_MAP_PIXEL_WIDTH
        x_start = int(crop.shape[1] / 2 - LOCAL_MAP_PIXEL_WIDTH / 2)
        x_end = x_start + LOCAL_MAP_PIXEL_WIDTH
        return crop[y_start:y_end, x_start:x_end]

    @staticmethod
    def _get_local_map(global_map, robot_position, robot_heading):
        crop_width = round_up_to_even(LOCAL_MAP_PIXEL_WIDTH * np.sqrt(2))
        rotation_angle = 90 - np.degrees(robot_heading)
        pixel_i = int(np.floor(-robot_position[1] * LOCAL_MAP_PIXELS_PER_METER + global_map.shape[0] / 2))
        pixel_j = int(np.floor(robot_position[0] * LOCAL_MAP_PIXELS_PER_METER + global_map.shape[1] / 2))
        crop = global_map[pixel_i - crop_width // 2:pixel_i + crop_width // 2, pixel_j - crop_width // 2:pixel_j + crop_width // 2]
        rotated_crop = rotate_image(crop, rotation_angle, order=0)
        local_map = rotated_crop[
            rotated_crop.shape[0] // 2 - LOCAL_MAP_PIXEL_WIDTH // 2:rotated_crop.shape[0] // 2 + LOCAL_MAP_PIXEL_WIDTH // 2,
            rotated_crop.shape[1] // 2 - LOCAL_MAP_PIXEL_WIDTH // 2:rotated_crop.shape[1] // 2 + LOCAL_MAP_PIXEL_WIDTH // 2
        ]
        return local_map

    def _get_local_distance_map(self, global_map, robot_position, robot_heading):
        local_map = self._get_local_map(global_map, robot_position, robot_heading)
        local_map -= local_map.min()  # Move the min to 0 to make invariant to size of environment
        return local_map

    def get_state(self, done=False):
        if done:
            return None

        # Overhead map
        channels = []
        channels.append(self._get_local_overhead_map())

        # Robot state
        channels.append(self.robot_state_channel)

        # Additional channels
        if self.use_distance_to_receptacle_channel:
            channels.append(self._get_local_distance_map(self.global_distance_to_receptacle_map, self.robot_position, self.robot_heading))
        if self.use_shortest_path_to_receptacle_channel:
            global_shortest_path_to_receptacle_map = self._create_global_shortest_path_to_receptacle_map()
            channels.append(self._get_local_distance_map(global_shortest_path_to_receptacle_map, self.robot_position, self.robot_heading))
        if self.use_shortest_path_channel:
            global_shortest_path_map = self._create_global_shortest_path_map(self.robot_position)
            channels.append(self._get_local_distance_map(global_shortest_path_map, self.robot_position, self.robot_heading))
        if self.use_position_channel:
            channels.append(self.local_position_map_x)
            channels.append(self.local_position_map_y)

        assert all(channel.dtype == np.float32 for channel in channels)
        return np.stack(channels, axis=2)

    def _shortest_path(self, source_position, target_position, check_straight=False, configuration_space=None):
        if configuration_space is None:
            configuration_space = self.configuration_space

        # Convert positions to pixel indices
        source_i, source_j = position_to_pixel_indices(source_position[0], source_position[1], configuration_space.shape)
        target_i, target_j = position_to_pixel_indices(target_position[0], target_position[1], configuration_space.shape)

        # Check if there is a straight line path
        if check_straight:
            rr, cc = line(source_i, source_j, target_i, target_j)
            if (1 - self.configuration_space_thin[rr, cc]).sum() == 0:
                return [source_position, target_position]

        # Run SPFA
        source_i, source_j = self._closest_valid_cspace_indices(source_i, source_j)  # Note: does not use the cspace passed into this method
        target_i, target_j = self._closest_valid_cspace_indices(target_i, target_j)
        _, parents = spfa.spfa(configuration_space, (source_i, source_j))

        # Recover shortest path
        parents_ij = np.stack((parents // parents.shape[1], parents % parents.shape[1]), axis=2)
        parents_ij[parents < 0, :] = [-1, -1]
        i, j = target_i, target_j
        coords = [[i, j]]
        while not (i == source_i and j == source_j):
            i, j = parents_ij[i, j]
            if i + j < 0:
                break
            coords.append([i, j])

        # Convert dense path to sparse path (waypoints)
        coords = approximate_polygon(np.asarray(coords), tolerance=1)

        # Remove unnecessary waypoints
        new_coords = [coords[0]]
        for i in range(1, len(coords) - 1):
            rr, cc = line(*new_coords[-1], *coords[i + 1])
            if (1 - configuration_space[rr, cc]).sum() > 0:
                new_coords.append(coords[i])
        if len(coords) > 1:
            new_coords.append(coords[-1])
        coords = new_coords

        # Convert pixel indices back to positions
        path = []
        for coord in coords[::-1]:
            position_x, position_y = pixel_indices_to_position(coord[0], coord[1], configuration_space.shape)
            path.append([position_x, position_y, 0])

        if len(path) < 2:
            path = [source_position, target_position]
        else:
            path[0] = source_position
            path[-1] = target_position

        return path

    def _shortest_path_distance(self, source_position, target_position, configuration_space=None):
        path = self._shortest_path(source_position, target_position, configuration_space=configuration_space)
        return sum(distance(path[i - 1], path[i]) for i in range(1, len(path)))

    def _closest_valid_cspace_indices(self, i, j):
        return self.closest_cspace_indices[:, i, j]

class RealEnvironment(Environment):
    CUBE_REMOVAL_THRESHOLD = 2
    REMOVED_BODY_Z = -1000

    def __init__(self, robot_index, cube_indices, **kwargs):
        super().__init__(**kwargs)
        self.robot_index = robot_index
        self.cube_indices = cube_indices
        assert len(self.cube_indices) == self.num_cubes
        self.robot_requires_reset = True
        self.cube_removal_counter = {cube_index: 0 for cube_index in self.cube_indices}

    def update_poses(self, data):
        if data['cubes'] is not None:
            for cube_index, cube_id in zip(self.cube_indices, self.cube_ids):
                pose = data['cubes'].get(cube_index)
                if pose is None or self._cube_position_in_receptacle(pose['position']):
                    self.cube_removal_counter[cube_index] += 1
                    if self.cube_removal_counter[cube_index] > self.CUBE_REMOVAL_THRESHOLD:
                        p.resetBasePositionAndOrientation(cube_id, [0, 0, self.REMOVED_BODY_Z], [0, 0, 0, 1])
                else:
                    self.cube_removal_counter[cube_index] = 0
                    p.resetBasePositionAndOrientation(cube_id, pose['position'] + [CUBE_WIDTH / 2], p.getQuaternionFromEuler([0, 0, pose['heading']]))

        if data['robots'] is not None:
            pose = data['robots'].get(self.robot_index)
            if pose is not None:
                robot_position = pose['position'] + [0]
                robot_orientation = p.getQuaternionFromEuler([0, 0, pose['heading']])
                if self.robot_requires_reset:
                    p.resetBasePositionAndOrientation(self.robot_id, robot_position, robot_orientation)
                    self.robot_requires_reset = False
                p.changeConstraint(self.robot_cid, jointChildPivot=robot_position, jointChildFrameOrientation=robot_orientation, maxForce=500)

        self._step_simulation_until_still()
        self.robot_position, self.robot_heading = self._get_robot_pose()
        self._update_state()

    def try_action(self, action):
        target_end_effector_position = self._step(action, dry_run=True)
        result = {'target_end_effector_position': target_end_effector_position[:2]}
        return result

    def _reset_poses(self):
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, self.REMOVED_BODY_Z], [0, 0, 0, 1])
        self.robot_requires_reset = True
        for cube_id in self.cube_ids:
            p.resetBasePositionAndOrientation(cube_id, [0, 0, self.REMOVED_BODY_Z], [0, 0, 0, 1])

################################################################################
# Helper functions

def round_up_to_even(x):
    return int(2 * np.ceil(x / 2))

def distance(position1, position2):
    return np.linalg.norm(np.asarray(position1)[:2] - np.asarray(position2)[:2])

def orientation_to_heading(orientation):
    # Note: only works for z-axis rotations
    return 2 * np.arccos(np.sign(orientation[2]) * orientation[3])

def restrict_heading_range(heading):
    return np.mod(heading + np.pi, 2 * np.pi) - np.pi

def heading_difference(heading1, heading2):
    return restrict_heading_range(heading2 - heading1)

def position_to_pixel_indices(position_x, position_y, image_shape):
    pixel_i = np.floor(image_shape[0] / 2 - position_y * LOCAL_MAP_PIXELS_PER_METER).astype(np.int32)
    pixel_j = np.floor(image_shape[1] / 2 + position_x * LOCAL_MAP_PIXELS_PER_METER).astype(np.int32)
    pixel_i = np.clip(pixel_i, 0, image_shape[0] - 1)
    pixel_j = np.clip(pixel_j, 0, image_shape[1] - 1)
    return pixel_i, pixel_j

def pixel_indices_to_position(pixel_i, pixel_j, image_shape):
    position_x = (pixel_j - image_shape[1] / 2) / LOCAL_MAP_PIXELS_PER_METER
    position_y = (image_shape[0] / 2 - pixel_i) / LOCAL_MAP_PIXELS_PER_METER
    return position_x, position_y
