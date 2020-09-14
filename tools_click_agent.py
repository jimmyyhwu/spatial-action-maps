import cv2
import numpy as np
#import pybullet as p

import environment
import utils


class ClickAgent:
    def __init__(self, env):
        self.env = env
        self.window_name = 'window'
        self.reward_img_height = 12
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.selected_action = None

    def mouse_callback(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_action = (max(0, y - self.reward_img_height), x)

    def update_display(self, state, last_reward, last_ministeps):
        reward_img = utils.get_reward_img(last_reward, last_ministeps, self.reward_img_height, self.env.get_state_width())
        state_img = utils.get_state_visualization(state)[:, :, ::-1]
        cv2.imshow(self.window_name, np.concatenate((reward_img, state_img), axis=0))

    def run(self):
        state = self.env.reset()
        last_reward = 0
        last_ministeps = 0

        done = False
        force_reset_env = False
        while True:
            self.update_display(state, last_reward, last_ministeps)

            # Read keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                force_reset_env = True
            elif key == ord('q'):
                break

            if self.selected_action is not None:
                action = self.selected_action[0] * self.env.get_state_width() + self.selected_action[1]
                state, reward, done, info = self.env.step(action)
                last_reward = reward
                last_ministeps = info['ministeps']
                self.selected_action = None
            else:
                #p.stepSimulation()  # Uncomment to make pybullet window interactive
                pass

            if done or force_reset_env:
                state = self.env.reset()
                done = False
                force_reset_env = False
                last_reward = 0
                last_ministeps = 0
        cv2.destroyAllWindows()

def main():
    # Obstacle config
    obstacle_config = 'small_empty'
    #obstacle_config = 'small_columns'
    #obstacle_config = 'large_columns'
    #obstacle_config = 'large_divider'

    # Room config
    kwargs = {}
    kwargs['room_width'] = 1 if obstacle_config.startswith('large') else 0.5
    kwargs['num_cubes'] = 20 if obstacle_config.startswith('large') else 10
    kwargs['obstacle_config'] = obstacle_config
    #kwargs['random_seed'] = 0

    # Visualization
    kwargs['use_gui'] = True
    kwargs['show_debug_annotations'] = True
    #kwargs['show_occupancy_map'] = True

    # Shortest path components
    #kwargs['use_distance_to_receptacle_channel'] = False
    #kwargs['use_shortest_path_to_receptacle_channel'] = True
    #kwargs['use_shortest_path_channel'] = True
    #kwargs['use_shortest_path_partial_rewards'] = True
    #kwargs['use_shortest_path_movement'] = True

    env = environment.Environment(**kwargs)
    agent = ClickAgent(env)
    agent.run()
    env.close()

main()
