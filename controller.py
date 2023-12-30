import ray
from ray import tune
import numpy as np

# # Custom Callbacks

from typing import Dict
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune import CLIReporter

import gym
from gym.spaces import Discrete, Box

import PIL
from PIL import ImageGrab, ImageOps, Image
import keyboard
import cv2
from ray.rllib.models import ModelCatalog
from fcnet import FullyConnectedNetwork
import time
import scipy.misc
import matplotlib


# My custom callbacks class, where I define custom metrics to visualize actions taken and episode rewards.
# As in:
# > https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py

# # Custom Environment
#
# A simple environment that returns a reward of:
#
# * $0.2 * Action$ if in 'train' mode
# * -$0.2 * Action$ if in 'test' mode (evaluation mode)
#
# $Action \in \{0,1,2,3,4\}$
#
# For the sake of completeness, observation at each step is only the episode step number.


class ControllerEnv(gym.Env):
    """Simple Env"""

    def __init__(self, config):
        print("Initializing controller")
        self.config = config
        self.mode = self.config["mode"]
        self.reload_template = self.config["reload_template"]
        self.nA = 3

        # Canvas position
        # From x_left to x_right
        self.top_left_x = 1920 + 550
        self.bottom_right_x = 3600 - 490

        # From y_top to y_bottom
        self.top_left_y = 150
        self.bottom_right_y = 310

        width = self.bottom_right_x - self.top_left_x
        height = self.bottom_right_y - self.top_left_y

        assert width > 0, "x_left should be less than x_right"
        assert height > 0, "y_TOP should be LESS than y_BOTTOM: (0,0) is at the top left corner!"

        # Setting the points for cropped image
        self.left = 0
        self.top = 0#int(height * 3 / 4)
        self.right = width
        self.bottom = height#int(height * 3 / 4) + 1


        # Observation stats
        resize_scale = 0.2625  # (0, 1]
        obs_width = max(1, int((self.right - self.left) * resize_scale))
        obs_height = 1#max(1, int((self.bottom - self.top) * resize_scale))

        print(" >>> obs_width x obs_height:", obs_width, obs_height)

        # To separate too wide or too tall images
        self.image_parts = 1
        self.num_stacking_frames = 1
        self.obsDim = (obs_height, obs_height, self.image_parts * self.num_stacking_frames)  # Needs to be tuple.
        self.action_space = Discrete(self.nA)
        self.observation_space = Box(low=-10, high=10, shape=self.obsDim, dtype=np.float32)

        self.count = 0
        self.done = False

        self.last_action = 0
        self.last_observation = np.zeros((obs_width, obs_height))
        self.last_n_observations = np.zeros(self.obsDim)


        self.num_sticking_frames = 8
        self.stick_countdown = self.num_sticking_frames
        self.stack_countdown = self.num_stacking_frames
        # To keep order of frames
        self.count_indexes = np.zeros(self.num_stacking_frames)

        self.file_num = 0

    def step(self, action):
        if self.done:
            return np.zeros(self.obsDim), 0, self.done, {}

        self.count += 1
        self.stick_countdown -= 1
        self.stack_countdown -= 1

        sticked = self.stick_countdown % self.num_sticking_frames != 0
        if sticked:
            # Stick to previous action
            action = self.last_action
        else:
            # Save new action to stick later
            self.last_action = action

            # Release previous key
            self.release_keys()

            if action == 1:
                # Jump
                keyboard.press('up')
            elif action == 2:
                # Hide
                keyboard.press('down')

        full_stack = self.stack_countdown % self.num_stacking_frames == 0
        if full_stack:
            # Get new observation and update past one
            obs_state = self.get_obs()
            self.last_observation = obs_state[-1]
        else:
            # Keep seeing previous observation
            obs_state = self.last_observation


        # if self.mode == 'train':
        #     # Best strategy: Choose bigger action
        #     reward = 0.2 * action
        # elif self.mode == 'test':
        #     # Best strategy: Choose smaller action
        #     reward = -0.2 * action
        if self.done:
            print("> Dead! Count: {}".format(self.count))
            reward = -0.1

            # Release previous key
            self.release_keys()

            # Restart game
            keyboard.press_and_release('space')
        else:
            if not sticked:
                reward = 0 #+ int(action == 0)
            else:
                # Do not give reward for observations in sticked actions
                reward = 0 # 0

        # Pause
        #time.sleep(0.2)

        return obs_state, reward, self.done, {}

    def get_obs(self):
        pil_screenshot_for_reload = ImageGrab.grab(bbox=(self.top_left_x,     self.top_left_y,
                                         self.bottom_right_x, self.bottom_right_y))

        width, height = pil_screenshot_for_reload.size

        # Cropped image of above dimension
        # (It will not change orginal image)
        pil_screenshot_for_observation = pil_screenshot_for_reload#.crop((self.left, self.top, self.right, self.bottom))

        # Convert to opencv to use template search. TODO: Check if could use only OpenCV (no PIL)
        open_cv_image = cv2.cvtColor(np.array(pil_screenshot_for_reload), cv2.COLOR_RGB2GRAY)

        # Check if dead (search for reload logo)
        self.done = self.check_if_dead(open_cv_image)

        open_cv_image = resize_cv2(open_cv_image)
        # res = cv2.imwrite("./img/img-{}.png".format(time.time()), open_cv_image)
        # print("> > > res:", res)


        pil_screenshot_for_observation = ImageOps.grayscale(pil_screenshot_for_observation)
        pil_screenshot_for_observation = ImageOps.autocontrast(pil_screenshot_for_observation)
        pil_screenshot_for_observation = resize_pil(pil_screenshot_for_observation)

        width, height = pil_screenshot_for_observation.size
        #print(pil_screenshot_for_observation.size)
        obs_h = int(11/16 * height)
        pil_screenshot_for_observation = pil_screenshot_for_observation.crop((0, obs_h, width, obs_h+1))
        #print(pil_screenshot_for_observation.size)
        # Save file .png
        if self.done:
            #pil_screenshot_for_observation.save("../../dead-{}.png".format(self.file_num))
            pass
        else:
            pass
        self.file_num += 1

        # For debugging
        show_input_img = False
        # if show_input_img and self.file_num < 10:
        #     #pil_screenshot_for_observation.show()
        #     pass
        # elif self.file_num > 10:
        #     ray.shutdown()
        #     assert False, "This only works for 10 imgs, to not have many windows being opened"

        # Reshape / downsample observed image
        obs_h = self.obsDim[0]
        obs_w = self.obsDim[1]
        #assert obs_w == 4*obs_h, "Width should be EXACTLY 4 times the height. Now is: w={} x h={}".format(obs_w,obs_h)
        #obs_state = np.reshape(np.array(pil_screenshot_for_observation.getdata()), (obs_w, obs_h))
        obs_state = np.array(pil_screenshot_for_observation.getdata())
        # Normalize
        obs_state = obs_state / 255.

        return obs_state

    def release_keys(self):
        if keyboard.is_pressed('up'): keyboard.release('up')
        if keyboard.is_pressed('down'): keyboard.release('down')

    def check_if_dead(self, obs) -> bool:
        # pil_image = PIL.Image.open("reload.png")
        #
        # open_cv_template = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        # template = cv2.imread("reload.bmp", 0) # 0: To grayscale
        # print("\n\n\n\nTemplate:",template)
        # print("\n\n\n\n")
        res = cv2.matchTemplate(obs, self.reload_template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        # loc = np.where(res >= threshold)

        return np.sum(res >= threshold) > 0

    def reset(self):
        # Release previous key (if any)
        self.release_keys()

        self.count = 0
        self.done = False
        self.stick_countdown = self.num_sticking_frames


        obs_state = self.get_obs()

        if self.done:
            # Restart game
            keyboard.press_and_release('space')

        return obs_state

def resize_cv2(img):
    scale = 0.2625
    width = max(1, int(img.shape[1] * scale)) # At least 1 pixel
    height = max(1, int(img.shape[0] * scale))
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def resize_pil(img):
    scale = 0.2625
    width, height = img.size
    width = max(1, int(width * scale)) # At least 1 pixel
    height = max(1, int(height * scale))
    resized = img.resize((width, height), PIL.Image.LANCZOS)
    return resized

if __name__ == "__main__":
    num_cpus = 8
