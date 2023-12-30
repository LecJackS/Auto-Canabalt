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

class MyCustomCallbacks(DefaultCallbacks):
    # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        episode.hist_data["actions"] = []
        # Action counters per episode
        for i in range(worker.env.nA):
            episode.user_data["actions/action_" + str(i)] = []
            # episode.custom_metrics["action_dummy"] = []

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        if 'agent0' in episode._agent_to_last_action.keys():
            # To visualize actions chosen on tensorboard
            episode.hist_data["actions"].append(episode._agent_to_last_action['agent0'])
            episode.user_data["actions/action_" + str(episode._agent_to_last_action['agent0'])].append(1)

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        # https://docs.ray.io/en/master/rllib-package-ref.html?highlight=MultiAgentEpisode#ray.rllib.evaluation.MultiAgentEpisode
        if worker.env.mode == 'test':
            # Test episode ended, save metrics
            # I want this value (or a mean of several of them) to be used as metric for the checkpoints
            episode.custom_metrics["test_return"] = episode.agent_rewards[('agent0', 'default_policy')]

        for i in range(worker.env.nA):
            episode.custom_metrics["actions/action_" + str(i)] = sum(episode.user_data["actions/action_" + str(i)])

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        # print("returned sample batch of size {}".format(samples.count))
        pass

    def on_train_result(self, trainer, result: dict, **kwargs):
        """Called at the end of Trainable.train().
            Args:
                trainer (Trainer): Current trainer instance.
                result (dict): Dict of results returned from trainer.train() call.
                    You can mutate this object to add additional metrics.
                kwargs: Forward compatibility placeholder.
        """
        # you can mutate the result dict to add new fields to return
        # result["callback_ok"] = True
        # print('trainer.config[\'env\'].mode:', trainer.config['env_config']['mode'])
        # result["test_return"] = 'I would love to add some values here'
        pass


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


class SimpleEnv(gym.Env):
    """Simple Env"""

    def __init__(self, config):
        print("entra a inint")
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
            self.last_observation = obs_state
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
        show_input_img = True
        if show_input_img and self.file_num < 10:
            #pil_screenshot_for_observation.show()
            pass
        elif self.file_num > 10:
            assert False, "This only works for 10 imgs, to not have many windows being opened"

        # Reshape / downsample observed image
        obs_h = self.obsDim[0]
        obs_w = self.obsDim[1]
        #assert obs_w == 4*obs_h, "Width should be EXACTLY 4 times the height. Now is: w={} x h={}".format(obs_w,obs_h)
        #obs_state = np.reshape(np.array(pil_screenshot_for_observation.getdata()), (obs_w, obs_h))
        obs_state = np.array(pil_screenshot_for_observation.getdata())
        # Normalize
        obs_state = obs_state / 255.
        print("obs_state.shape:",obs_state.shape)
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


# In[7]:


# my_env = ControllerEnv(config={'mode':'train'})
#
#
# # Play at random:
#
# # In[8]:
#
#
# action_space = np.arange(0, my_env.nA)
# o = my_env.reset()

# print('First Observation: (shape={})\n{}'.format(o.shape, o))
# for i in range(my_env.episode_length + 3):
#     # Take 13 random actions
#     o, r, d, i = my_env.step(np.random.choice(action_space))
#     print('Obs:{} - Reward:{} - Done:{} - Info:{}'.format(o,r,d,i))


#
#
# Config for `Tune.run()`
#

# In[25]:
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

# Load game over template image
template = cv2.imread("./img/game_over.png", 0)  # 0: To grayscale
assert template is not None, "Game Over image file not found!"
#template = resize_cv2(template)
ModelCatalog.register_custom_model("fcnet", FullyConnectedNetwork)
my_config = {"env": SimpleEnv,
             "callbacks": MyCustomCallbacks,
             "env_config": {'mode': 'train',
                            'reload_template': template},
             # "evaluation_interval": 1,
             # "evaluation_num_episodes": 1,
             "evaluation_config": {"env_config": {'mode': 'test'}},
             "num_gpus": 1,
             "num_workers": 0,
             "learning_starts": 0,
             "model":{
                "custom_model": "fcnet",
                "fcnet_hiddens": [1],
                "conv_filters": None,#[[32, 4, 4], [64, 2, 4], [64, 2, 4]],
                #"post_fcnet_hiddens": [32, 32],

             },
             "n_step": 4,
             "framework": "torch",
             }

stop_def = {
    "training_iteration": 1e5,
    # "timesteps_total": 1e4,
    # "episode_reward_mean": 150,
}

# # Here comes the *issue*
#
# Run default DQN using custom env and custom callbacks.
#
# This shows an error/warning as `custom_metrics/test_return_mean` is not found in the result dict.

# In[27]:

if __name__ == "__main__":
    num_cpus = 8
    try:
        # Shutdown previous ray session
        ray.shutdown()
    except Exception as e:
        pass
    ray.init(num_cpus=num_cpus, num_gpus=1)
    # Limit the number of rows.
    reporter = CLIReporter(max_progress_rows=10)
    results = tune.run("APEX",
                       verbose=0,

                       config=my_config,
                       stop=stop_def,
                       checkpoint_freq=1,
                       checkpoint_at_end=True,
                       sync_on_checkpoint=False,
                       keep_checkpoints_num=3,
                       # checkpoint_score_attr='custom_metrics/test_return_mean'
                       # checkpoint_score_attr='test_return_mean' # Doesn't work either
                       checkpoint_score_attr='training_iteration',
                       local_dir="/home/jack/Auto-Canabalt/results",
                       # name="all"
                       )

# In[28]:


# For debugging error files
# %pycat /root/ray_results/IMPALA/IMPALA_SimpleEnv_1b3ff_00000_0_2020-11-30_22-12-16/error.txt


# Custom metrics are recorded in the results, as shown here (search for `evaluation.custom_metrics.test_return_mean`)

# In[29]:


df = results.results_df
[x for x in list(df.columns) if 'custom' in x]
