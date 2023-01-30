"""
Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""

import argparse
import imageio
import numpy as np

from robosuite import make
from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv

import robosuite as suite
from robosuite.wrappers import GymWrapper
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from robosuite import load_controller_config


def evaluate(model: "base_class.BaseAlgorithm",
             env: Union[gym.Env, VecEnv],
            n_eval_episodes: int = 10,
            deterministic: bool = True,
            render: bool = False,
            callback: Optional[Callable] = None,
            reward_threshold: Optional[float] = None,
            return_episode_rewards: bool = False,
        ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    :return: Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="PegInHole")
    parser.add_argument("--robots", nargs="+", type=str, default="UR5e", help="Which robot(s) to use in the env")
    parser.add_argument("--camera", type=str, default="sideview", help="Name of camera to render") #robot0_eye_in_hand,robot0_robotview
    parser.add_argument("--video_path", type=str, default="video_8.mp4")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    args = parser.parse_args()

    control_param = dict(type='IMPEDANCE_PB', input_max=1, input_min=-1, output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                         output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], kp=150, damping_ratio=1,
                         impedance_mode='fixed', kp_limits=[0, 300], damping_ratio_limits=[0, 10], position_limits=None,
                         orientation_limits=None, uncouple_pos_ori=True, control_delta=True, interpolation=None,
                         ramp_ratio=0.2)
    # initialize an environment with offscreen renderer
    env = make(
    # env = GymWrapper(
    #     suite.make(
        args.environment,
        args.robots,
        has_renderer=False,
        ignore_done=False,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        # use_camera_obs=False,  # do not use pixel observations
        # has_offscreen_renderer=False,  # not needed since not using pixel obs
        # has_renderer=True,  # make sure we can render to the screen
        reward_shaping=True,  # use dense rewards
        control_freq=20,  # control should happen fast enough so that simulation looks smooth
        controller_configs=control_param
    )
    # )
    # env = GymWrapper(
    #     suite.make(
    #         "PegInHole",
    #         robots="UR5e",  # use UR5e robot
    #         use_camera_obs=False,  # do not use pixel observations
    #         has_offscreen_renderer=False,  # not needed since not using pixel obs
    #         has_renderer=True,  # make sure we can render to the screen
    #         reward_shaping=True,  # use dense rewards
    #         control_freq=20,  # control should happen fast enough so that simulation looks smooth
    #         controller_configs=control_param
    #     )
    # )

    obs = env.reset()
    ndim = env.action_dim

    # create a video writer with imageio
    writer = imageio.get_writer(args.video_path, fps=20)

    frames = []
    for i in range(args.timesteps):

        # run a uniformly random agent
        action = 0.5 * np.random.randn(ndim)
        obs, reward, done, info = env.step(action)

        # dump a frame from every K frames
        if i % args.skip_frame == 0:
            frame = obs[args.camera + "_image"][::-1]
            writer.append_data(frame)
            print("Saving frame #{}".format(i))

        if done:
            break

    writer.close()
