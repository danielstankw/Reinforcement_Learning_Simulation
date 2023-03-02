import csv
import time
from typing import Callable, List, Optional, Tuple, Union
import os
import gym
import numpy as np
import random

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

import torch
import robosuite as suite
from robosuite.wrappers import GymWrapper
import pickle

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1, mean_eps=100):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model_callback')
        self.best_mean_reward = -np.inf
        self.mean_eps = mean_eps
        self.temp = {"Max Reward": [], "Episode": []}

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        assert self.check_freq >= self.mean_eps, "Check freq needs to be larger than mean_eps"

    def _on_step(self) -> bool:
        a = self.model
        if self.n_calls % self.check_freq == 0:
            print('----CallBack----')
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-self.mean_eps:])
                if self.verbose > 0:
                    print(f'Evaluating model at episode: {self.num_timesteps}')
                    print(f"Current mean reward over last {self.mean_eps} episodes is: {mean_reward}")
                    print(f"Previous best mean reward was {self.best_mean_reward}")
                # New best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    self.temp['Max Reward'].append(self.best_mean_reward)
                    self.temp['Episode'].append(self.num_timesteps)

                    path = os.path.join(self.save_path, f'best_model_{self.num_timesteps}')
                    print("Saving new best model to {}".format(path))
                    path2 = os.path.join(self.save_path, 'callback_best_runs')
                    # save every best episode and reward to csv file
                    dict_csv(path2, self.temp)
                    # save every model when it is best for the current point of simulation
                    self.model.save(path)
                    # save one final best model
                    self.model.save(self.save_path)

        return True


def evaluate(model: "base_class.BaseAlgorithm",
             env: Union[gym.Env, VecEnv],
             n_eval_episodes: int = 10,
             deterministic: bool = True,
             render: bool = True,
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
    global _info, obs
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"
    folder_name = "data_collection"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    actions = []
    episode_rewards, episode_lengths = [], []
    episode_success = 0
    # my_dict = {'success': [], 'error': []}
    my_dict = {'error': []}
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            actions.append(action)
            # TODO this is how i pass number of episode!
            action = np.ones(control_dim)* (i+1)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        length = sum(episode_lengths)
        episode_success += int(_info.get('is_success'))
        print(f"Success rate of episode {i+1}: {episode_success/length*100}%")
        print(f"Episode number: {i+1},reward: {episode_reward}, sim time: {np.round(_info.get('time'),7)} "
              f"| horizon: {_info.get('horizon')} | real time {_info.get('episode').get('t')} ")

        my_dict['error'].append(_info.get('error'))
        file_name = '100_simulation_errors.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(my_dict, f)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward, episode_success


def dict_csv(name, dict):
    '''
    Function saving dictionary as csv file:
        Parameters:
            name (str): name of the created csv file
            dict (dict): dictionary to be saved
    '''
    file_name = name + '.csv'
    with open(file_name, 'w') as f:
        for key in dict.keys():
            f.write("%s,%s\n" % (key, dict[key]))
    return


if __name__ == "__main__":
    # Create log dir
    log_dir = './robosuite/'
    log_dir_extras = os.path.join(log_dir, 'extras')
    log_dir_callback = os.path.join(log_dir, 'callback')
    os.makedirs(log_dir_callback, exist_ok=True)
    os.makedirs(log_dir_extras, exist_ok=True)

    plot_graphs = False
    render = False
    error_type = "ring"
    error_vec = np.array([0.0, -3.6, 0.0])/1000  # in mm
    circle_motion = False  # parameters of circle motion are in the controller file

    # RANDOMIZATION OF PARAMS
    # 1. hole position
    # 2. use ring error with appropriate boundaries
    # 3. use pd randomization and find threshold
    # 4. Spiral
    # - linear velocity (v)
    # - pitch (p)
    # 5. Circle
    # - time per circle -> theta\dot
    # - radius (r)

    total_sim_time = 25 + 60
    time_free_space = 2.5
    time_insertion = 13.5

    """
    For collection:
    base.py: uncomment the data saving
    main: set action to ep number
    env: uncomment reward function/ randomization of hole pose
    """

    control_freq = 20
    control_dim = 26 # 38 # 32
    horizon = total_sim_time * control_freq

    control_param = dict(type="IMPEDANCE_SPIRAL_LABEL_COLLECTION", input_max=1, input_min=-1,
                         output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                         output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], damping_ratio=np.sqrt(2),
                         impedance_mode='fixed', kp_limits=[0, 100000], damping_ratio_limits=[0, 10],
                         position_limits=None, orientation_limits=None, uncouple_pos_ori=True, control_delta=True,
                         interpolation=None, ramp_ratio=0.2, control_dim=control_dim, ori_method='rotation',
                         show_params=False, total_time=total_sim_time, plotter=plot_graphs,
                         circle=circle_motion)

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "PegInHoleSmall",
            robots="UR5e",  # use UR5e robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=render,  # Make sure we can render to the screen
            reward_shaping=True,
            ignore_done=False,
            plot_graphs=plot_graphs,
            horizon=horizon,
            time_free=time_free_space,
            time_insertion=time_insertion,
            control_freq=control_freq,  # control should happen fast enough so that simulation looks smooth
            controller_configs=control_param,
            r_reach_value=0.2,
            tanh_value=20.0,
            error_type=error_type,
            control_spec=control_dim,
            dist_error=0.8,  # mm
            fixed_error_vec=error_vec  # mm
        )
    )
    eval_steps = 100# how many runs per set of params
    mode = 'eval'

    env = Monitor(env, log_dir_callback, allow_early_resets=True)
    # Create the callback: check every check_freq steps
    reward_callback = SaveOnBestTrainingRewardCallback(mean_eps=100, check_freq=200, log_dir=log_dir_callback)

    if mode == 'eval':
        model = PPO.load("./daniel_n8_sim/sim11_n8/robosuite/callback/best_model_callback.zip", verbose=1, env=env)

    # evaluation
    mean_reward, std_reward, episode_success = evaluate(model, env, n_eval_episodes=eval_steps, render=False)
    print(
        f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f} \nsuccess rate: {episode_success / eval_steps * 100:.1f}")
