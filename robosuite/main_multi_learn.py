import time
from typing import Callable, List, Optional, Tuple, Union
import os
import gym
import numpy as np
import random

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

import torch
import robosuite as suite
from robosuite.wrappers import GymWrapper


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

class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``
    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.
        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True
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

    def _on_step(self) -> bool:
        # n_call is incremented every n_process steps/ episodes
        print(self.num_timesteps)
        if self.n_calls % self.check_freq == 0:
            print('---Callback---')

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-self.mean_eps:])
                if self.verbose > 0:
                    print(f'Evaluating model at episode: {self.num_timesteps}')
                    print(f"Current mean reward over last {self.mean_eps} episodes is: {mean_reward}")
                    print(f"Previous best mean reward was {self.best_mean_reward}")
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    self.temp['Max Reward'].append(self.best_mean_reward)
                    self.temp['Episode'].append(self.num_timesteps)

                    path = os.path.join(self.save_path, f'best_model_{self.num_timesteps}')
                    path2 = os.path.join(self.save_path, 'callback_best_runs')
                    print("Saving new best model to {}".format(path))
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
    actions = []
    episode_rewards, episode_lengths = [], []
    episode_success = 0
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
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_success += int(_info.get('is_success'))
        print(f"episode number: {i},reward: {episode_reward}, episode lenght: {_info.get('time')} ")
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    # a = np.array(actions)
    # print(a.shape)
    # np.savetxt('action_matrix.csv', a, delimiter=',')
    # print('Saved CSV')
    return mean_reward, std_reward, episode_success


def make_robosuite_env(env_id, options, rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param options: additional arguments to pass to the specific env class initializer
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = GymWrapper(suite.make(env_id, **options))
        monitor_path = os.path.join(log_dir_callback, str(rank + 1)) if log_dir_callback is not None else None
        env = Monitor(env, filename=monitor_path)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def model_info_collect(model):
    init_weights = model.get_parameters()
    dict_csv(name=os.path.join(log_dir_extras, 'init_weights'), dict=init_weights)
    learn_info = {}
    learn_info['Num of envs'] = num_proc
    learn_info['Initialization seed of NN'] = model.seed
    learn_info['N_steps'] = model.n_steps
    learn_info['Training steps'] = learning_steps
    learn_info['Policy_kwargs'] = model.policy_kwargs
    learn_info['Policy'] = model.policy

    dict_csv(name=os.path.join(log_dir_extras, 'model'), dict=learn_info)
    return


def seed_initializer():
    '''
    Function used to define seed as well as to save it in csv file:
        Returns:
            seed:
    '''
    seed = random.randint(0, 1000)
    print('Seed used', seed)
    return seed


if __name__ == "__main__":
    # Create log dir
    log_dir = './robosuite/'
    log_dir_extras = os.path.join(log_dir, 'extras')
    log_dir_callback = os.path.join(log_dir, 'callback')
    log_dir_checkpoint = os.path.join(log_dir, 'checkpoint')
    os.makedirs(log_dir_checkpoint, exist_ok=True)
    os.makedirs(log_dir_callback, exist_ok=True)
    os.makedirs(log_dir_extras, exist_ok=True)

    use_spiral = False
    use_impedance = True
    plot_graphs = False
    render = False
    error_type = "ring"

    # shir
    total_sim_time = 25 # + 20  # + 20
    time_free_space = 2.5
    time_insertion = 13.5
    # daniel - long
    # total_sim_time = 35.0
    # time_free_space = 5
    # time_insertion = 25.0
    # elad
    #     total_sim_time = 15.0
    #     time_free_space = 5.0
    #     time_insertion = 4.0
    time_impedance = total_sim_time - (time_free_space + time_insertion)

    control_freq = 20
    control_dim = 26 # 26, 38
    horizon = total_sim_time * control_freq
    control_param = dict(type='IMPEDANCE_POSE_Partial', input_max=1, input_min=-1,
                         output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                         output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], kp=700, damping_ratio=np.sqrt(2),
                         impedance_mode='fixed', kp_limits=[0, 100000], damping_ratio_limits=[0, 10],
                         position_limits=None, orientation_limits=None, uncouple_pos_ori=True, control_delta=True,
                         interpolation=None, ramp_ratio=0.2, control_dim=control_dim, ori_method='rotation',
                         show_params=False, total_time=total_sim_time, plotter=plot_graphs, use_impedance=use_impedance,
                         use_spiral=use_spiral)

    env_id = 'PegInHoleSmall'

    env_options = dict(robots="UR5e", use_camera_obs=False, has_offscreen_renderer=False, has_renderer=render,
                       reward_shaping=True, ignore_done=False, plot_graphs=plot_graphs, horizon=horizon,
                       time_free=time_free_space, time_insertion=time_insertion, control_freq=control_freq,
                       controller_configs=control_param, r_reach_value=0.2, tanh_value=20.0, error_type=error_type,
                       control_spec=control_dim, dist_error=0.0008)

    mini_buffer = 20
    seed_val = 2
    num_proc = 10
    learning_steps = 10_000
    seed = 4#  seed_initializer()
    # seed = 4
    assert mini_buffer >= num_proc, f"Number of mini_buffer >= num_proc, but is n_step:{mini_buffer}, num_proc: {num_proc}"

    check_callback_every_eps = 20
    n_call = int(check_callback_every_eps / num_proc)  # how many times' callback will be called (every x number of episodes)
    print(n_call)

    # rollout buffer size = n_steps * num_proc
    reward_callback = SaveOnBestTrainingRewardCallback(mean_eps=20, check_freq=n_call, log_dir=log_dir_callback)
    # Each call to env.step() will effectively correspond to n_envs steps.
    checkpoint_callback_ever_eps = 500
    checkpoint_freq = int(checkpoint_callback_ever_eps / num_proc)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=log_dir_checkpoint,
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=2)
    env = SubprocVecEnv([make_robosuite_env(env_id, env_options, i, seed_val) for i in range(num_proc)])

    policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU, net_arch=[32, 32])
    model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, n_steps=int(mini_buffer / num_proc),
                tensorboard_log="./learning_log/ppo_tensorboard/", seed=seed)

    model_info_collect(model=model)
    t_start = time.time()
    model.learn(total_timesteps=learning_steps, tb_log_name="learning", callback=[reward_callback, checkpoint_callback])
    model.save('final13.zip')
    print("Model Saved")
    print('Total learning time:', time.time()-t_start)
    #
    # print('Training Continuation')
    # model = PPO.load("./daniel_n8_sim/sim7_n8/Multiprocess_32_32_nstep_20_pd_lear.zip",
    #                  tensorboard_log="./learning_log/ppo_tensorboard/", verbose=1, env=env)
    # model.set_env(env)
    # model.learn(total_timesteps=learning_steps, tb_log_name="learning", callback=reward_callback, reset_num_timesteps=False)
    # model.save('Multiprocess_32_32_nstep_20_pd_lear_part2.zip')
    # print("------------ Done Retraining -------------")