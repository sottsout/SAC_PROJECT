# import copy
# from abc import ABC, abstractmethod
# from typing import Iterable, List, Optional
#
# import numpy as np
# from numpy.typing import DTypeLike
#
#
# class ActionNoise(ABC):
#     """
#     The action noise base class
#     """
#
#     def __init__(self) -> None:
#         super().__init__()
#
#     def reset(self) -> None:
#         """
#         Call end of episode reset for the noise
#         """
#         pass
#
#     @abstractmethod
#     def __call__(self) -> np.ndarray:
#         raise NotImplementedError()
#
#
# class NormalActionNoise(ActionNoise):
#     """
#     A Gaussian action noise.
#
#     :param mean: Mean value of the noise
#     :param sigma: Scale of the noise (std here)
#     :param dtype: Type of the output noise
#     """
#
#     def __init__(self, mean: np.ndarray, sigma: np.ndarray, dtype: DTypeLike = np.float32) -> None:
#         self._mu = mean
#         self._sigma = sigma
#         self._dtype = dtype
#         super().__init__()
#
#     def __call__(self) -> np.ndarray:
#         return np.random.normal(self._mu, self._sigma).astype(self._dtype)
#
#     def __repr__(self) -> str:
#         return f"NormalActionNoise(mu={self._mu}, sigma={self._sigma})"
#
#
# class OrnsteinUhlenbeckActionNoise(ActionNoise):
#     """
#     An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.
#
#     Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
#
#     :param mean: Mean of the noise
#     :param sigma: Scale of the noise
#     :param theta: Rate of mean reversion
#     :param dt: Timestep for the noise
#     :param initial_noise: Initial value for the noise output, (if None: 0)
#     :param dtype: Type of the output noise
#     """
#
#     def __init__(
#         self,
#         mean: np.ndarray,
#         sigma: np.ndarray,
#         theta: float = 0.15,
#         dt: float = 1e-2,
#         initial_noise: Optional[np.ndarray] = None,
#         dtype: DTypeLike = np.float32,
#     ) -> None:
#         self._theta = theta
#         self._mu = mean
#         self._sigma = sigma
#         self._dt = dt
#         self._dtype = dtype
#         self.initial_noise = initial_noise
#         self.noise_prev = np.zeros_like(self._mu)
#         self.reset()
#         super().__init__()
#
#     def __call__(self) -> np.ndarray:
#         noise = (
#             self.noise_prev
#             + self._theta * (self._mu - self.noise_prev) * self._dt
#             + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
#         )
#         self.noise_prev = noise
#         return noise.astype(self._dtype)
#
#     def reset(self) -> None:
#         """
#         reset the Ornstein Uhlenbeck noise, to the initial position
#         """
#         self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)
#
#     def __repr__(self) -> str:
#         return f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma})"
#
#
# class VectorizedActionNoise(ActionNoise):
#     """
#     A Vectorized action noise for parallel environments.
#
#     :param base_noise: Noise generator to use
#     :param n_envs: Number of parallel environments
#     """
#
#     def __init__(self, base_noise: ActionNoise, n_envs: int) -> None:
#         try:
#             self.n_envs = int(n_envs)
#             assert self.n_envs > 0
#         except (TypeError, AssertionError) as e:
#             raise ValueError(f"Expected n_envs={n_envs} to be positive integer greater than 0") from e
#
#         self.base_noise = base_noise
#         self.noises = [copy.deepcopy(self.base_noise) for _ in range(n_envs)]
#
#     def reset(self, indices: Optional[Iterable[int]] = None) -> None:
#         """
#         Reset all the noise processes, or those listed in indices.
#
#         :param indices: The indices to reset. Default: None.
#             If the parameter is None, then all processes are reset to their initial position.
#         """
#         if indices is None:
#             indices = range(len(self.noises))
#
#         for index in indices:
#             self.noises[index].reset()
#
#     def __repr__(self) -> str:
#         return f"VecNoise(BaseNoise={self.base_noise!r}), n_envs={len(self.noises)})"
#
#     def __call__(self) -> np.ndarray:
#         """
#         Generate and stack the action noise from each noise object.
#         """
#         noise = np.stack([noise() for noise in self.noises])
#         return noise
#
#     @property
#     def base_noise(self) -> ActionNoise:
#         return self._base_noise
#
#     @base_noise.setter
#     def base_noise(self, base_noise: ActionNoise) -> None:
#         if base_noise is None:
#             raise ValueError("Expected base_noise to be an instance of ActionNoise, not None", ActionNoise)
#         if not isinstance(base_noise, ActionNoise):
#             raise TypeError("Expected base_noise to be an instance of type ActionNoise", ActionNoise)
#         self._base_noise = base_noise
#
#     @property
#     def noises(self) -> List[ActionNoise]:
#         return self._noises
#
#     @noises.setter
#     def noises(self, noises: List[ActionNoise]) -> None:
#         noises = list(noises)  # raises TypeError if not iterable
#         assert len(noises) == self.n_envs, f"Expected a list of {self.n_envs} ActionNoises, found {len(noises)}."
#
#         different_types = [i for i, noise in enumerate(noises) if not isinstance(noise, type(self.base_noise))]
#
#         if len(different_types):
#             raise ValueError(
#                 f"Noise instances at indices {different_types} don't match the type of base_noise", type(self.base_noise)
#             )
#
#         self._noises = noises
#         for noise in noises:
#             noise.reset()
# import io
# import pathlib
# import sys
# import time
# import warnings
# from copy import deepcopy
# from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
#
# import numpy as np
# import torch as th
# from gymnasium import spaces
#
# from stable_baselines3.common.base_class import BaseAlgorithm
# from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
# from stable_baselines3.common.policies import BasePolicy
# from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
# from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
# from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
# from stable_baselines3.common.vec_env import VecEnv
# from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
#
# SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")
#
#
# class OffPolicyAlgorithm(BaseAlgorithm):
#     """
#     The base for Off-Policy algorithms (ex: SAC/TD3)
#
#     :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
#     :param env: The environment to learn from
#                 (if registered in Gym, can be str. Can be None for loading trained models)
#     :param learning_rate: learning rate for the optimizer,
#         it can be a function of the current progress remaining (from 1 to 0)
#     :param buffer_size: size of the replay buffer
#     :param learning_starts: how many steps of the model to collect transitions for before learning starts
#     :param batch_size: Minibatch size for each gradient update
#     :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
#     :param gamma: the discount factor
#     :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
#         like ``(5, "step")`` or ``(2, "episode")``.
#     :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
#         Set to ``-1`` means to do as many gradient steps as steps done in the environment
#         during the rollout.
#     :param action_noise: the action noise type (None by default), this can help
#         for hard exploration problem. Cf common.noise for the different action noise type.
#     :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
#         If ``None``, it will be automatically selected.
#     :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
#     :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
#         at a cost of more complexity.
#         See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
#     :param policy_kwargs: Additional arguments to be passed to the policy on creation
#     :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
#         the reported success rate, mean episode length, and mean reward over
#     :param tensorboard_log: the log location for tensorboard (if None, no logging)
#     :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
#         debug messages
#     :param device: Device on which the code should run.
#         By default, it will try to use a Cuda compatible device and fallback to cpu
#         if it is not possible.
#     :param support_multi_env: Whether the algorithm supports training
#         with multiple environments (as in A2C)
#     :param monitor_wrapper: When creating an environment, whether to wrap it
#         or not in a Monitor wrapper.
#     :param seed: Seed for the pseudo random generators
#     :param use_sde: Whether to use State Dependent Exploration (SDE)
#         instead of action noise exploration (default: False)
#     :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
#         Default: -1 (only sample at the beginning of the rollout)
#     :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
#         during the warm up phase (before learning starts)
#     :param sde_support: Whether the model support gSDE or not
#     :param supported_action_spaces: The action spaces supported by the algorithm.
#     """
#
#     actor: th.nn.Module
#
#     def __init__(
#         self,
#         policy: Union[str, Type[BasePolicy]],
#         env: Union[GymEnv, str],
#         learning_rate: Union[float, Schedule],
#         buffer_size: int = 1_000_000,  # 1e6
#         learning_starts: int = 100,
#         batch_size: int = 256,
#         tau: float = 0.005,
#         gamma: float = 0.99,
#         train_freq: Union[int, Tuple[int, str]] = (1, "step"),
#         gradient_steps: int = 1,
#         action_noise: Optional[ActionNoise] = None,
#         replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
#         replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
#         optimize_memory_usage: bool = False,
#         policy_kwargs: Optional[Dict[str, Any]] = None,
#         stats_window_size: int = 100,
#         tensorboard_log: Optional[str] = None,
#         verbose: int = 0,
#         device: Union[th.device, str] = "auto",
#         support_multi_env: bool = False,
#         monitor_wrapper: bool = True,
#         seed: Optional[int] = None,
#         use_sde: bool = False,
#         sde_sample_freq: int = -1,
#         use_sde_at_warmup: bool = False,
#         sde_support: bool = True,
#         supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
#     ):
#         super().__init__(
#             policy=policy,
#             env=env,
#             learning_rate=learning_rate,
#             policy_kwargs=policy_kwargs,
#             stats_window_size=stats_window_size,
#             tensorboard_log=tensorboard_log,
#             verbose=verbose,
#             device=device,
#             support_multi_env=support_multi_env,
#             monitor_wrapper=monitor_wrapper,
#             seed=seed,
#             use_sde=use_sde,
#             sde_sample_freq=sde_sample_freq,
#             supported_action_spaces=supported_action_spaces,
#         )
#         self.buffer_size = buffer_size
#         self.batch_size = batch_size
#         self.learning_starts = learning_starts
#         self.tau = tau
#         self.gamma = gamma
#         self.gradient_steps = gradient_steps
#         self.action_noise = action_noise
#         self.optimize_memory_usage = optimize_memory_usage
#         self.replay_buffer: Optional[ReplayBuffer] = None
#         self.replay_buffer_class = replay_buffer_class
#         self.replay_buffer_kwargs = replay_buffer_kwargs or {}
#         self._episode_storage = None
#
#         # Save train freq parameter, will be converted later to TrainFreq object
#         self.train_freq = train_freq
#
#         # Update policy keyword arguments
#         if sde_support:
#             self.policy_kwargs["use_sde"] = self.use_sde
#         # For gSDE only
#         self.use_sde_at_warmup = use_sde_at_warmup
#
#     def _convert_train_freq(self) -> None:
#         """
#         Convert `train_freq` parameter (int or tuple)
#         to a TrainFreq object.
#         """
#         if not isinstance(self.train_freq, TrainFreq):
#             train_freq = self.train_freq
#
#             # The value of the train frequency will be checked later
#             if not isinstance(train_freq, tuple):
#                 train_freq = (train_freq, "step")
#
#             try:
#                 train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))  # type: ignore[assignment]
#             except ValueError as e:
#                 raise ValueError(
#                     f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
#                 ) from e
#
#             if not isinstance(train_freq[0], int):
#                 raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")
#
#             self.train_freq = TrainFreq(*train_freq)  # type: ignore[assignment,arg-type]
#
#     def _setup_model(self) -> None:
#         self._setup_lr_schedule()
#         self.set_random_seed(self.seed)
#
#         if self.replay_buffer_class is None:
#             if isinstance(self.observation_space, spaces.Dict):
#                 self.replay_buffer_class = DictReplayBuffer
#             else:
#                 self.replay_buffer_class = ReplayBuffer
#
#         if self.replay_buffer is None:
#             # Make a local copy as we should not pickle
#             # the environment when using HerReplayBuffer
#             replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
#             if issubclass(self.replay_buffer_class, HerReplayBuffer):
#                 assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
#                 replay_buffer_kwargs["env"] = self.env
#             self.replay_buffer = self.replay_buffer_class(
#                 self.buffer_size,
#                 self.observation_space,
#                 self.action_space,
#                 device=self.device,
#                 n_envs=self.n_envs,
#                 optimize_memory_usage=self.optimize_memory_usage,
#                 **replay_buffer_kwargs,
#             )
#
#         self.policy = self.policy_class(
#             self.observation_space,
#             self.action_space,
#             self.lr_schedule,
#             **self.policy_kwargs,
#         )
#         self.policy = self.policy.to(self.device)
#
#         # Convert train freq parameter to TrainFreq object
#         self._convert_train_freq()
#
#     def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
#         """
#         Save the replay buffer as a pickle file.
#
#         :param path: Path to the file where the replay buffer should be saved.
#             if path is a str or pathlib.Path, the path is automatically created if necessary.
#         """
#         assert self.replay_buffer is not None, "The replay buffer is not defined"
#         save_to_pkl(path, self.replay_buffer, self.verbose)
#
#     def load_replay_buffer(
#         self,
#         path: Union[str, pathlib.Path, io.BufferedIOBase],
#         truncate_last_traj: bool = True,
#     ) -> None:
#         """
#         Load a replay buffer from a pickle file.
#
#         :param path: Path to the pickled replay buffer.
#         :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
#             If set to ``True``, we assume that the last trajectory in the replay buffer was finished
#             (and truncate it).
#             If set to ``False``, we assume that we continue the same trajectory (same episode).
#         """
#         self.replay_buffer = load_from_pkl(path, self.verbose)
#         assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"
#
#         # Backward compatibility with SB3 < 2.1.0 replay buffer
#         # Keep old behavior: do not handle timeout termination separately
#         if not hasattr(self.replay_buffer, "handle_timeout_termination"):  # pragma: no cover
#             self.replay_buffer.handle_timeout_termination = False
#             self.replay_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)
#
#         if isinstance(self.replay_buffer, HerReplayBuffer):
#             assert self.env is not None, "You must pass an environment at load time when using `HerReplayBuffer`"
#             self.replay_buffer.set_env(self.env)
#             if truncate_last_traj:
#                 self.replay_buffer.truncate_last_trajectory()
#
#         # Update saved replay buffer device to match current setting, see GH#1561
#         self.replay_buffer.device = self.device
#
#     def _setup_learn(
#         self,
#         total_timesteps: int,
#         callback: MaybeCallback = None,
#         reset_num_timesteps: bool = True,
#         tb_log_name: str = "run",
#         progress_bar: bool = False,
#     ) -> Tuple[int, BaseCallback]:
#         """
#         cf `BaseAlgorithm`.
#         """
#         # Prevent continuity issue by truncating trajectory
#         # when using memory efficient replay buffer
#         # see https://github.com/DLR-RM/stable-baselines3/issues/46
#
#         replay_buffer = self.replay_buffer
#
#         truncate_last_traj = (
#             self.optimize_memory_usage
#             and reset_num_timesteps
#             and replay_buffer is not None
#             and (replay_buffer.full or replay_buffer.pos > 0)
#         )
#
#         if truncate_last_traj:
#             warnings.warn(
#                 "The last trajectory in the replay buffer will be truncated, "
#                 "see https://github.com/DLR-RM/stable-baselines3/issues/46."
#                 "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
#                 "to avoid that issue."
#             )
#             assert replay_buffer is not None  # for mypy
#             # Go to the previous index
#             pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
#             replay_buffer.dones[pos] = True
#
#         assert self.env is not None, "You must set the environment before calling _setup_learn()"
#         # Vectorize action noise if needed
#         if (
#             self.action_noise is not None
#             and self.env.num_envs > 1
#             and not isinstance(self.action_noise, VectorizedActionNoise)
#         ):
#             self.action_noise = VectorizedActionNoise(self.action_noise, self.env.num_envs)
#
#         return super()._setup_learn(
#             total_timesteps,
#             callback,
#             reset_num_timesteps,
#             tb_log_name,
#             progress_bar,
#         )
#
#     def learn(
#         self: SelfOffPolicyAlgorithm,
#         total_timesteps: int,
#         callback: MaybeCallback = None,
#         log_interval: int = 4,
#         tb_log_name: str = "run",
#         reset_num_timesteps: bool = True,
#         progress_bar: bool = False,
#     ) -> SelfOffPolicyAlgorithm:
#         total_timesteps, callback = self._setup_learn(
#             total_timesteps,
#             callback,
#             reset_num_timesteps,
#             tb_log_name,
#             progress_bar,
#         )
#
#         callback.on_training_start(locals(), globals())
#
#         assert self.env is not None, "You must set the environment before calling learn()"
#         assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()
#
#         while self.num_timesteps < total_timesteps:
#             rollout = self.collect_rollouts(
#                 self.env,
#                 train_freq=self.train_freq,
#                 action_noise=self.action_noise,
#                 callback=callback,
#                 learning_starts=self.learning_starts,
#                 replay_buffer=self.replay_buffer,
#                 log_interval=log_interval,
#             )
#
#             if not rollout.continue_training:
#                 break
#
#             if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
#                 # If no `gradient_steps` is specified,
#                 # do as many gradients steps as steps performed during the rollout
#                 gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
#                 # Special case when the user passes `gradient_steps=0`
#                 if gradient_steps > 0:
#                     self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
#
#         callback.on_training_end()
#
#         return self
#
#     def train(self, gradient_steps: int, batch_size: int) -> None:
#         """
#         Sample the replay buffer and do the updates
#         (gradient descent and update target networks)
#         """
#         raise NotImplementedError()
#
#     def _sample_action(
#         self,
#         learning_starts: int,
#         action_noise: Optional[ActionNoise] = None,
#         n_envs: int = 1,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Sample an action according to the exploration policy.
#         This is either done by sampling the probability distribution of the policy,
#         or sampling a random action (from a uniform distribution over the action space)
#         or by adding noise to the deterministic output.
#
#         :param action_noise: Action noise that will be used for exploration
#             Required for deterministic policy (e.g. TD3). This can also be used
#             in addition to the stochastic policy for SAC.
#         :param learning_starts: Number of steps before learning for the warm-up phase.
#         :param n_envs:
#         :return: action to take in the environment
#             and scaled action that will be stored in the replay buffer.
#             The two differs when the action space is not normalized (bounds are not [-1, 1]).
#         """
#         # Select action randomly or according to policy
#         if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
#             # Warmup phase
#             unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
#         else:
#             # Note: when using continuous actions,
#             # we assume that the policy uses tanh to scale the action
#             # We use non-deterministic action in the case of SAC, for TD3, it does not matter
#             assert self._last_obs is not None, "self._last_obs was not set"
#             unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
#
#         # Rescale the action from [low, high] to [-1, 1]
#         if isinstance(self.action_space, spaces.Box):
#             scaled_action = self.policy.scale_action(unscaled_action)
#
#             # Add noise to the action (improve exploration)
#             if action_noise is not None:
#                 scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
#
#             # We store the scaled action in the buffer
#             buffer_action = scaled_action
#             action = self.policy.unscale_action(scaled_action)
#         else:
#             # Discrete case, no need to normalize or clip
#             buffer_action = unscaled_action
#             action = buffer_action
#         return action, buffer_action
#
#     def _dump_logs(self) -> None:
#         """
#         Write log.
#         """
#         assert self.ep_info_buffer is not None
#         assert self.ep_success_buffer is not None
#
#         time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
#         fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
#         self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
#         if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
#             self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
#             self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
#         self.logger.record("time/fps", fps)
#         self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
#         self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
#         if self.use_sde:
#             self.logger.record("train/std", (self.actor.get_std()).mean().item())
#
#         if len(self.ep_success_buffer) > 0:
#             self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
#         # Pass the number of timesteps for tensorboard
#         self.logger.dump(step=self.num_timesteps)
#
#     def _on_step(self) -> None:
#         """
#         Method called after each step in the environment.
#         It is meant to trigger DQN target network update
#         but can be used for other purposes
#         """
#         pass
#
#     def _store_transition(
#         self,
#         replay_buffer: ReplayBuffer,
#         buffer_action: np.ndarray,
#         new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
#         reward: np.ndarray,
#         dones: np.ndarray,
#         infos: List[Dict[str, Any]],
#     ) -> None:
#         """
#         Store transition in the replay buffer.
#         We store the normalized action and the unnormalized observation.
#         It also handles terminal observations (because VecEnv resets automatically).
#
#         :param replay_buffer: Replay buffer object where to store the transition.
#         :param buffer_action: normalized action
#         :param new_obs: next observation in the current episode
#             or first observation of the episode (when dones is True)
#         :param reward: reward for the current transition
#         :param dones: Termination signal
#         :param infos: List of additional information about the transition.
#             It may contain the terminal observations and information about timeout.
#         """
#         # Store only the unnormalized version
#         if self._vec_normalize_env is not None:
#             new_obs_ = self._vec_normalize_env.get_original_obs()
#             reward_ = self._vec_normalize_env.get_original_reward()
#         else:
#             # Avoid changing the original ones
#             self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
#
#         # Avoid modification by reference
#         next_obs = deepcopy(new_obs_)
#         # As the VecEnv resets automatically, new_obs is already the
#         # first observation of the next episode
#         for i, done in enumerate(dones):
#             if done and infos[i].get("terminal_observation") is not None:
#                 if isinstance(next_obs, dict):
#                     next_obs_ = infos[i]["terminal_observation"]
#                     # VecNormalize normalizes the terminal observation
#                     if self._vec_normalize_env is not None:
#                         next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
#                     # Replace next obs for the correct envs
#                     for key in next_obs.keys():
#                         next_obs[key][i] = next_obs_[key]
#                 else:
#                     next_obs[i] = infos[i]["terminal_observation"]
#                     # VecNormalize normalizes the terminal observation
#                     if self._vec_normalize_env is not None:
#                         next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])
#
#         replay_buffer.add(
#             self._last_original_obs,  # type: ignore[arg-type]
#             next_obs,  # type: ignore[arg-type]
#             buffer_action,
#             reward_,
#             dones,
#             infos,
#         )
#
#         self._last_obs = new_obs
#         # Save the unnormalized observation
#         if self._vec_normalize_env is not None:
#             self._last_original_obs = new_obs_
#
#     def collect_rollouts(
#         self,
#         env: VecEnv,
#         callback: BaseCallback,
#         train_freq: TrainFreq,
#         replay_buffer: ReplayBuffer,
#         action_noise: Optional[ActionNoise] = None,
#         learning_starts: int = 0,
#         log_interval: Optional[int] = None,
#     ) -> RolloutReturn:
#         """
#         Collect experiences and store them into a ``ReplayBuffer``.
#
#         :param env: The training environment
#         :param callback: Callback that will be called at each step
#             (and at the beginning and end of the rollout)
#         :param train_freq: How much experience to collect
#             by doing rollouts of current policy.
#             Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
#             or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
#             with ``<n>`` being an integer greater than 0.
#         :param action_noise: Action noise that will be used for exploration
#             Required for deterministic policy (e.g. TD3). This can also be used
#             in addition to the stochastic policy for SAC.
#         :param learning_starts: Number of steps before learning for the warm-up phase.
#         :param replay_buffer:
#         :param log_interval: Log data every ``log_interval`` episodes
#         :return:
#         """
#         # Switch to eval mode (this affects batch norm / dropout)
#         self.policy.set_training_mode(False)
#
#         num_collected_steps, num_collected_episodes = 0, 0
#
#         assert isinstance(env, VecEnv), "You must pass a VecEnv"
#         assert train_freq.frequency > 0, "Should at least collect one step or episode."
#
#         if env.num_envs > 1:
#             assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."
#
#         if self.use_sde:
#             self.actor.reset_noise(env.num_envs)
#
#         callback.on_rollout_start()
#         continue_training = True
#         while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
#             if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
#                 # Sample a new noise matrix
#                 self.actor.reset_noise(env.num_envs)
#
#             # Select action randomly or according to policy
#             actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
#
#             # Rescale and perform action
#             new_obs, rewards, dones, infos = env.step(actions)
#
#             self.num_timesteps += env.num_envs
#             num_collected_steps += 1
#
#             # Give access to local variables
#             callback.update_locals(locals())
#             # Only stop training if return value is False, not when it is None.
#             if not callback.on_step():
#                 return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)
#
#             # Retrieve reward and episode length if using Monitor wrapper
#             self._update_info_buffer(infos, dones)
#
#             # Store data in replay buffer (normalized action and unnormalized observation)
#             self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]
#
#             self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
#
#             # For DQN, check if the target network should be updated
#             # and update the exploration schedule
#             # For SAC/TD3, the update is dones as the same time as the gradient update
#             # see https://github.com/hill-a/stable-baselines/issues/900
#             self._on_step()
#
#             for idx, done in enumerate(dones):
#                 if done:
#                     # Update stats
#                     num_collected_episodes += 1
#                     self._episode_num += 1
#
#                     if action_noise is not None:
#                         kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
#                         action_noise.reset(**kwargs)
#
#                     # Log training infos
#                     if log_interval is not None and self._episode_num % log_interval == 0:
#                         self._dump_logs()
#         callback.on_rollout_end()
#
#         return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
# from typing import Any, Dict, List, Optional, Tuple, Type, Union
#
# import torch as th
# from gymnasium import spaces
# from torch import nn
#
# from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
# from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
# from stable_baselines3.common.preprocessing import get_action_dim
# from stable_baselines3.common.torch_layers import (
#     BaseFeaturesExtractor,
#     CombinedExtractor,
#     FlattenExtractor,
#     NatureCNN,
#     create_mlp,
#     get_actor_critic_arch,
# )
# from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
#
# # CAP the standard deviation of the actor
# LOG_STD_MAX = 2
# LOG_STD_MIN = -20
#
#
# class Actor(BasePolicy):
#     """
#     Actor network (policy) for SAC.
#
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param net_arch: Network architecture
#     :param features_extractor: Network to extract features
#         (a CNN when using images, a nn.Flatten() layer otherwise)
#     :param features_dim: Number of features
#     :param activation_fn: Activation function
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param full_std: Whether to use (n_features x n_actions) parameters
#         for the std instead of only (n_features,) when using gSDE.
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     """
#
#     action_space: spaces.Box
#
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Box,
#         net_arch: List[int],
#         features_extractor: nn.Module,
#         features_dim: int,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         use_sde: bool = False,
#         log_std_init: float = -3,
#         full_std: bool = True,
#         use_expln: bool = False,
#         clip_mean: float = 2.0,
#         normalize_images: bool = True,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             features_extractor=features_extractor,
#             normalize_images=normalize_images,
#             squash_output=True,
#         )
#
#         # Save arguments to re-create object at loading
#         self.use_sde = use_sde
#         self.sde_features_extractor = None
#         self.net_arch = net_arch
#         self.features_dim = features_dim
#         self.activation_fn = activation_fn
#         self.log_std_init = log_std_init
#         self.use_expln = use_expln
#         self.full_std = full_std
#         self.clip_mean = clip_mean
#
#         action_dim = get_action_dim(self.action_space)
#         latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
#         self.latent_pi = nn.Sequential(*latent_pi_net)
#         last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
#
#         if self.use_sde:
#             self.action_dist = StateDependentNoiseDistribution(
#                 action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
#             )
#             self.mu, self.log_std = self.action_dist.proba_distribution_net(
#                 latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
#             )
#             # Avoid numerical issues by limiting the mean of the Gaussian
#             # to be in [-clip_mean, clip_mean]
#             if clip_mean > 0.0:
#                 self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
#         else:
#             self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
#             self.mu = nn.Linear(last_layer_dim, action_dim)
#             self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]
#
#     def _get_constructor_parameters(self) -> Dict[str, Any]:
#         data = super()._get_constructor_parameters()
#
#         data.update(
#             dict(
#                 net_arch=self.net_arch,
#                 features_dim=self.features_dim,
#                 activation_fn=self.activation_fn,
#                 use_sde=self.use_sde,
#                 log_std_init=self.log_std_init,
#                 full_std=self.full_std,
#                 use_expln=self.use_expln,
#                 features_extractor=self.features_extractor,
#                 clip_mean=self.clip_mean,
#             )
#         )
#         return data
#
#     def get_std(self) -> th.Tensor:
#         """
#         Retrieve the standard deviation of the action distribution.
#         Only useful when using gSDE.
#         It corresponds to ``th.exp(log_std)`` in the normal case,
#         but is slightly different when using ``expln`` function
#         (cf StateDependentNoiseDistribution doc).
#
#         :return:
#         """
#         msg = "get_std() is only available when using gSDE"
#         assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
#         return self.action_dist.get_std(self.log_std)
#
#     def reset_noise(self, batch_size: int = 1) -> None:
#         """
#         Sample new weights for the exploration matrix, when using gSDE.
#
#         :param batch_size:
#         """
#         msg = "reset_noise() is only available when using gSDE"
#         assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
#         self.action_dist.sample_weights(self.log_std, batch_size=batch_size)
#
#     def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
#         """
#         Get the parameters for the action distribution.
#
#         :param obs:
#         :return:
#             Mean, standard deviation and optional keyword arguments.
#         """
#         features = self.extract_features(obs, self.features_extractor)
#         latent_pi = self.latent_pi(features)
#         mean_actions = self.mu(latent_pi)
#
#         if self.use_sde:
#             return mean_actions, self.log_std, dict(latent_sde=latent_pi)
#         # Unstructured exploration (Original implementation)
#         log_std = self.log_std(latent_pi)  # type: ignore[operator]
#         # Original Implementation to cap the standard deviation
#         log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         return mean_actions, log_std, {}
#
#     def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
#         mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
#         # Note: the action is squashed
#         return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)
#
#     def action_log_prob(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor]:
#         mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
#         # return action and associated log prob
#         return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
#
#     def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
#         return self(observation, deterministic)
#
#
# class SACPolicy(BasePolicy):
#     """
#     Policy class (with both actor and critic) for SAC.
#
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
#     :param features_extractor_class: Features extractor to use.
#     :param features_extractor_kwargs: Keyword arguments
#         to pass to the features extractor.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     :param n_critics: Number of critic networks to create.
#     :param share_features_extractor: Whether to share or not the features extractor
#         between the actor and the critic (this saves computation time)
#     """
#
#     actor: Actor
#     critic: ContinuousCritic
#     critic_target: ContinuousCritic
#
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Box,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         use_sde: bool = False,
#         log_std_init: float = -3,
#         use_expln: bool = False,
#         clip_mean: float = 2.0,
#         features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#         n_critics: int = 2,
#         share_features_extractor: bool = False,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             features_extractor_class,
#             features_extractor_kwargs,
#             optimizer_class=optimizer_class,
#             optimizer_kwargs=optimizer_kwargs,
#             squash_output=True,
#             normalize_images=normalize_images,
#         )
#
#         if net_arch is None:
#             net_arch = [256, 256]
#
#         actor_arch, critic_arch = get_actor_critic_arch(net_arch)
#
#         self.net_arch = net_arch
#         self.activation_fn = activation_fn
#         self.net_args = {
#             "observation_space": self.observation_space,
#             "action_space": self.action_space,
#             "net_arch": actor_arch,
#             "activation_fn": self.activation_fn,
#             "normalize_images": normalize_images,
#         }
#         self.actor_kwargs = self.net_args.copy()
#
#         sde_kwargs = {
#             "use_sde": use_sde,
#             "log_std_init": log_std_init,
#             "use_expln": use_expln,
#             "clip_mean": clip_mean,
#         }
#         self.actor_kwargs.update(sde_kwargs)
#         self.critic_kwargs = self.net_args.copy()
#         self.critic_kwargs.update(
#             {
#                 "n_critics": n_critics,
#                 "net_arch": critic_arch,
#                 "share_features_extractor": share_features_extractor,
#             }
#         )
#
#         self.share_features_extractor = share_features_extractor
#
#         self._build(lr_schedule)
#
#     def _build(self, lr_schedule: Schedule) -> None:
#         self.actor = self.make_actor()
#         self.actor.optimizer = self.optimizer_class(
#             self.actor.parameters(),
#             lr=lr_schedule(1),  # type: ignore[call-arg]
#             **self.optimizer_kwargs,
#         )
#
#         if self.share_features_extractor:
#             self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
#             # Do not optimize the shared features extractor with the critic loss
#             # otherwise, there are gradient computation issues
#             critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
#         else:
#             # Create a separate features extractor for the critic
#             # this requires more memory and computation
#             self.critic = self.make_critic(features_extractor=None)
#             critic_parameters = list(self.critic.parameters())
#
#         # Critic target should not share the features extractor with critic
#         self.critic_target = self.make_critic(features_extractor=None)
#         self.critic_target.load_state_dict(self.critic.state_dict())
#
#         self.critic.optimizer = self.optimizer_class(
#             critic_parameters,
#             lr=lr_schedule(1),  # type: ignore[call-arg]
#             **self.optimizer_kwargs,
#         )
#
#         # Target networks should always be in eval mode
#         self.critic_target.set_training_mode(False)
#
#     def _get_constructor_parameters(self) -> Dict[str, Any]:
#         data = super()._get_constructor_parameters()
#
#         data.update(
#             dict(
#                 net_arch=self.net_arch,
#                 activation_fn=self.net_args["activation_fn"],
#                 use_sde=self.actor_kwargs["use_sde"],
#                 log_std_init=self.actor_kwargs["log_std_init"],
#                 use_expln=self.actor_kwargs["use_expln"],
#                 clip_mean=self.actor_kwargs["clip_mean"],
#                 n_critics=self.critic_kwargs["n_critics"],
#                 lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
#                 optimizer_class=self.optimizer_class,
#                 optimizer_kwargs=self.optimizer_kwargs,
#                 features_extractor_class=self.features_extractor_class,
#                 features_extractor_kwargs=self.features_extractor_kwargs,
#             )
#         )
#         return data
#
#     def reset_noise(self, batch_size: int = 1) -> None:
#         """
#         Sample new weights for the exploration matrix, when using gSDE.
#
#         :param batch_size:
#         """
#         self.actor.reset_noise(batch_size=batch_size)
#
#     def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
#         actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
#         return Actor(**actor_kwargs).to(self.device)
#
#     def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
#         critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
#         return ContinuousCritic(**critic_kwargs).to(self.device)
#
#     def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
#         return self._predict(obs, deterministic=deterministic)
#
#     def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
#         return self.actor(observation, deterministic)
#
#     def set_training_mode(self, mode: bool) -> None:
#         """
#         Put the policy in either training or evaluation mode.
#
#         This affects certain modules, such as batch normalisation and dropout.
#
#         :param mode: if true, set to training mode, else set to evaluation mode
#         """
#         self.actor.set_training_mode(mode)
#         self.critic.set_training_mode(mode)
#         self.training = mode
#
#
# MlpPolicy = SACPolicy
#
#
# class CnnPolicy(SACPolicy):
#     """
#     Policy class (with both actor and critic) for SAC.
#
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
#     :param features_extractor_class: Features extractor to use.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     :param n_critics: Number of critic networks to create.
#     :param share_features_extractor: Whether to share or not the features extractor
#         between the actor and the critic (this saves computation time)
#     """
#
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Box,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         use_sde: bool = False,
#         log_std_init: float = -3,
#         use_expln: bool = False,
#         clip_mean: float = 2.0,
#         features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#         n_critics: int = 2,
#         share_features_extractor: bool = False,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             use_sde,
#             log_std_init,
#             use_expln,
#             clip_mean,
#             features_extractor_class,
#             features_extractor_kwargs,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#             n_critics,
#             share_features_extractor,
#         )
#
#
# class MultiInputPolicy(SACPolicy):
#     """
#     Policy class (with both actor and critic) for SAC.
#
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
#     :param features_extractor_class: Features extractor to use.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     :param n_critics: Number of critic networks to create.
#     :param share_features_extractor: Whether to share or not the features extractor
#         between the actor and the critic (this saves computation time)
#     """
#
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Box,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         use_sde: bool = False,
#         log_std_init: float = -3,
#         use_expln: bool = False,
#         clip_mean: float = 2.0,
#         features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#         n_critics: int = 2,
#         share_features_extractor: bool = False,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             use_sde,
#             log_std_init,
#             use_expln,
#             clip_mean,
#             features_extractor_class,
#             features_extractor_kwargs,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#             n_critics,
#             share_features_extractor,
#         )
# """Common aliases for type hints"""
#
# from enum import Enum
# from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Protocol, SupportsFloat, Tuple, Union
#
# import gymnasium as gym
# import numpy as np
# import torch as th
#
# # Avoid circular imports, we use type hint as string to avoid it too
# if TYPE_CHECKING:
#     from stable_baselines3.common.callbacks import BaseCallback
#     from stable_baselines3.common.vec_env import VecEnv
#
# GymEnv = Union[gym.Env, "VecEnv"]
# GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
# GymResetReturn = Tuple[GymObs, Dict]
# AtariResetReturn = Tuple[np.ndarray, Dict[str, Any]]
# GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]
# AtariStepReturn = Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]
# TensorDict = Dict[str, th.Tensor]
# OptimizerStateDict = Dict[str, Any]
# MaybeCallback = Union[None, Callable, List["BaseCallback"], "BaseCallback"]
# PyTorchObs = Union[th.Tensor, TensorDict]
#
# # A schedule takes the remaining progress as input
# # and outputs a scalar (e.g. learning rate, clip range, ...)
# Schedule = Callable[[float], float]
#
#
# class RolloutBufferSamples(NamedTuple):
#     observations: th.Tensor
#     actions: th.Tensor
#     old_values: th.Tensor
#     old_log_prob: th.Tensor
#     advantages: th.Tensor
#     returns: th.Tensor
#
#
# class DictRolloutBufferSamples(NamedTuple):
#     observations: TensorDict
#     actions: th.Tensor
#     old_values: th.Tensor
#     old_log_prob: th.Tensor
#     advantages: th.Tensor
#     returns: th.Tensor
#
#
# class ReplayBufferSamples(NamedTuple):
#     observations: th.Tensor
#     actions: th.Tensor
#     next_observations: th.Tensor
#     dones: th.Tensor
#     rewards: th.Tensor
#
#
# class DictReplayBufferSamples(NamedTuple):
#     observations: TensorDict
#     actions: th.Tensor
#     next_observations: TensorDict
#     dones: th.Tensor
#     rewards: th.Tensor
#
#
# class RolloutReturn(NamedTuple):
#     episode_timesteps: int
#     n_episodes: int
#     continue_training: bool
#
#
# class TrainFrequencyUnit(Enum):
#     STEP = "step"
#     EPISODE = "episode"
#
#
# class TrainFreq(NamedTuple):
#     frequency: int
#     unit: TrainFrequencyUnit  # either "step" or "episode"
#
#
# class PolicyPredictor(Protocol):
#     def predict(
#         self,
#         observation: Union[np.ndarray, Dict[str, np.ndarray]],
#         state: Optional[Tuple[np.ndarray, ...]] = None,
#         episode_start: Optional[np.ndarray] = None,
#         deterministic: bool = False,
#     ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
#         """
#         Get the policy action from an observation (and optional hidden state).
#         Includes sugar-coating to handle different observations (e.g. normalizing images).
#
#         :param observation: the input observation
#         :param state: The last hidden states (can be None, used in recurrent policies)
#         :param episode_start: The last masks (can be None, used in recurrent policies)
#             this correspond to beginning of episodes,
#             where the hidden states of the RNN must be reset.
#         :param deterministic: Whether or not to return deterministic actions.
#         :return: the model's action and the next hidden state
#             (used in recurrent policies)
#         """
# import glob
# import os
# import platform
# import random
# import re
# from collections import deque
# from itertools import zip_longest
# from typing import Dict, Iterable, List, Optional, Tuple, Union
#
# import cloudpickle
# import gymnasium as gym
# import numpy as np
# import torch as th
# from gymnasium import spaces
#
# import stable_baselines3 as sb3
#
# # Check if tensorboard is available for pytorch
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     SummaryWriter = None  # type: ignore[misc, assignment]
#
# from stable_baselines3.common.logger import Logger, configure
# from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict, TrainFreq, TrainFrequencyUnit
#
#
# def set_random_seed(seed: int, using_cuda: bool = False) -> None:
#     """
#     Seed the different random generators.
#
#     :param seed:
#     :param using_cuda:
#     """
#     # Seed python RNG
#     random.seed(seed)
#     # Seed numpy RNG
#     np.random.seed(seed)
#     # seed the RNG for all devices (both CPU and CUDA)
#     th.manual_seed(seed)
#
#     if using_cuda:
#         # Deterministic operations for CuDNN, it may impact performances
#         th.backends.cudnn.deterministic = True
#         th.backends.cudnn.benchmark = False
#
#
# # From stable baselines
# def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
#     """
#     Computes fraction of variance that ypred explains about y.
#     Returns 1 - Var[y-ypred] / Var[y]
#
#     interpretation:
#         ev=0  =>  might as well have predicted zero
#         ev=1  =>  perfect prediction
#         ev<0  =>  worse than just predicting zero
#
#     :param y_pred: the prediction
#     :param y_true: the expected value
#     :return: explained variance of ypred and y
#     """
#     assert y_true.ndim == 1 and y_pred.ndim == 1
#     var_y = np.var(y_true)
#     return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
#
#
# def update_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
#     """
#     Update the learning rate for a given optimizer.
#     Useful when doing linear schedule.
#
#     :param optimizer: Pytorch optimizer
#     :param learning_rate: New learning rate value
#     """
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = learning_rate
#
#
# def get_schedule_fn(value_schedule: Union[Schedule, float]) -> Schedule:
#     """
#     Transform (if needed) learning rate and clip range (for PPO)
#     to callable.
#
#     :param value_schedule: Constant value of schedule function
#     :return: Schedule function (can return constant value)
#     """
#     # If the passed schedule is a float
#     # create a constant function
#     if isinstance(value_schedule, (float, int)):
#         # Cast to float to avoid errors
#         value_schedule = constant_fn(float(value_schedule))
#     else:
#         assert callable(value_schedule)
#     # Cast to float to avoid unpickling errors to enable weights_only=True, see GH#1900
#     # Some types are have odd behaviors when part of a Schedule, like numpy floats
#     return lambda progress_remaining: float(value_schedule(progress_remaining))
#
#
# def get_linear_fn(start: float, end: float, end_fraction: float) -> Schedule:
#     """
#     Create a function that interpolates linearly between start and end
#     between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
#     This is used in DQN for linearly annealing the exploration fraction
#     (epsilon for the epsilon-greedy strategy).
#
#     :params start: value to start with if ``progress_remaining`` = 1
#     :params end: value to end with if ``progress_remaining`` = 0
#     :params end_fraction: fraction of ``progress_remaining``
#         where end is reached e.g 0.1 then end is reached after 10%
#         of the complete training process.
#     :return: Linear schedule function.
#     """
#
#     def func(progress_remaining: float) -> float:
#         if (1 - progress_remaining) > end_fraction:
#             return end
#         else:
#             return start + (1 - progress_remaining) * (end - start) / end_fraction
#
#     return func
#
#
# def constant_fn(val: float) -> Schedule:
#     """
#     Create a function that returns a constant
#     It is useful for learning rate schedule (to avoid code duplication)
#
#     :param val: constant value
#     :return: Constant schedule function.
#     """
#
#     def func(_):
#         return val
#
#     return func
#
#
# def get_device(device: Union[th.device, str] = "auto") -> th.device:
#     """
#     Retrieve PyTorch device.
#     It checks that the requested device is available first.
#     For now, it supports only cpu and cuda.
#     By default, it tries to use the gpu.
#
#     :param device: One for 'auto', 'cuda', 'cpu'
#     :return: Supported Pytorch device
#     """
#     # Cuda by default
#     if device == "auto":
#         device = "cuda"
#     # Force conversion to th.device
#     device = th.device(device)
#
#     # Cuda not available
#     if device.type == th.device("cuda").type and not th.cuda.is_available():
#         return th.device("cpu")
#
#     return device
#
#
# def get_latest_run_id(log_path: str = "", log_name: str = "") -> int:
#     """
#     Returns the latest run number for the given log name and log path,
#     by finding the greatest number in the directories.
#
#     :param log_path: Path to the log folder containing several runs.
#     :param log_name: Name of the experiment. Each run is stored
#         in a folder named ``log_name_1``, ``log_name_2``, ...
#     :return: latest run number
#     """
#     max_run_id = 0
#     for path in glob.glob(os.path.join(log_path, f"{glob.escape(log_name)}_[0-9]*")):
#         file_name = path.split(os.sep)[-1]
#         ext = file_name.split("_")[-1]
#         if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
#             max_run_id = int(ext)
#     return max_run_id
#
#
# def configure_logger(
#     verbose: int = 0,
#     tensorboard_log: Optional[str] = None,
#     tb_log_name: str = "",
#     reset_num_timesteps: bool = True,
# ) -> Logger:
#     """
#     Configure the logger's outputs.
#
#     :param verbose: Verbosity level: 0 for no output, 1 for the standard output to be part of the logger outputs
#     :param tensorboard_log: the log location for tensorboard (if None, no logging)
#     :param tb_log_name: tensorboard log
#     :param reset_num_timesteps:  Whether the ``num_timesteps`` attribute is reset or not.
#         It allows to continue a previous learning curve (``reset_num_timesteps=False``)
#         or start from t=0 (``reset_num_timesteps=True``, the default).
#     :return: The logger object
#     """
#     save_path, format_strings = None, ["stdout"]
#
#     if tensorboard_log is not None and SummaryWriter is None:
#         raise ImportError("Trying to log data to tensorboard but tensorboard is not installed.")
#
#     if tensorboard_log is not None and SummaryWriter is not None:
#         latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
#         if not reset_num_timesteps:
#             # Continue training in the same directory
#             latest_run_id -= 1
#         save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
#         if verbose >= 1:
#             format_strings = ["stdout", "tensorboard"]
#         else:
#             format_strings = ["tensorboard"]
#     elif verbose == 0:
#         format_strings = [""]
#     return configure(save_path, format_strings=format_strings)
#
#
# def check_for_correct_spaces(env: GymEnv, observation_space: spaces.Space, action_space: spaces.Space) -> None:
#     """
#     Checks that the environment has same spaces as provided ones. Used by BaseAlgorithm to check if
#     spaces match after loading the model with given env.
#     Checked parameters:
#     - observation_space
#     - action_space
#
#     :param env: Environment to check for valid spaces
#     :param observation_space: Observation space to check against
#     :param action_space: Action space to check against
#     """
#     if observation_space != env.observation_space:
#         raise ValueError(f"Observation spaces do not match: {observation_space} != {env.observation_space}")
#     if action_space != env.action_space:
#         raise ValueError(f"Action spaces do not match: {action_space} != {env.action_space}")
#
#
# def check_shape_equal(space1: spaces.Space, space2: spaces.Space) -> None:
#     """
#     If the spaces are Box, check that they have the same shape.
#
#     If the spaces are Dict, it recursively checks the subspaces.
#
#     :param space1: Space
#     :param space2: Other space
#     """
#     if isinstance(space1, spaces.Dict):
#         assert isinstance(space2, spaces.Dict), "spaces must be of the same type"
#         assert space1.spaces.keys() == space2.spaces.keys(), "spaces must have the same keys"
#         for key in space1.spaces.keys():
#             check_shape_equal(space1.spaces[key], space2.spaces[key])
#     elif isinstance(space1, spaces.Box):
#         assert space1.shape == space2.shape, "spaces must have the same shape"
#
#
# def is_vectorized_box_observation(observation: np.ndarray, observation_space: spaces.Box) -> bool:
#     """
#     For box observation type, detects and validates the shape,
#     then returns whether or not the observation is vectorized.
#
#     :param observation: the input observation to validate
#     :param observation_space: the observation space
#     :return: whether the given observation is vectorized or not
#     """
#     if observation.shape == observation_space.shape:
#         return False
#     elif observation.shape[1:] == observation_space.shape:
#         return True
#     else:
#         raise ValueError(
#             f"Error: Unexpected observation shape {observation.shape} for "
#             + f"Box environment, please use {observation_space.shape} "
#             + "or (n_env, {}) for the observation shape.".format(", ".join(map(str, observation_space.shape)))
#         )
#
#
# def is_vectorized_discrete_observation(observation: Union[int, np.ndarray], observation_space: spaces.Discrete) -> bool:
#     """
#     For discrete observation type, detects and validates the shape,
#     then returns whether or not the observation is vectorized.
#
#     :param observation: the input observation to validate
#     :param observation_space: the observation space
#     :return: whether the given observation is vectorized or not
#     """
#     if isinstance(observation, int) or observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
#         return False
#     elif len(observation.shape) == 1:
#         return True
#     else:
#         raise ValueError(
#             f"Error: Unexpected observation shape {observation.shape} for "
#             + "Discrete environment, please use () or (n_env,) for the observation shape."
#         )
#
#
# def is_vectorized_multidiscrete_observation(observation: np.ndarray, observation_space: spaces.MultiDiscrete) -> bool:
#     """
#     For multidiscrete observation type, detects and validates the shape,
#     then returns whether or not the observation is vectorized.
#
#     :param observation: the input observation to validate
#     :param observation_space: the observation space
#     :return: whether the given observation is vectorized or not
#     """
#     if observation.shape == (len(observation_space.nvec),):
#         return False
#     elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
#         return True
#     else:
#         raise ValueError(
#             f"Error: Unexpected observation shape {observation.shape} for MultiDiscrete "
#             + f"environment, please use ({len(observation_space.nvec)},) or "
#             + f"(n_env, {len(observation_space.nvec)}) for the observation shape."
#         )
#
#
# def is_vectorized_multibinary_observation(observation: np.ndarray, observation_space: spaces.MultiBinary) -> bool:
#     """
#     For multibinary observation type, detects and validates the shape,
#     then returns whether or not the observation is vectorized.
#
#     :param observation: the input observation to validate
#     :param observation_space: the observation space
#     :return: whether the given observation is vectorized or not
#     """
#     if observation.shape == observation_space.shape:
#         return False
#     elif len(observation.shape) == len(observation_space.shape) + 1 and observation.shape[1:] == observation_space.shape:
#         return True
#     else:
#         raise ValueError(
#             f"Error: Unexpected observation shape {observation.shape} for MultiBinary "
#             + f"environment, please use {observation_space.shape} or "
#             + f"(n_env, {observation_space.n}) for the observation shape."
#         )
#
#
# def is_vectorized_dict_observation(observation: np.ndarray, observation_space: spaces.Dict) -> bool:
#     """
#     For dict observation type, detects and validates the shape,
#     then returns whether or not the observation is vectorized.
#
#     :param observation: the input observation to validate
#     :param observation_space: the observation space
#     :return: whether the given observation is vectorized or not
#     """
#     # We first assume that all observations are not vectorized
#     all_non_vectorized = True
#     for key, subspace in observation_space.spaces.items():
#         # This fails when the observation is not vectorized
#         # or when it has the wrong shape
#         if observation[key].shape != subspace.shape:
#             all_non_vectorized = False
#             break
#
#     if all_non_vectorized:
#         return False
#
#     all_vectorized = True
#     # Now we check that all observation are vectorized and have the correct shape
#     for key, subspace in observation_space.spaces.items():
#         if observation[key].shape[1:] != subspace.shape:
#             all_vectorized = False
#             break
#
#     if all_vectorized:
#         return True
#     else:
#         # Retrieve error message
#         error_msg = ""
#         try:
#             is_vectorized_observation(observation[key], observation_space.spaces[key])
#         except ValueError as e:
#             error_msg = f"{e}"
#         raise ValueError(
#             f"There seems to be a mix of vectorized and non-vectorized observations. "
#             f"Unexpected observation shape {observation[key].shape} for key {key} "
#             f"of type {observation_space.spaces[key]}. {error_msg}"
#         )
#
#
# def is_vectorized_observation(observation: Union[int, np.ndarray], observation_space: spaces.Space) -> bool:
#     """
#     For every observation type, detects and validates the shape,
#     then returns whether or not the observation is vectorized.
#
#     :param observation: the input observation to validate
#     :param observation_space: the observation space
#     :return: whether the given observation is vectorized or not
#     """
#
#     is_vec_obs_func_dict = {
#         spaces.Box: is_vectorized_box_observation,
#         spaces.Discrete: is_vectorized_discrete_observation,
#         spaces.MultiDiscrete: is_vectorized_multidiscrete_observation,
#         spaces.MultiBinary: is_vectorized_multibinary_observation,
#         spaces.Dict: is_vectorized_dict_observation,
#     }
#
#     for space_type, is_vec_obs_func in is_vec_obs_func_dict.items():
#         if isinstance(observation_space, space_type):
#             return is_vec_obs_func(observation, observation_space)  # type: ignore[operator]
#     else:
#         # for-else happens if no break is called
#         raise ValueError(f"Error: Cannot determine if the observation is vectorized with the space type {observation_space}.")
#
#
# def safe_mean(arr: Union[np.ndarray, list, deque]) -> float:
#     """
#     Compute the mean of an array if there is at least one element.
#     For empty array, return NaN. It is used for logging only.
#
#     :param arr: Numpy array or list of values
#     :return:
#     """
#     return np.nan if len(arr) == 0 else float(np.mean(arr))  # type: ignore[arg-type]
#
#
# def get_parameters_by_name(model: th.nn.Module, included_names: Iterable[str]) -> List[th.Tensor]:
#     """
#     Extract parameters from the state dict of ``model``
#     if the name contains one of the strings in ``included_names``.
#
#     :param model: the model where the parameters come from.
#     :param included_names: substrings of names to include.
#     :return: List of parameters values (Pytorch tensors)
#         that matches the queried names.
#     """
#     return [param for name, param in model.state_dict().items() if any([key in name for key in included_names])]
#
#
# def zip_strict(*iterables: Iterable) -> Iterable:
#     r"""
#     ``zip()`` function but enforces that iterables are of equal length.
#     Raises ``ValueError`` if iterables not of equal length.
#     Code inspired by Stackoverflow answer for question #32954486.
#
#     :param \*iterables: iterables to ``zip()``
#     """
#     # As in Stackoverflow #32954486, use
#     # new object for "empty" in case we have
#     # Nones in iterable.
#     sentinel = object()
#     for combo in zip_longest(*iterables, fillvalue=sentinel):
#         if sentinel in combo:
#             raise ValueError("Iterables have different lengths")
#         yield combo
#
#
# def polyak_update(
#     params: Iterable[th.Tensor],
#     target_params: Iterable[th.Tensor],
#     tau: float,
# ) -> None:
#     """
#     Perform a Polyak average update on ``target_params`` using ``params``:
#     target parameters are slowly updated towards the main parameters.
#     ``tau``, the soft update coefficient controls the interpolation:
#     ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
#     The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
#     or a computation graph, reducing memory cost and improving performance.  We scale the target params
#     by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
#     params (in place).
#     See https://github.com/DLR-RM/stable-baselines3/issues/93
#
#     :param params: parameters to use to update the target params
#     :param target_params: parameters to update
#     :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
#     """
#     with th.no_grad():
#         # zip does not raise an exception if length of parameters does not match.
#         for param, target_param in zip_strict(params, target_params):
#             target_param.data.mul_(1 - tau)
#             th.add(target_param.data, param.data, alpha=tau, out=target_param.data)
#
#
# def obs_as_tensor(obs: Union[np.ndarray, Dict[str, np.ndarray]], device: th.device) -> Union[th.Tensor, TensorDict]:
#     """
#     Moves the observation to the given device.
#
#     :param obs:
#     :param device: PyTorch device
#     :return: PyTorch tensor of the observation on a desired device.
#     """
#     if isinstance(obs, np.ndarray):
#         return th.as_tensor(obs, device=device)
#     elif isinstance(obs, dict):
#         return {key: th.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
#     else:
#         raise Exception(f"Unrecognized type of observation {type(obs)}")
#
#
# def should_collect_more_steps(
#     train_freq: TrainFreq,
#     num_collected_steps: int,
#     num_collected_episodes: int,
# ) -> bool:
#     """
#     Helper used in ``collect_rollouts()`` of off-policy algorithms
#     to determine the termination condition.
#
#     :param train_freq: How much experience should be collected before updating the policy.
#     :param num_collected_steps: The number of already collected steps.
#     :param num_collected_episodes: The number of already collected episodes.
#     :return: Whether to continue or not collecting experience
#         by doing rollouts of the current policy.
#     """
#     if train_freq.unit == TrainFrequencyUnit.STEP:
#         return num_collected_steps < train_freq.frequency
#
#     elif train_freq.unit == TrainFrequencyUnit.EPISODE:
#         return num_collected_episodes < train_freq.frequency
#
#     else:
#         raise ValueError(
#             "The unit of the `train_freq` must be either TrainFrequencyUnit.STEP "
#             f"or TrainFrequencyUnit.EPISODE not '{train_freq.unit}'!"
#         )
#
#
# def get_system_info(print_info: bool = True) -> Tuple[Dict[str, str], str]:
#     """
#     Retrieve system and python env info for the current system.
#
#     :param print_info: Whether to print or not those infos
#     :return: Dictionary summing up the version for each relevant package
#         and a formatted string.
#     """
#     env_info = {
#         # In OS, a regex is used to add a space between a "#" and a number to avoid
#         # wrongly linking to another issue on GitHub. Example: turn "#42" to "# 42".
#         "OS": re.sub(r"#(\d)", r"# \1", f"{platform.platform()} {platform.version()}"),
#         "Python": platform.python_version(),
#         "Stable-Baselines3": sb3.__version__,
#         "PyTorch": th.__version__,
#         "GPU Enabled": str(th.cuda.is_available()),
#         "Numpy": np.__version__,
#         "Cloudpickle": cloudpickle.__version__,
#         "Gymnasium": gym.__version__,
#     }
#     try:
#         import gym as openai_gym
#
#         env_info.update({"OpenAI Gym": openai_gym.__version__})
#     except ImportError:
#         pass
#
#     env_info_str = ""
#     for key, value in env_info.items():
#         env_info_str += f"- {key}: {value}\n"
#     if print_info:
#         print(env_info_str)
#     return env_info, env_info_str
#
# from typing import Dict, List, Optional, Tuple, Type, Union
#
# import gymnasium as gym
# import torch as th
# from gymnasium import spaces
# from torch import nn
#
# from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
# from stable_baselines3.common.type_aliases import TensorDict
# from stable_baselines3.common.utils import get_device
#
#
# class BaseFeaturesExtractor(nn.Module):
#     """
#     Base class that represents a features extractor.
#
#     :param observation_space: The observation space of the environment
#     :param features_dim: Number of features extracted.
#     """
#
#     def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
#         super().__init__()
#         assert features_dim > 0
#         self._observation_space = observation_space
#         self._features_dim = features_dim
#
#     @property
#     def features_dim(self) -> int:
#         """The number of features that the extractor outputs."""
#         return self._features_dim
#
#
# class FlattenExtractor(BaseFeaturesExtractor):
#     """
#     Feature extract that flatten the input.
#     Used as a placeholder when feature extraction is not needed.
#
#     :param observation_space: The observation space of the environment
#     """
#
#     def __init__(self, observation_space: gym.Space) -> None:
#         super().__init__(observation_space, get_flattened_obs_dim(observation_space))
#         self.flatten = nn.Flatten()
#
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.flatten(observations)
#
#
# class NatureCNN(BaseFeaturesExtractor):
#     """
#     CNN from DQN Nature paper:
#         Mnih, Volodymyr, et al.
#         "Human-level control through deep reinforcement learning."
#         Nature 518.7540 (2015): 529-533.
#
#     :param observation_space: The observation space of the environment
#     :param features_dim: Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     :param normalized_image: Whether to assume that the image is already normalized
#         or not (this disables dtype and bounds checks): when True, it only checks that
#         the space is a Box and has 3 dimensions.
#         Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
#     """
#
#     def __init__(
#         self,
#         observation_space: gym.Space,
#         features_dim: int = 512,
#         normalized_image: bool = False,
#     ) -> None:
#         assert isinstance(observation_space, spaces.Box), (
#             "NatureCNN must be used with a gym.spaces.Box ",
#             f"observation space, not {observation_space}",
#         )
#         super().__init__(observation_space, features_dim)
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
#             "You should use NatureCNN "
#             f"only with images not with {observation_space}\n"
#             "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
#             "If you are using a custom environment,\n"
#             "please check it using our env checker:\n"
#             "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
#             "If you are using `VecNormalize` or already normalized channel-first images "
#             "you should pass `normalize_images=False`: \n"
#             "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
#         )
#         n_input_channels = observation_space.shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#
#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
#
#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
#
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.linear(self.cnn(observations))
#
#
# def create_mlp(
#     input_dim: int,
#     output_dim: int,
#     net_arch: List[int],
#     activation_fn: Type[nn.Module] = nn.ReLU,
#     squash_output: bool = False,
#     with_bias: bool = True,
#     pre_linear_modules: Optional[List[Type[nn.Module]]] = None,
#     post_linear_modules: Optional[List[Type[nn.Module]]] = None,
# ) -> List[nn.Module]:
#     """
#     Create a multi layer perceptron (MLP), which is
#     a collection of fully-connected layers each followed by an activation function.
#
#     :param input_dim: Dimension of the input vector
#     :param output_dim: Dimension of the output (last layer, for instance, the number of actions)
#     :param net_arch: Architecture of the neural net
#         It represents the number of units per layer.
#         The length of this list is the number of layers.
#     :param activation_fn: The activation function
#         to use after each layer.
#     :param squash_output: Whether to squash the output using a Tanh
#         activation function
#     :param with_bias: If set to False, the layers will not learn an additive bias
#     :param pre_linear_modules: List of nn.Module to add before the linear layers.
#         These modules should maintain the input tensor dimension (e.g. BatchNorm).
#         The number of input features is passed to the module's constructor.
#         Compared to post_linear_modules, they are used before the output layer (output_dim > 0).
#     :param post_linear_modules: List of nn.Module to add after the linear layers
#         (and before the activation function). These modules should maintain the input
#         tensor dimension (e.g. Dropout, LayerNorm). They are not used after the
#         output layer (output_dim > 0). The number of input features is passed to
#         the module's constructor.
#     :return: The list of layers of the neural network
#     """
#
#     pre_linear_modules = pre_linear_modules or []
#     post_linear_modules = post_linear_modules or []
#
#     modules = []
#     if len(net_arch) > 0:
#         # BatchNorm maintains input dim
#         for module in pre_linear_modules:
#             modules.append(module(input_dim))
#
#         modules.append(nn.Linear(input_dim, net_arch[0], bias=with_bias))
#
#         # LayerNorm, Dropout maintain output dim
#         for module in post_linear_modules:
#             modules.append(module(net_arch[0]))
#
#         modules.append(activation_fn())
#
#     for idx in range(len(net_arch) - 1):
#         for module in pre_linear_modules:
#             modules.append(module(net_arch[idx]))
#
#         modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
#
#         for module in post_linear_modules:
#             modules.append(module(net_arch[idx + 1]))
#
#         modules.append(activation_fn())
#
#     if output_dim > 0:
#         last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
#         # Only add BatchNorm before output layer
#         for module in pre_linear_modules:
#             modules.append(module(last_layer_dim))
#
#         modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
#     if squash_output:
#         modules.append(nn.Tanh())
#     return modules
#
#
# class MlpExtractor(nn.Module):
#     """
#     Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
#     the observations (if no features extractor is applied) as an input and outputs a latent representation
#     for the policy and a value network.
#
#     The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
#     It can be in either of the following forms:
#     1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
#         policy and value nets individually. If it is missing any of the keys (pi or vf),
#         zero layers will be considered for that key.
#     2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
#         in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
#         where int_list is the same for the actor and critic.
#
#     .. note::
#         If a key is not specified or an empty list is passed ``[]``, a linear network will be used.
#
#     :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
#     :param net_arch: The specification of the policy and value networks.
#         See above for details on its formatting.
#     :param activation_fn: The activation function to use for the networks.
#     :param device: PyTorch device.
#     """
#
#     def __init__(
#         self,
#         feature_dim: int,
#         net_arch: Union[List[int], Dict[str, List[int]]],
#         activation_fn: Type[nn.Module],
#         device: Union[th.device, str] = "auto",
#     ) -> None:
#         super().__init__()
#         device = get_device(device)
#         policy_net: List[nn.Module] = []
#         value_net: List[nn.Module] = []
#         last_layer_dim_pi = feature_dim
#         last_layer_dim_vf = feature_dim
#
#         # save dimensions of layers in policy and value nets
#         if isinstance(net_arch, dict):
#             # Note: if key is not specified, assume linear network
#             pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
#             vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
#         else:
#             pi_layers_dims = vf_layers_dims = net_arch
#         # Iterate through the policy layers and build the policy net
#         for curr_layer_dim in pi_layers_dims:
#             policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
#             policy_net.append(activation_fn())
#             last_layer_dim_pi = curr_layer_dim
#         # Iterate through the value layers and build the value net
#         for curr_layer_dim in vf_layers_dims:
#             value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
#             value_net.append(activation_fn())
#             last_layer_dim_vf = curr_layer_dim
#
#         # Save dim, used to create the distributions
#         self.latent_dim_pi = last_layer_dim_pi
#         self.latent_dim_vf = last_layer_dim_vf
#
#         # Create networks
#         # If the list of layers is empty, the network will just act as an Identity module
#         self.policy_net = nn.Sequential(*policy_net).to(device)
#         self.value_net = nn.Sequential(*value_net).to(device)
#
#     def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         :return: latent_policy, latent_value of the specified network.
#             If all layers are shared, then ``latent_policy == latent_value``
#         """
#         return self.forward_actor(features), self.forward_critic(features)
#
#     def forward_actor(self, features: th.Tensor) -> th.Tensor:
#         return self.policy_net(features)
#
#     def forward_critic(self, features: th.Tensor) -> th.Tensor:
#         return self.value_net(features)
#
#
# class CombinedExtractor(BaseFeaturesExtractor):
#     """
#     Combined features extractor for Dict observation spaces.
#     Builds a features extractor for each key of the space. Input from each space
#     is fed through a separate submodule (CNN or MLP, depending on input shape),
#     the output features are concatenated and fed through additional MLP network ("combined").
#
#     :param observation_space:
#     :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
#         256 to avoid exploding network sizes.
#     :param normalized_image: Whether to assume that the image is already normalized
#         or not (this disables dtype and bounds checks): when True, it only checks that
#         the space is a Box and has 3 dimensions.
#         Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
#     """
#
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         cnn_output_dim: int = 256,
#         normalized_image: bool = False,
#     ) -> None:
#         # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
#         super().__init__(observation_space, features_dim=1)
#
#         extractors: Dict[str, nn.Module] = {}
#
#         total_concat_size = 0
#         for key, subspace in observation_space.spaces.items():
#             if is_image_space(subspace, normalized_image=normalized_image):
#                 extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
#                 total_concat_size += cnn_output_dim
#             else:
#                 # The observation key is a vector, flatten it if needed
#                 extractors[key] = nn.Flatten()
#                 total_concat_size += get_flattened_obs_dim(subspace)
#
#         self.extractors = nn.ModuleDict(extractors)
#
#         # Update the features dim manually
#         self._features_dim = total_concat_size
#
#     def forward(self, observations: TensorDict) -> th.Tensor:
#         encoded_tensor_list = []
#
#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))
#         return th.cat(encoded_tensor_list, dim=1)
#
#
# def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
#     """
#     Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).
#
#     The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
#     which can be different for the actor and the critic.
#     It is assumed to be a list of ints or a dict.
#
#     1. If it is a list, actor and critic networks will have the same architecture.
#         The architecture is represented by a list of integers (of arbitrary length (zero allowed))
#         each specifying the number of units per layer.
#        If the number of ints is zero, the network will be linear.
#     2. If it is a dict,  it should have the following structure:
#        ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
#        where the network architecture is a list as described in 1.
#
#     For example, to have actor and critic that share the same network architecture,
#     you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).
#
#     If you want a different architecture for the actor and the critic,
#     then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.
#
#     .. note::
#         Compared to their on-policy counterparts, no shared layers (other than the features extractor)
#         between the actor and the critic are allowed (to prevent issues with target networks).
#
#     :param net_arch: The specification of the actor and critic networks.
#         See above for details on its formatting.
#     :return: The network architectures for the actor and the critic
#     """
#     if isinstance(net_arch, list):
#         actor_arch, critic_arch = net_arch, net_arch
#     else:
#         assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
#         assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
#         assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
#         actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
#     return actor_arch, critic_arch
#
# import gymnasium as gym
#
# envs = gym.envs.registry.keys()
# for env in sorted(envs):
#     print(env)
#
# import os
# import warnings
# from pathlib import Path
# import numpy as np
# import torch
# import gymnasium as gym
# from sac_torch import Agent
# from utils import plot_learning_curve
#
# # -----------------------------------------
# # Paths and setup
# # -----------------------------------------
# CHECKPOINT_DIR = Path("tmp/sac")
# VIDEO_DIR = Path("tmp/video")
# PLOTS_DIR = Path("plots")
#
# for p in (CHECKPOINT_DIR, VIDEO_DIR, PLOTS_DIR):
#     p.mkdir(parents=True, exist_ok=True)
#
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
#
# print("Running SAC on Walker2d-v5...")
# print("CUDA available:", torch.cuda.is_available())
# print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
#
# # -----------------------------------------
# # Create environment
# # -----------------------------------------
# env = gym.make("Walker2d-v5", render_mode="human")
# obs_shape = env.observation_space.shape
# n_actions = env.action_space.shape[0]
#
# # -----------------------------------------
# # Agent setup
# # -----------------------------------------
# agent = Agent(
#     input_dims=obs_shape,
#     env=env,
#     n_actions=n_actions,
#     automatic_entropy_tuning=True
# )
#
# # -----------------------------------------
# # Training loop
# # -----------------------------------------
# N_GAMES = 100_000
# BEST_SCORE = -np.inf
# scores = []
#
# for episode in range(N_GAMES):
#     obs, _ = env.reset()
#     done, score = False, 0.0
#
#     while not done:
#         action = agent.choose_action(obs)
#         obs_, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#
#         agent.remember(obs, action, reward, obs_, done)
#         agent.learn()
#
#         obs = obs_
#         score += reward
#
#     scores.append(score)
#     avg_score = np.mean(scores[-100:])
#
#     if avg_score > BEST_SCORE:
#         BEST_SCORE = avg_score
#         agent.save_models()
#
#     if episode % 10 == 0:
#         print(f"Episode {episode:5d} | Score: {score:8.2f} | Avg100: {avg_score:8.2f}")
#
# # -----------------------------------------
# # Save results
# # -----------------------------------------
# x = [i + 1 for i in range(len(scores))]
# plot_learning_curve(x, scores, str(PLOTS_DIR / "sac_walker2d.png"))
# print("\n✅ SAC training complete for Walker2d-v5.")


