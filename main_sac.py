# import pybullet_envs
# from flat_wrapper import ObsOnlyWrapper
#
# import numpy as np
# from sac_torch import Agent
# from utils import plot_learning_curve
# from gym import wrappers
# import os
# import warnings
# import torch
# warnings.filterwarnings("ignore", category=DeprecationWarning)  # Optional
# import gymnasium as gym
#
# import panda_gym
#
# if __name__ == '__main__':
#
#     print("Running main...")
#     ###############################DEBUGGING#########################
#     print("CUDA available:", torch.cuda.is_available())
#     print("PyTorch CUDA version:", torch.version.cuda)
#     print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
#     ###############################DEBUGGING#########################
#
#     os.makedirs("tmp", exist_ok=True)
#     os.makedirs("plots", exist_ok=True)
#     os.makedirs("tmp/video", exist_ok=True)
#
#     warnings.filterwarnings("ignore", category=UserWarning)
#     warnings.filterwarnings("ignore", category=DeprecationWarning)
#
#
#     #env = gym.make('PandaReachDense-v3')
#     #env = gym.make("PandaPushDense-v3", render_mode="human")
#     # env = gym.make(
#     #     "PandaPushDense-v3",
#     #     control_type="joints",  # ή "torque"
#     #     reward_type="dense",
#     #     render_mode="human"
#     # )
#     base_env = gym.make(
#         "PandaPickAndPlaceJoints-v3",
#         control_type="joints",  # ή "torque"
#         reward_type="dense",
#         render_mode="human"
#     )
#     env = ObsOnlyWrapper(base_env)
#     #agent = Agent(input_dims=env.observation_space.shape, env=env,
#     #        n_actions=env.action_space.shape[0])
#
#     #obs_shape = env.observation_space["observation"].shape  # tuple π.χ. (30,)
#     obs_shape = env.observation_space.shape
#     n_actions = env.action_space.shape[0]
#
#     #agent = Agent(input_dims=obs_shape, env=env, n_actions=n_actions)
#     agent = Agent(
#         input_dims=env.observation_space.shape,  # π.χ. (30,)
#         env=env,
#         n_actions=env.action_space.shape[0]
#     )
#     n_games = 100000
#     # uncomment this line and do a mkdir tmp && mkdir video if you want to
#     # record video of the agent playing the game.
#     #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
#     filename = 'inverted_pendulum.png'
#
#     figure_file = 'plots/' + filename
#
#     best_score = -np.inf          # αντί για env.reward_range[0]
#     score_history = []
#     load_checkpoint = False
#
#     if load_checkpoint:
#         agent.load_models()
#         env.render(mode='human')
#
#     for i in range(n_games):
#         obs, info = env.reset()  # obs είναι το flat vector
#         done = False
#         score = 0.0
#
#         while not done:
#             action = agent.choose_action(obs)
#
#             obs_, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#
#             agent.remember(obs, action, reward, obs_, done)
#             if not load_checkpoint:
#                 agent.learn()
#
#             score += reward
#             obs = obs_  # move to next state
#
#         score_history.append(score)
#         avg_score = np.mean(score_history[-100:])
#
#         if avg_score > best_score:
#             best_score = avg_score
#             if not load_checkpoint:
#                 agent.save_models()
#
#         print(f"episode {i}  score {score:.1f}  avg_score {avg_score:.1f}")
#
#     if not load_checkpoint:
#         x = [i+1 for i in range(n_games)]
#         plot_learning_curve(x, score_history, figure_file)


"""
SAC training script for PandaPickAndPlaceJoints‑v3
--------------------------------------------------
* Flat observations via `ObsOnlyWrapper`
* Dense reward, joints control
* Video recorded **κάθε 1 000 επεισόδια**
* Models αποθηκεύονται όταν ο µέσος όρος των 100 τελευταίων επεισοδίων βελτιώνεται
"""

# import os
# import warnings
# from pathlib import Path
#
# import gymnasium as gym
# from gymnasium.wrappers import RecordVideo
# import numpy as np
# import torch
#
# import panda_gym  # ⟵ registration side‑effect
#
# from flat_wrapper import ObsOnlyWrapper  # δικός σου wrapper (flat obs)
# from sac_torch import Agent
# from utils import plot_learning_curve
#
#
# # ----------------------------------------------------------------------------
# # 1.  Video trigger (κάθε 1000 επεισόδια)
# # ----------------------------------------------------------------------------
# VIDEO_INTERVAL = 1000
# video_folder = Path("tmp/video")
# video_folder.mkdir(parents=True, exist_ok=True)
#
# def video_trigger(episode_id: int) -> bool:
#     return episode_id % VIDEO_INTERVAL == 0
#
#
# # ----------------------------------------------------------------------------
# # 2.  Create environment (wrapped + video)
# # ----------------------------------------------------------------------------
# base_env = gym.make(
#     "PandaPickAndPlaceJoints-v3",
#     control_type="joints",
#     reward_type="dense",
#     render_mode="rgb_array",   # απαραίτητο για RecordVideo
# )
#
# env = RecordVideo(
#     ObsOnlyWrapper(base_env),
#     video_folder=str(video_folder),
#     episode_trigger=video_trigger,
# )
#
#
# # ----------------------------------------------------------------------------
# # 3.  Agent initialization
# # ----------------------------------------------------------------------------
# obs_shape = env.observation_space.shape  # e.g. (30,)
# agent = Agent(
#     input_dims=obs_shape,
#     env=env,
#     n_actions=env.action_space.shape[0],
# )
#
#
# # ----------------------------------------------------------------------------
# # 4.  Training loop
# # ----------------------------------------------------------------------------
# N_GAMES = 60_000
# BEST_SCORE = -np.inf
# scores, avg_scores = [], []
#
# print("CUDA available:", torch.cuda.is_available())
# print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
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
#         # store + learn
#         agent.remember(obs, action, reward, obs_, done)
#         agent.learn()
#
#         obs = obs_
#         score += reward
#
#     scores.append(score)
#     avg_score = np.mean(scores[-100:])
#     avg_scores.append(avg_score)
#
#     if avg_score > BEST_SCORE:
#         BEST_SCORE = avg_score
#         agent.save_models()
#
#     if episode % 10 == 0:
#         print(f"Episode {episode:>6} | Score: {score:>7.2f} | Avg100: {avg_score:>7.2f}")
#
# # ----------------------------------------------------------------------------
# # 5.  Plot learning curve
# # ----------------------------------------------------------------------------
# x = [i + 1 for i in range(len(scores))]
# plot_learning_curve(x, scores, "plots/sac_pickplace.png")
#
# print("Training finished. Videos saved to", video_folder)


# Two-Stage SAC Training: Reach ➜ PickAndPlace
# --------------------------------------------
# * 1η φάση: PandaReachDense-v3
# * 2η φάση: PandaPickAndPlaceJoints-v3
# * Χρήση flat παρατηρήσεων (ObsOnlyWrapper)
# * Καταγραφή βίντεο μόνο στη 2η φάση

# ---------------------------------------------
# main_twostage_partial.py
# ---------------------------------------------
"""
Two-Stage SAC Training with *Partial* Weight Transfer
Phase-1  : PandaReachDense-v3
Phase-2  : PandaPickAndPlaceJoints-v3
Transfer : Φορτώνουμε ΜΟΝΟ τα βάρη που ταιριάζουν σε σχήμα
           (τα fc1 layers επανεκκινούνται).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # quick fix για OpenMP

import warnings
from pathlib import Path
from typing import Dict
from stable_baselines3.common.logger import configure

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import panda_gym                         # register Panda envs
from flat_wrapper import ObsOnlyWrapper  # flattens obs dictionaries
from sac_torch import Agent
from utils import plot_learning_curve

# Example: save logs στο φάκελο "logs/"
log_dir = "logs/"
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# ------------------------------------------------------------------
# 0.  paths & hyper-params
# ------------------------------------------------------------------
REACH_GAMES      = 10_000
PICKPLACE_GAMES  = 50_000
CHECKPOINT_DIR   = Path("tmp/sac")      # όπου αποθηκεύει ο Agent
VIDEO_DIR        = Path("tmp/video")
PLOTS_DIR        = Path("plots")

for p in (CHECKPOINT_DIR, VIDEO_DIR, PLOTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Running 2-stage SAC training (partial transfer)...")
print("CUDA:", torch.cuda.is_available(),
      "| Device:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# ------------------------------------------------------------------
# helper: generic training loop
# ------------------------------------------------------------------
def train_agent(env, agent, n_games: int,
                plot_name: str,
                save_every: int = 100) -> None:
    """Train *agent* on *env* for *n_games* episodes."""

    scores, successes = [], []
    best = -np.inf
    early_stop = False

    for ep in range(n_games):
        obs, _ = env.reset()
        done, ep_return = False, 0.0
        ep_success = 0

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            agent.remember(obs, action, reward, obs_, done)
            agent.learn()

            obs = obs_
            ep_return += reward
            ep_success = info.get("is_success", ep_success)

        # Metrics
        scores.append(ep_return)
        successes.append(ep_success)
        avg_score = np.mean(scores[-100:])
        avg_success = np.mean(successes[-100:])

        # Logging
        if agent.logger is not None:
            agent.logger.record("rollout/ep_rew_mean", avg_score)
            agent.logger.record("rollout/success_rate", avg_success)
            agent.logger.record("time/episodes", ep)
            agent.logger.dump(step=ep)

        # Save models if best
        if avg_score > best:
            best = avg_score
            agent.save_models()

        # Entropy freeze if high success
        if agent.automatic_entropy_tuning and avg_success > 0.50:
            agent.automatic_entropy_tuning = False
            agent.alpha = agent.log_alpha.exp().item()
            print(f"🔒  Fixing α = {agent.alpha:.5f} (success={avg_success:.2f})")

        # Early stopping
        if avg_success > 0.95 and ep >= 200:
            print(f"🎯 Early stopping: success rate {avg_success:.2f} at episode {ep}")
            early_stop = True
            agent.save_models()
            break

        # Extra model saving
        if save_every and ep % save_every == 0:
            agent.save_models()

        # Progress print
        if ep % 10 == 0:
            print(f"Ep {ep:5d} | Return: {ep_return:7.2f} | "
                  f"Avg100: {avg_score:7.2f} | "
                  f"Succ100: {avg_success:.2f}")

    # Plot learning curve
    xs = list(range(1, len(scores) + 1))
    plot_learning_curve(xs, scores, str(PLOTS_DIR / plot_name))

    if early_stop:
        print("✅ Early Stopping Triggered.")

    # learning curve
    xs = list(range(1, len(scores) + 1))
    plot_learning_curve(xs, scores, str(PLOTS_DIR / plot_name))


# ------------------------------------------------------------------
# helper: partial-load (filter state-dict by matching shapes)
# ------------------------------------------------------------------
def _filter_state(target_net: torch.nn.Module,
                  preload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return only those params from *preload* που ταιριάζουν σε *target_net*."""
    keep = {}
    tgt_sd = target_net.state_dict()
    for k, v in preload.items():
        if k in tgt_sd and tgt_sd[k].shape == v.shape:
            keep[k] = v
    return keep


def load_partial_weights(pick_agent: Agent, ckpt_dir: Path = CHECKPOINT_DIR) -> None:
    files = {
        "actor":    ckpt_dir / "actor",
        "critic_1": ckpt_dir / "critic_1",
        "critic_2": ckpt_dir / "critic_2",
    }

    # actor
    sd = torch.load(files["actor"])
    pick_agent.actor.load_state_dict(_filter_state(pick_agent.actor, sd), strict=False)

    # critics
    for net_name in ("critic_1", "critic_2"):
        sd = torch.load(files[net_name])
        getattr(pick_agent, net_name).load_state_dict(
            _filter_state(getattr(pick_agent, net_name), sd), strict=False
        )

    print("✓ Partial weights loaded (non-matching layers re-initialized).")


# ------------------------------------------------------------------
# 1. Phase-1: Reach
# ------------------------------------------------------------------
print("\n— Phase 1: PandaReach-v3 —")
reach_env = ObsOnlyWrapper(gym.make(
    "PandaReach-v3",
    reward_type="sparse",
    render_mode="human"        # ή "rgb_array"
))
reach_agent = Agent(
    input_dims=reach_env.observation_space.shape,
    env=reach_env,
    n_actions=reach_env.action_space.shape[0],
    automatic_entropy_tuning=True,
    logger=new_logger
)
train_agent(reach_env, reach_agent, REACH_GAMES, "sac_reach.png")


# ------------------------------------------------------------------
# 2. Phase-2: Pick&Place (νέος agent + partial load)
# ------------------------------------------------------------------
print("\n— Phase 2: PandaPickAndPlaceJoints-v3 —")

def trigger(ep): return ep % 1000 == 0
# pick_env = RecordVideo(
#     ObsOnlyWrapper(gym.make(
#         "PandaPickAndPlaceJoints-v3",
#         control_type="joints",
#         reward_type="dense",
#         render_mode="rgb_array"
#     )),
#     video_folder=str(VIDEO_DIR),
#     episode_trigger=trigger
# )
pick_env = ObsOnlyWrapper(gym.make(
    "PandaPickAndPlaceJoints-v3",
    control_type="joints",
    reward_type="dense",
    render_mode="human"  # πιο γρήγορο από "rgb_array"
))

pick_agent = Agent(
    input_dims=pick_env.observation_space.shape,   # (π.χ.) 19-dim
    env=pick_env,
    n_actions=pick_env.action_space.shape[0],
    automatic_entropy_tuning=True,
    logger=new_logger
)

# φορτώνουμε ό,τι ταιριάζει από Reach
load_partial_weights(pick_agent)

# … και συνεχίζουμε το training
train_agent(pick_env, pick_agent, PICKPLACE_GAMES, "sac_pickplace.png")


# ------------------------------------------------------------------
# 3. Done
# ------------------------------------------------------------------
print("\n✅ Training complete.")
print("📈 Reach plot      :", PLOTS_DIR / "sac_reach.png")
print("📈 Pick&Place plot :", PLOTS_DIR / "sac_pickplace.png")
print("🎥 Videos @        :", VIDEO_DIR.resolve())

