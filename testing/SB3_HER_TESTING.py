import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

# ---------------------------------------------------------------
# Custom Wrapper: Reach and Grasp only
# ---------------------------------------------------------------
class PandaReachAndGraspWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.grasp_threshold = 0.02  # Threshold για να θεωρείται grasp επιτυχία

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Πάρε gripper position και object position
        grip_pos = obs["observation"][0:3]  # x,y,z gripper
        object_pos = obs["observation"][3:6]  # x,y,z αντικειμένου
        dist = np.linalg.norm(grip_pos - object_pos)

        # Νέο reward
        if dist < self.grasp_threshold:
            reward = 1.0
            terminated = True  # Τερματισμός επεισοδίου μόλις γίνει grasp
        else:
            reward = -dist  # Αρνητικό reward όσο πιο μακριά είναι

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# ---------------------------------------------------------------
# Paths & Params
# ---------------------------------------------------------------
TOTAL_STEPS = 500_000
MODEL_DIR = Path("models/sac_her_reach_grasp")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("✅ Starting SAC+HER Reach & Grasp training")
print("📁 Models saved to:", MODEL_DIR.resolve())

# ---------------------------------------------------------------
# Helper: Train Agent
# ---------------------------------------------------------------
def train_sb3_agent(env, agent, total_timesteps: int, name: str):
    agent.learn(total_timesteps=total_timesteps, log_interval=10)
    agent.save(MODEL_DIR / name)

    mean_r, std_r = evaluate_policy(agent, env, n_eval_episodes=10)
    print(f"📊 {name} | Mean reward: {mean_r:.2f} ± {std_r:.2f}")

# ---------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------
print("\n— Creating Reach & Grasp Environment —")

base_env = gym.make(
    "PandaPickAndPlaceJoints-v3",
    control_type="joints",
    reward_type="dense",
    render_mode="human",
)

reach_grasp_env = Monitor(PandaReachAndGraspWrapper(base_env))

# ---------------------------------------------------------------
# SAC + HER Setup
# ---------------------------------------------------------------
sac_model = SAC(
    policy="MultiInputPolicy",
    env=reach_grasp_env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=GoalSelectionStrategy.FUTURE,
    ),
    verbose=1
)

# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------
train_sb3_agent(reach_grasp_env, sac_model, TOTAL_STEPS, "sac_her_reach_grasp")
reach_grasp_env.close()

print("\n✅ SAC + HER Reach & Grasp Training complete!")
print("📁 Model stored at:", MODEL_DIR.resolve())
