"""
2-phase SAC training σε PandaReachDense-v3 και PandaPickAndPlaceJoints-v3
χωρίς καταγραφή βίντεο
"""

import os
from pathlib import Path

import gymnasium as gym
import panda_gym  # noqa: F401  (απλώς για να εγγραφούν τα envs)
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# ---------------------------------------------------------------
# Paths & params
# ---------------------------------------------------------------
REACH_STEPS = 30_000
PICKPLACE_STEPS = 800_000
MODEL_DIR = Path("models/sac")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("✅ Starting 2-phase SAC training (NO video recording)")
print("📁 Models will be saved to:", MODEL_DIR.resolve())

# ---------------------------------------------------------------
# Helper: Train SB3 Agent
# ---------------------------------------------------------------
def train_sb3_agent(env, agent, total_timesteps: int, name: str):
    agent.learn(total_timesteps=total_timesteps, log_interval=10)

    agent.save(MODEL_DIR / name)
    mean_r, std_r = evaluate_policy(agent, env, n_eval_episodes=10)
    print(f"📊 {name} | Mean reward: {mean_r:.2f} ± {std_r:.2f}")

# ---------------------------------------------------------------
# Phase 1 – PandaReachDense-v3
# ---------------------------------------------------------------
print("\n— Phase 1: PandaReachDense-v3 —")

reach_env = Monitor(
    gym.make(
        "PandaReachDense-v3",
        reward_type="dense",
        render_mode="human",  # GUI για οπτική επιβεβαίωση (χωρίς video)
    )
)

sac_reach = SAC("MultiInputPolicy", reach_env, verbose=1)
train_sb3_agent(reach_env, sac_reach, REACH_STEPS, "sac_reach")

# Κλείσε το env για να ελευθερωθεί το PyBullet GUI πριν ανοίξεις άλλο
reach_env.close()

# ---------------------------------------------------------------
# Phase 2 – PandaPickAndPlaceJoints-v3
# ---------------------------------------------------------------
print("\n— Phase 2: PandaPickAndPlaceJoints-v3 —")

pick_env = Monitor(
    gym.make(
        "PandaPickAndPlaceJoints-v3",
        control_type="joints",
        reward_type="dense",
        render_mode="human",   # και εδώ GUI, ακόμη καμιά καταγραφή
    )
)

sac_pick = SAC("MultiInputPolicy", pick_env, verbose=1)
train_sb3_agent(pick_env, sac_pick, PICKPLACE_STEPS, "sac_pickplace")

pick_env.close()

print("\n✅ Training complete.")
print("📁 Trained models stored in:", MODEL_DIR.resolve())
