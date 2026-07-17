from Dualenv import Dualenv
from stable_baselines3 import PPO
from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Monitor import Monitor
from successRateCallBack import successRateCallBack
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

################################################# Define Variables ##################################################
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """ Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes
    current learning rate depending on remaining progress """

    def func(progress_remaining: float) -> float:
        """ Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate """
        # result = progress_remaining * initial_value
        if progress_remaining >= 0.6:
            return initial_value * 1
        if 0.6 > progress_remaining >= 0.3:
            return initial_value * 0.9
        else:
            return initial_value * 0.8

    return func


n_envs = 1
step = 1024
timeStep = step * 5000  # quick test run — change to step * 5000 for full training
orientation = 1  # 0 from side, 1 from above, 2 from above 1
graspType = "poPmAb25"
log_dir = "log"
fileName = log_dir + "/episodeData"
modelName = log_dir + "/" + graspType
envName = log_dir + "/" + graspType + ".pkl"

################################################# Training and Evaluation ############################################
# create environment and custom callback
env = Dualenv(renders=False, is_discrete=False, max_steps=step)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_reward=True)  # when training norm_reward = True
success = successRateCallBack(successRates=0.99, verbose=1, check_freq=step * 50, path=log_dir, n_eval_episodes=100)
# load and train the model
#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, learning_rate=linear_schedule(3e-5), gamma=0.99,
#            gae_lambda=0.95, clip_range=0.2, batch_size=64)

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, learning_rate=1.6e-6, gamma=0.985,
#             use_sde=False, sde_sample_freq=1, clip_range=0.2, batch_size=32)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, learning_rate=3e-4, gamma=0.99,
            clip_range=0.2, batch_size=64, ent_coef=0.01)
model.learn(timeStep, callback=success)
# save training info
#episodeData = model.get_env().get_attr("evaluation")
#episodeData = np.array(episodeData, dtype=object)
#np.save(fileName, episodeData)
# save model
model.save(modelName)
env.save(envName)
env.close()

################################################# Convergence Graph ##################################################
# load episode rewards from monitor log
log_data = pd.read_csv(log_dir + "/inSiAd2.csv")
rewards = log_data["r"].values
episodes = np.arange(1, len(rewards) + 1)

# smooth rewards with a rolling average for clearer trend
window = max(10, len(rewards) // 20)
smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values

# detect convergence: first episode where smoothed reward stays within 5% of final value
final_val = smoothed[-1]
threshold = 0.05 * abs(final_val) if final_val != 0 else 1.0
converge_idx = len(smoothed) - 1
for i in range(len(smoothed) - 1, -1, -1):
    if abs(smoothed[i] - final_val) > threshold:
        converge_idx = i + 1
        break

# plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(episodes, rewards, color="cornflowerblue", alpha=0.65, linewidth=0.8, label="Episode Reward")
ax.plot(episodes, smoothed, color="steelblue", linewidth=2, label=f"Smoothed (window={window})")

# arrow pointing to convergence point
ax.annotate(
    f"Convergence\n(Episode {converge_idx})",
    xy=(episodes[converge_idx], smoothed[converge_idx]),
    xytext=(episodes[converge_idx] + len(episodes) * 0.08, smoothed[converge_idx] - abs(final_val) * 0.3),
    fontsize=10,
    color="red",
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red")
)
ax.axvline(x=episodes[converge_idx], color="red", linestyle="--", alpha=0.5)

ax.set_title("PPO Training Convergence — Bottle 3422 Push Task", fontsize=13)
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Reward", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(log_dir + "/convergence_graph.png", dpi=150)
plt.show()
print("Convergence graph saved to log/convergence_graph.png")
