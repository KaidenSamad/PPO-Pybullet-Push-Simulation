
from Dualenv import Dualenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np

################################################# Define Variables ########################################################################
orientation = 1 # 0 from side, 1 from above, 2 from above 1
graspType = "poPmAb25"
log_dir = "log"
vName = graspType
modelName = log_dir + "/" + vName
envName = log_dir + "/" + vName + ".pkl"
################################################# Testing and Evaluation #################################################################

env = Dualenv(renders=True, is_discrete=False, max_steps=1024)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env = VecNormalize.load(envName, env)
env.training = False  # not continue training the model while testing
env.norm_reward = False  # reward normalization is not needed at test time
# load model 
model = PPO.load(modelName, env=env)

test = 1000  # quick test run — change to 1000 for final evaluation
successes = 0

# track how many steps each finger part contacts the bottle
finger_names = ["Palm", "Thumb", "Index", "Middle", "Ring", "Pinky"]
finger_counts = [0, 0, 0, 0, 0, 0]

for i in range(test):
	obs = env.reset()
	done = False
	rewards = float('-inf')
	while (not done):
		action, _states = model.predict(obs)
		obs, rewards, done, info = env.step(action)
		# record which fingers contacted the bottle this step
		contact = env.env_method('contactInfo', 1)[0]
		for j in range(6):
			finger_counts[j] += contact[j]
	if 'episode' in info[0] and info[0]['episode']['s'] == 1:
		successes += 1
	print(f"Episode {i+1}/{test} — {'SUCCESS' if info[0]['episode']['s'] == 1 else 'FAIL'} "
	      f"| fail reason: {info[0]['episode']['f']}")

print(f"\nSUCCESS RATE: {successes}/{test} ({(successes/test)*100:.1f}%)")
env.close()

################################################# Finger Contact Bar Graph #################################################
fig, ax = plt.subplots(figsize=(9, 6))

colors = ["steelblue", "mediumseagreen", "tomato", "mediumpurple", "darkorange", "deeppink"]
bars = ax.bar(finger_names, finger_counts, color=colors, edgecolor="black", linewidth=0.8)

# label the count above each bar
for bar, count in zip(bars, finger_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(finger_counts) * 0.01,
            str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")

# text box showing successful trials
ax.text(0.98, 0.95,
        f"Successful Trials: {successes}/{test}\nSuccess Rate: {(successes/test)*100:.1f}%",
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="black"))

ax.set_title("Finger Contact Frequency with Bottle 3422 — Push Task", fontsize=13)
ax.set_xlabel("Finger / Hand Part", fontsize=11)
ax.set_ylabel("Total Contact Steps", fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(log_dir + "/finger_contact_graph.png", dpi=150)
plt.show()
print("Finger contact graph saved to log/finger_contact_graph.png")

############### write to txt #######################################

# fileName = "log/" + str((sus[0]/test)*100) + ".txt"
#
# with open(fileName, 'w') as f:
# 	f.write(str((sus[0]/test)*100))













