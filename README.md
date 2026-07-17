# PPO Dexterous Grasping & Push Task — PyBullet

A reinforcement learning project built on the DexMobile PyBullet framework, training a dexterous robotic hand to perform object manipulation tasks using Proximal Policy Optimization (PPO). Extended from the base framework with a push task, physics analysis tools, and training visualization.

For the full 8-page IEEE paper I wrote about my findings refer to:
[Final Project .pdf](https://github.com/user-attachments/files/30109230/Final.Project.pdf)

---

## What This Project Does

The robot is a multi-fingered dexterous hand simulated in PyBullet. It learns to manipulate objects on a table through two categories of tasks:

**Grasping tasks** — the hand reaches an object, closes around it with a specified grasp topology, lifts it off the table, and holds it stably.

**Push task (`platform`)** — the arm approaches a bottle from the side and pushes it off the edge of the table. This task uses a simplified reward structure focused on arm-x progress and bottle displacement rather than finger closure.

---

## Additions Made to the Base Framework

### Push Task
A new `"platform"` grasp type was added to `Dualenv.py` that repurposes the existing two-stage environment into a pure push task:
- The arm targets a point 15 cm past the bottle in the x-direction, maintaining constant forward pressure
- Success is triggered when the bottle's z position drops below the table surface (`z < 0.05`)
- Failure is triggered only if the bottle falls off the table entirely (`z < -0.1`)
- The push reward function (`reward_s1`) combines arm-x progress, bottle-x displacement, hand-bottle contact, and an alignment penalty on y/z deviation

### Physics Analysis
Three functions were added to `Dualenv.py` for contact and stability analysis:

- `check_equilibrium()` — computes total force and torque balance across all contact points between the hand and object, used as part of the observation space and stage-2 reward
- `in_friction_cone()` — checks whether all contact forces satisfy the friction cone constraint given a coefficient of friction (default 0.6); a friction-cone bonus is added to the success reward
- `get_finger_contact_forces()` / `print_finger_forces_on_new_contact()` — compute per-finger contact force vectors (Fx, Fy, Fz) and print them whenever a new finger makes contact during a simulation step

### Training Visualization (`train.py`)
After training completes, a convergence graph is automatically generated and saved to `log/convergence_graph.png`:
- Plots raw episode rewards and a smoothed rolling average
- Detects the convergence point (first episode where smoothed reward stays within 5% of the final value)
- Annotates the convergence episode with an arrow on the plot

### Evaluation Visualization (`evaluate.py`)
After evaluation runs, a finger contact bar graph is saved to `log/finger_contact_graph.png`:
- Counts total contact steps per finger part (Palm, Thumb, Index, Middle, Ring, Pinky) across all evaluation episodes
- Displays success rate in a text box on the plot

---

## Grasp Types

| Type | Fingers Used |
|---|---|
| `platform` | Push task — no finger closure |
| `inSiAd2` | Thumb only |
| `pPdAb2` | Thumb + Index |
| `pPdAb23` | Thumb + Index + Middle |
| `pPdAb25` | All five fingers |
| `poPmAb25` | All five fingers (palm-opposed) |

---

## Environment Details

- **Simulator:** PyBullet at 240 Hz
- **Action space:** 8-dimensional continuous (`[-1, 1]`) — 3 for end-effector xyz translation (stage 1), 5 for finger joints (stage 2)
- **Observation space:** 50-dimensional — relative positions and orientations of palm and all five fingertips to the target, distance to target, in-position flags, friction cone flag, force/torque balance
- **Reward shaping:** staged — stage 1 rewards moving the hand to the target position; stage 2 rewards contact quality; success gives +100,000; timeout gives −20,000; out-of-range gives −10,000; object slip gives +40,000

---

## Setup

```bash
conda create -n dexmobile python=3.10.12 -y
conda activate dexmobile
python -m pip install --upgrade pip
pip install gym==0.26.1 gym-notices==0.1.0 gymnasium==1.2.3 Shimmy==2.0.0 \
    numpy==1.26.4 pandas==2.3.3 pybullet==3.2.7 stable-baselines3==2.7.1 setuptools==80.9.0
```

---

## Usage

**Train:**
```bash
cd DexMobile
python train.py
```

Configure the grasp type and timesteps at the top of `train.py`:
```python
graspType = "poPmAb25"   # or "platform" for push task
timeStep  = step * 5000  # total training steps
```

**Evaluate:**
```bash
python evaluate.py
```

Loads the saved model and runs 1000 test episodes with PyBullet GUI rendering enabled.

---

## PPO Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 |
| Gamma | 0.99 |
| Clip range | 0.2 |
| Batch size | 64 |
| Entropy coefficient | 0.01 |
| Reward normalization | VecNormalize (training only) |

A custom `successRateCallBack` checkpoints the model every 50,000 steps based on success rate over the last 100 episodes and stops training early once the target rate (99%) is reached.

---

## Output Files

| File | Description |
|---|---|
| `log/<graspType>.zip` | Trained PPO model |
| `log/<graspType>.pkl` | Saved VecNormalize statistics |
| `log/convergence_graph.png` | Training reward curve with convergence annotation |
| `log/finger_contact_graph.png` | Per-finger contact frequency from evaluation |
| `log/inSiAd2.csv` | Episode log (reward, success flag, fail reason) |
