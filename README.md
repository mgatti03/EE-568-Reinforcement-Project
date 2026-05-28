# EE-568 — Reinforcement Learning from Human Feedback

This repository contains the code and experiments for an EE-568 project on **Reinforcement Learning from Human Feedback (RLHF)** in simple control environments.

We compare two preference-based fine-tuning methods:

* **PPO-RLHF**: learns a reward model from pairwise preferences, then runs PPO using the learned reward.
* **DPO**: directly optimizes the policy from preference pairs without learning an explicit reward model.

Experiments are run on:

* **CartPole-v1**, a discrete-action balancing task.
* **Pendulum-v1**, a continuous-action torque-control task.

The main goal is to study how both methods behave as the amount of preference data increases.

---

## Repository Structure

```text
.
├── exploration_pi1_vs_pi2_local.ipynb
├── policies_saving.ipynb
├── preference_data.ipynb
├── rlhf_training_cartpole.ipynb
├── rlhf_training_pendulum.ipynb
├── Poster.pdf
└── outputs/
    ├── checkpoints/
    ├── datasets/
    └── part3/
```

---

## Notebooks

### `exploration_pi1_vs_pi2_local.ipynb`

Explores candidate environments and training regimes.

Environments considered include:

* CartPole-v1
* Pendulum-v1
* Acrobot-v1
* MountainCar-v0
* MountainCarContinuous-v0
* LunarLander-v3

The objective is to find environments where PPO can reliably produce both an expert policy and a stable intermediate policy.

---

### `policies_saving.ipynb`

Trains PPO policies and saves selected `π1` and `π2` checkpoints.

CartPole uses PPO with default-style hyperparameters. Pendulum requires longer rollouts and a higher learning rate to obtain stable learning.

---

### `preference_data.ipynb`

Generates pairwise trajectory preference datasets.

Each preference pair contains:

* a trajectory from `π1`;
* a trajectory from `π2`;
* cumulative true rewards;
* a binary preference label sampled with the Bradley–Terry model;
* the shared random seed used to initialize both trajectories.

---

### `rlhf_training_cartpole.ipynb`

Runs reward-model training, PPO-RLHF, DPO, and evaluation for CartPole-v1.

CartPole is useful as a discrete-action benchmark where PPO-RLHF can recover expert-level behavior.

---

### `rlhf_training_pendulum.ipynb`

Runs the same RLHF pipeline for Pendulum-v1.

Pendulum is a continuous-action benchmark with a denser but more difficult reward structure. It is used to test whether the same RLHF methods remain stable outside discrete control.

---

## Poster

The project poster is available here: [Poster.pdf](Poster.pdf)

It summarizes the motivation, method, experimental setup, and main results.

---

## Installation

Create a Python environment and install the dependencies:


```bash
pip install stable-baselines3 gymnasium torch numpy matplotlib pandas
```

Depending on the environment, you may also need:

```bash
pip install swig
pip install gymnasium[box2d]
```

for Box2D-based environments such as LunarLander.

---

## Reproducing the Experiments

Run the notebooks in the following order:

```text
1. policies_saving.ipynb
2. preference_data.ipynb
3. rlhf_training_cartpole.ipynb
4. rlhf_training_pendulum.ipynb
```

A typical workflow is:

1. Train and save an expert policy `π1` and a weaker policy `π2`.
2. Roll out both policies with matched seeds to collect paired trajectories.
3. Build preference datasets using Bradley–Terry labelling for different values of `K`.
4. Run PPO-RLHF and DPO starting from `π2`.
5. Evaluate both methods over 20 held-out seeds on CartPole-v1 and Pendulum-v1.


---

## Notes

Generated outputs are not included in the repository. Checkpoints, preference datasets, reward models, plots, and training logs can be large and are meant to be reproduced locally by running the notebooks. All generated files are saved under outputs/.

---

## Authors

* Loïc Misenta
* Martina Gatti
* Bryan Gotti

EPFL — EE-568
