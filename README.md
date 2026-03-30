# Gymnasium Breakout RL Project

This project implements reinforcement learning agents (DQN and PPO) for the Atari Breakout environment using Gymnasium and Stable-Baselines3.

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
```

2.	Activate the environment:

```bash
source venv/bin/activate
```

3.	Install dependencies:

```bash
pip install -r requirements.txt
```

4.	Install the project in editable mode:

```bash
pip install -e .
```

## Training

### Train from scratch

```bash
python scripts/train.py --algo dqn
# or
python scripts/train.py --algo ppo
```

### Resume training from a checkpoint

```bash
python scripts/train.py --algo ppo --resume "path/to/model.zip"
```

## Evaluation

Evaluate a trained model:

```bash
python scripts/eval.py --algo ppo --model-path "path/to/model.zip"
```

## Plotting

Visualize training logs:

```bash
python scripts/plot.py --log-path "path/to/logfile"
```

## Project Structure

```
.
├── scripts/
│   ├── train.py        # Training entry point (DQN/PPO)
│   ├── eval.py         # Model evaluation
│   └── plot.py         # Training log visualization
│
├── src/
│   ├── envs/
│   │   ├── wrappers.py # Custom wrappers (EpisodicLifeEnv, FireResetEnv, etc.)
│   │   └── make_env.py # Train and eval environment creation
│   │
│   ├── callbacks/
│   │   └── callbacks.py # Evaluation and checkpoint callbacks
│   │
│   ├── models/
│   │   └── build_model.py # Model initialization
│   │
│   └── utils/            # Helper utilities
│
├── logs/                 # Training logs and evaluation outputs
├── checkpoints/          # Saved checkpoints
├── best_model/           # Best model (based on evaluation metric)
│
├── requirements.txt
├── setup.py
└── README.md
```

## Notes
	•	Supports both DQN and PPO via Stable-Baselines3
	•	Includes custom environment wrappers for improved training stability
	•	Supports checkpoint-based training and multi-stage experimentation
	•	Evaluation includes mean, standard deviation, and percentile metrics

## Future Work
	•	Multi-branch training from different checkpoints
	•	Hyperparameter scheduling
	•	Improved exploration strategies
