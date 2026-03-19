# GYMNASIUM/BREAKOUT

## Installation

1. python -m venv venv

2. source venv/bin/activate

3. pip install -r requirements.txt

4. pip install -e .

## Training

### Training from scratch

python scripts/train.py --algo dqn/ppo

### Training from the exiting 

python scripts/train.py --algo dqn/ppo --resume "Path to model.zip to continue training"

## Evaluating

python scripts/eval.py --algo dqn/ppo --model-path "path_to_model"

## Plotting

python scripts/plot.py --log-path "path_to_logfile"
