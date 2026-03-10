python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

pip install -e .

python scripts/train.py --algo dqn/ppo

python scripts/eval.py --algo dqn/ppo --model-path "path_to_model"

python scripts/plot.py
