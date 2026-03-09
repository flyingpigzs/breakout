import yaml
from pathlib import Path


def load_yaml(path: str | Path):
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(env_path, algo_path, train_path):

    cfg = {}

    env_cfg = load_yaml(env_path)
    algo_cfg = load_yaml(algo_path)
    train_cfg = load_yaml(train_path)

    cfg.update(env_cfg)
    cfg.update(algo_cfg)
    cfg.update(train_cfg)

    return cfg