"""Runs a W&B sweep from a YAML file, in this Python process.

Example: python -m src.sweep configs/a1_lr.yaml
"""

import argparse
import os

import wandb
import yaml

from src.paths import WANDB_DIR
from src.train import WANDB_PROJECT, main


def to_arguments(settings: dict) -> list[str]:
    """Turns sweep settings into the command-line arguments of src.train."""
    arguments = ["--wandb"]
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            arguments.append(flag)
        elif isinstance(value, list):
            arguments += [flag, *map(str, value)]
        elif value is not False:
            arguments += [flag, str(value)]
    return arguments


def run_one(group: str):
    run = wandb.init(dir=WANDB_DIR, group=group)
    main(to_arguments(dict(run.config)))
    wandb.finish()


def run_sweep(path: str):
    with open(path, encoding="utf-8") as file:
        sweep = yaml.safe_load(file)
    WANDB_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(WANDB_DIR)
    sweep_id = wandb.sweep(sweep, project=WANDB_PROJECT)
    group = sweep["parameters"]["arch"]["value"]
    wandb.agent(sweep_id, function=lambda: run_one(group), project=WANDB_PROJECT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    run_sweep(parser.parse_args().config)
