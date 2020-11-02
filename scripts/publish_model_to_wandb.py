#!/bin/env python
import os
import sys
import argparse
import logging
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import setup_logging  # noqa: E402
from utils import pull_model  # noqa: E402


def publish_model(path, name=None):
    import wandb
    wandb.run.save()
    run_name = wandb.run.name.replace('+', '')
    if name is None:
        name = run_name
    artifact = wandb.Artifact(f'{name}-model', 'model')
    for f in os.listdir(path):
        if f.startswith('wandb-'):
            continue  # noqa: 701
        if f == 'output.log':
            continue  # noqa: 701
        if f == 'requirements.txt':
            continue  # noqa: 701
        if f.startswith('events.'):
            continue  # noqa: 701
        if os.path.isdir(os.path.join(path, f)):
            continue  # noqa: 701
        artifact.add_file(os.path.join(path, f), f)
    wandb.run.log_artifact(artifact, aliases=['latest', run_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--name', default=None, help='artifact name')
    args = parser.parse_args()
    run = args.run
    root = pull_model(run)
    setup_logging()
    logger = logging.getLogger()
    logger.info('publishing artifact')
    wandb.init(resume=run)
    publish_model(root, args.name)
    logger.info('model published')
