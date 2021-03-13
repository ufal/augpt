#!/bin/env python
import os
import argparse
import logging
import torch
import transformers
from utils import setup_logging, pull_model  # noqa:E402
import data.evaluation.multiwoz.convlab  # noqa: E402
import nltk


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--num_dialogs', type=int, default=1000)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--num-beams', type=int, default=None)
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger()
    if args.resume is not None and args.model is None:
        args.model = f'wandb:{args.resume}'
        args.model = pull_model(args.model)

    # Update punkt
    nltk.download('punkt')

    if args.resume:
        import wandb

        # Resume run and fill metrics
        os.environ.pop('WANDB_NAME', None)
        wandb.init(resume=args.resume)
    elif args.wandb:
        import wandb

        # It is an artifact
        # Start a new evaluate run
        wandb.init(job_type='evaluation', config=args)
        args = argparse.Namespace(**wandb.config)
    else:
        wandb = None

    analyzer = data.evaluation.multiwoz.convlab.ConvLabAnalyzer()
    pipeline = transformers.pipeline('augpt-conversational', args.model, device=0 if torch.cuda.is_available() else -1)
    if args.num_beams is not None:
        pipeline.model.config.num_beams = args.num_beams

    # Analyze
    result = analyzer(pipeline, args.num_dialogs)
    if wandb and wandb.run:
        wandb.run.summary.update(result)
