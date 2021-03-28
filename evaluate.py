#!/bin/env python
import argparse
import logging
from itertools import chain
from collections import defaultdict
from utils import setup_logging, pull_model  # noqa:E402
from utils import Mean, F1
from data.utils import BeliefParser
from evaluation_utils import compute_bleu, compute_rouge
from generate import GeneratedPredictions


def flatten_belief_state(bs):
    return {f'{d}:{k}': v for d, pbs in bs.items() for k, v in pbs.items()}


def compute_tp_fp_fn(gbs, bs):
    tp = len(gbs_items.union(bs_items))
    fp = len(bs_items - gbs_items))
    fn = len(gbs_items - bs_items))
    return tp, fp, fn


def evaluate_belief_accuracies(beliefs, gold_beliefs):
    acc_slot = Mean()
    acc_joint = Mean()
    f1 = Mean()
    domain_accs = defaultdict(lambda: (Mean(), Mean(), F1()))
    for bs, gbs in zip(beliefs, gold_beliefs):
        for domain in set(chain(bs.keys(), gbs.keys())):
            domain_acc_slot, domain_acc_joint, domain_f1 = domain_accs[domain]
            gbs_data = set(gbs.get(domain, dict()).items())
            bs_data = set(bs.get(domain, dict()).items())
            domain_acc_slot.update_state(len(gbs_data.union(bs_data)) / len(gbs_data), weight=len(gbs_data))
            domain_acc_joint.update_state(len(gbs_data.union(bs_data)) == len(gbs_data))
            domain_f1.update_state(*compute_tp_fp_fn(gbs_data, bs_data))

        gbs_data = set(flatten_belief_state(gbs).items())
        bs_data = set(flatten_belief_state(bs).items())
        acc_slot.update_state(len(gbs_data.union(bs_data)) / len(gbs_data), weight=len(gbs_data))
        acc_joint.update_state(len(gbs_data.union(bs_data)) == len(gbs_data))
        f1.update_state(*compute_tp_fp_fn(gbs_data, bs_data))


    result = dict(acc_avg=acc_avg(), acc_joint=acc_joint(), f1=f1())
    for d, (avg, joint, f1) in domain_accs.items():
        result[f'acc_avg_{d}'] = avg()
        result[f'acc_joint_{d}'] = joint()
        result[f'f1_{d}'] = f1()
    return result


def analyze(predictions: GeneratedPredictions):
    parsed_beliefs = list(map(BeliefParser(), predictions.beliefs))
    result = evaluate_belief_accuracies(parsed_beliefs, predictions.gold_beliefs)
    bleu = compute_bleu(predictions.delex_responses, predictions.gold_delex_responses)
    rouge = compute_rouge(predictions.delex_responses, predictions.gold_delex_responses)
    result['bleu'] = bleu
    result['rouge'] = rouge
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger()
    if args.resume is not None and args.model is None:
        args.model = f'wandb:{args.resume}'
        args.model = pull_model(args.model)

    if args.wandb:
        import wandb

        # It is an artifact
        # Start a new evaluate run
        wandb.init(job_type='evaluation', config=args)
        args = argparse.Namespace(**wandb.config)
    else:
        wandb = None

    # Analyze
    with open(args.input, 'r') as f:
        predictions = GeneratedPredictions.load_predictions(f)
    result = analyze(predictions)
    if wandb and wandb.run:
        wandb.run.summary.update(result)
