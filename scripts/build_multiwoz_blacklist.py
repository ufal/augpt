#!/bin/env python
import argparse
import logging
from collections import Counter
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.evaluation import MultiWozEvaluator  # noqa: E402
from data import load_dataset  # noqa: E402
from data.utils import format_belief, wrap_dataset_with_cache  # noqa: E402
from utils import setup_logging  # noqa: E402


DATASETS_PATH = os.path.join(os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets')), 'augpt')


def fix_goal(goal, belief):
    goal = {k: v for k, v in goal.items() if k in belief}
    return goal


def enumerate_multiwoz_invalid_indices(dataset_name, logger, correct_requestables=False):
    dataset = load_dataset(dataset_name, use_goal=True)
    dataset = wrap_dataset_with_cache(dataset)
    evaluator = MultiWozEvaluator(dataset,
                                  logger=logger,
                                  is_multiwoz_eval=True)
    responses = (item.response for item in dataset)
    beliefs = (evaluator.belief_parser(item.belief) for item in dataset)
    dialogues = evaluator.pack_dialogues(dataset, beliefs, responses)
    successes, matches = 0, 0
    stats = tuple(Counter() for _ in range(3))
    domain_total, domain_match, domain_success = stats
    total = 0

    offset = 0
    with tqdm(total=len(dataset), desc='identifying bad dialogues') as progress:
        for idx, (items, goal, beliefs, responses, booked_domains) in enumerate(dialogues):
            goal, real_requestables = evaluator._get_goal_and_requestables(items[-1].raw_belief, goal)
            goal = fix_goal(goal, beliefs[-1])
            provided_requestables, venue_offered = evaluator._get_requestables_and_venues(
                beliefs, responses, booked_domains)
            success, match = evaluator._evaluate_generated_dialogue(
                real_requestables, provided_requestables, venue_offered, goal, stats)
            if match != 1 or (success != 1 and correct_requestables):
                for i in range(offset, offset + len(items)):
                    yield f'{i}'
            elif len(set(map(format_belief, beliefs))) == 1 and len(items) > 1:
                match, success = 0, 0
                for i in range(offset, offset + len(items)):
                    yield f'{i}'

            successes += success
            matches += match
            total += 1
            offset += len(items)
            progress.update(len(items))

        domain_results = dict()
        for key in domain_total.keys():
            domain_results[key] = domain_match[key] / float(domain_total[key]), \
                domain_success[key] / float(domain_total[key])

        match, success = matches / float(total), successes / float(total)
        logger.info(f'match: {match:.4f}, success: {success:.4f}')
        for domain, (match, success) in domain_results.items():
            logger.info(f'   - domain: {domain}, match: {match:.4f}, success: {success:.4f}')


if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--correct-requestables', action='store_true')
    parser.add_argument('--dataset', default='multiwoz-2.1')
    args = parser.parse_args()
    if args.upload:
        import wandb
        wandb.init(job_type='preprocessing')
        artifact = wandb.Artifact(f'{args.dataset}-blacklist', 'dataset')
    else:
        wandb = None
    sets = ['test', 'val', 'train']
    # sets = ['train']
    for dataset_type in sets:
        if wandb and wandb.run:
            f = artifact.new_file(f'{dataset_type}-blacklist.txt')
        else:
            f = open(os.path.join(DATASETS_PATH, args.dataset, f'{dataset_type}-blacklist.txt'), 'w+')
        with f:
            for bad_session_id in enumerate_multiwoz_invalid_indices(f'{args.dataset}-{dataset_type}', logger, correct_requestables=args.correct_requestables):
                print(bad_session_id, file=f)
                f.flush()
    if wandb and wandb.run:
        wandb.run.log_artifact(artifact)
