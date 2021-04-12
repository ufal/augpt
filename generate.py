#!/bin/env python
import logging
from typing import List
import torch
import argparse
import dataclasses
import itertools
import os
import shutil
from collections import OrderedDict
from data.utils import DialogDatasetItem
from data.utils import BeliefParser, InsertLabelsTransformation
from utils import pull_model, setup_logging
from pipelines import AuGPTConversation, get_context_from_conversation
import transformers
from tqdm import tqdm


def conversation_to_sample(conversation: AuGPTConversation):
    user = conversation.past_user_inputs
    if conversation.new_user_input is not None:
        user = user + [conversation.new_user_input]
    sys = conversation.generated_responses[:-1]
    context = get_context_from_conversation(user, sys)
    context = [x for x in itertools.chain(*itertools.zip_longest(user, sys)) if x is not None]
    database = OrderedDict((k, v[0]) for k, v in conversation.database_results.items())
    return DialogDatasetItem(context=context, belief=conversation.generated_belief,
                             raw_response=conversation.generated_responses[-1],
                             response=conversation.raw_response,
                             database=database)


def sample_to_conversation(sample, oracle_belief=False, oracle_database_results=False):
    conversation = AuGPTConversation()
    conversation.new_user_input = sample.context[-1]
    arr, other = conversation.generated_responses, conversation.past_user_inputs
    for utt in reversed(sample.context[:-1]):
        arr.append(utt)
        arr, other = other, arr
    arr.reverse()
    other.reverse()
    if oracle_belief:
        conversation.oracle_belief = sample.raw_belief
    if oracle_database_results:
        conversation.oracle_database_results = sample.database
    return conversation


def format_samples(samples):
    add_labels = InsertLabelsTransformation()
    formatted = []
    for i, sample in enumerate(samples):
        sample = dataclasses.replace(sample, context=[])
        sample = add_labels(sample)
        formatted.append('=>' + sample.belief + '<|eob|>' + sample.database +
                         '<|eokb|>' + sample.response + '<|endoftext|>')
    return formatted


@dataclasses.dataclass
class GeneratedPredictions:
    responses: List = dataclasses.field(default_factory=list)
    delex_responses: List = dataclasses.field(default_factory=list)
    beliefs: List = dataclasses.field(default_factory=list)
    gold_responses: List = dataclasses.field(default_factory=list)
    gold_delex_responses: List = dataclasses.field(default_factory=list)
    gold_beliefs: List = dataclasses.field(default_factory=list)

    def is_valid(self):
        n = len(self.responses)
        breakpoint()
        return len(self.delex_responses) == n and \
            len(self.beliefs) == n and \
            len(self.gold_responses) == n and \
            len(self.gold_delex_responses) == n and \
            len(self.gold_beliefs) == n

    @classmethod
    def load_predictions(cls, file, assert_valid=True):
        predictions = cls()
        parser = BeliefParser()
        for line in file:
            line = line.rstrip()
            if line.startswith('GT:'):
                predictions.gold_responses.append(line[len('GT:'):])
            elif line.startswith('GTD:'):
                predictions.gold_delex_responses.append(line[len('GTD:'):])
            elif line.startswith('BF:'):
                bf = line[len('BF:'):]
                bf = parser(bf)
                assert bf is not None
                predictions.beliefs.append(bf)
            elif line.startswith('RD:'):
                predictions.delex_responses.append(line[len('RD:'):])
            elif line.startswith('R:'):
                r = line[len('R:'):]
                predictions.responses.append(r)
            elif line.startswith('GBF:'):
                bf = line[len('GBF:'):]
                bf = parser(bf)
                assert bf is not None
                predictions.gold_beliefs.append(bf)
        if assert_valid:
            assert predictions.is_valid()
        return predictions


def generate_predictions(pipeline, dataset, output_file='predictions.txt', oracle_belief=False, orable_database_results=False):
    belief_parser = BeliefParser()
    add_labels = InsertLabelsTransformation('U:', 'S:', 'D:', 'BF:')
    gold_responses = []
    gold_beliefs = []
    responses = []
    delex_responses = []
    delex_gold_responses = []
    beliefs = []
    with open(output_file, 'w+') as fout:
        d = 0
        for i, sample in enumerate(tqdm(dataset, desc='generating predictions')):
            if len(sample.context) == 1:
                d += 1
                print(f'======== dialogue {d} ========', file=fout)
            print(f'U:{sample.context[-1]}', file=fout)
            print(f'GT:{sample.raw_response}', file=fout)
            print(f'GTD:{sample.response}', file=fout)
            print(f'GBF:{sample.belief}', file=fout)
            gold_beliefs.append(sample.belief)
            conversation = sample_to_conversation(sample, oracle_belief=oracle_belief, oracle_database_results=orable_database_results)
            gold_responses.append(sample.raw_response)
            delex_gold_responses.append(sample.response)
            conversation = pipeline(conversation)
            belief = conversation.generated_belief
            database = OrderedDict((d, v[0] if isinstance(v, tuple) else v) for d, v in conversation.database_results.items())
            sample = add_labels((sample.context, belief, database, conversation.generated_responses[-1], 1))
            print(sample.belief, file=fout)
            print(sample.database, file=fout)
            if pipeline.lexicalizer:
                print(f'R:{sample.response}', file=fout)
            else:
                print('R:', file=fout)
            print(f'RD:{conversation.raw_response}', file=fout)
            raw_belief = belief_parser(belief)
            beliefs.append(raw_belief)
            responses.append(conversation.generated_responses[-1])
            delex_responses.append(conversation.raw_response)
    return GeneratedPredictions(
        responses, delex_responses, beliefs, gold_responses, delex_gold_responses, gold_beliefs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='jkulhanek/augpt-mw-21')
    parser.add_argument('--file', default='predictions.txt')
    parser.add_argument('--dataset', default='multiwoz-2.1-test')
    parser.add_argument('--oracle-belief', action='store_true')
    parser.add_argument('--oracle-db', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger()

    if args.wandb:
        import wandb
        wandb.init(job_type='evaluation', config=args)
        args = argparse.Namespace(**wandb.config)

    model_name = pull_model(args.model)
    pipeline_kwargs = dict(lexicalizer=None)
    if args.oracle_db:
        pipeline_kwargs['database'] = None
    pipeline = transformers.pipeline('augpt-conversational', model_name, device=0 if torch.cuda.is_available() else -1, **pipeline_kwargs)

    # Generate
    from data import load_dataset
    dataset = load_dataset(args.dataset)
    generate_predictions(pipeline, dataset, args.file, oracle_belief=args.oracle_belief,
                         orable_database_results=args.oracle_db)

    # Copy file to wandb
    if args.wandb:
        fname = os.path.split(args.file)[-1]
        wandb_generated_path = os.path.join(wandb.run.dir, fname)
        shutil.copy(args.file, wandb_generated_path)
        wandb.save(fname)
