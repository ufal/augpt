#!/bin/env python
import logging
import torch
import argparse
import dataclasses
import itertools
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


def sample_to_conversation(sample):
    conversation = AuGPTConversation()
    conversation.new_user_input = sample.context[-1]
    arr, other = conversation.generated_responses, conversation.past_user_inputs
    for utt in reversed(sample.context[:-1]):
        arr.append(utt)
        arr, other = other, arr
    arr.reverse()
    other.reverse()
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


def generate_predictions(pipeline, dataset, output_file='predictions.txt'):
    belief_parser = BeliefParser()
    add_labels = InsertLabelsTransformation('U:', 'S:', 'D:', 'BF:')
    gold_responses = []
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
            conversation = sample_to_conversation(sample)
            gold_responses.append(sample.raw_response)
            delex_gold_responses.append(sample.response)
            conversation = pipeline(conversation)
            belief = conversation.generated_belief
            database = OrderedDict((d, v[0]) for d, v in conversation.database_results.items())
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
    return responses, beliefs, gold_responses, delex_responses, delex_gold_responses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='jkulhanek/augpt-mw-21')
    parser.add_argument('--file', default='predictions.txt')
    parser.add_argument('--dataset', default='multiwoz-2.1-test')
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger()

    model_name = pull_model(args.model)
    pipeline = transformers.pipeline('augpt-conversational', device=0 if torch.cuda.is_available() else -1)

    # Generate
    from data import load_dataset
    dataset = load_dataset(args.dataset)
    generate_predictions(pipeline, dataset, args.file)
