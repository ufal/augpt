#!/bin/env python
import logging
import argparse
from pipelines import AuGPTConversation
import transformers
from utils import setup_logging, seed


def interact(pipeline, args):
    conversation = AuGPTConversation()
    while True:
        user = input('user:')
        conversation.add_user_input(user)
        conversation = pipeline(conversation)
        if args.debug:
            print(f'belief:{conversation.generated_belief}')
            print(f'db:{", ".join(f"{d}: {len(r)}" for d, r in conversation.database_results.items())}')
            print(f'raw:{conversation.raw_response}')
        elif args.raw:
            print(f'output:{conversation.raw_response}')
        print(f'sys:{conversation.generated_responses[-1]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger()

    pipeline = transformers.pipeline('augpt-conversational', args.model)

    # Interact
    if args.seed is not None:
        seed(args.seed)
    interact(pipeline, args)
