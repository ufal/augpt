#!/usr/bin/env python3

import logging
import argparse
from pipelines import AuGPTConversation
import transformers
from utils import setup_logging, seed
import tempfile


def interact(pipeline, args):
    conversation = None
    file = open(args.log, 'a+') if args.log is not None else tempfile.TemporaryFile('a+')
    with file:
        while True:
            if conversation is None:
                conversation = AuGPTConversation()
                print('=' * 20 + ' dialogue started ' + '=' * 20)
                print('=' * 20 + ' dialogue started ' + '=' * 20, file=file)
            user = input('user:')
            if user == 'reset':
                conversation = None
                continue
            else:
                print(f'user:{user}', file=file)

            conversation.add_user_input(user)
            conversation = pipeline(conversation)
            if args.debug:
                print(f'belief:{conversation.generated_belief}')
                print(f'db:{", ".join(f"{d}: {r[0]}" for d, r in conversation.database_results.items())}')
                print(f'raw:{conversation.raw_response}')

            elif args.raw:
                print(f'output:{conversation.raw_response}')
            print(f'sys:{conversation.generated_responses[-1]}')
            print(f'belief:{conversation.generated_belief}', file=file)
            print(f'db:{", ".join(f"{d}: {r[0]}" for d, r in conversation.database_results.items())}', file=file)
            print(f'raw:{conversation.raw_response}', file=file)
            print(f'output:{conversation.raw_response}', file=file)
            print(f'sys:{conversation.generated_responses[-1]}', file=file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--log', default=None, help='Path to a log file')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger()

    pipeline = transformers.pipeline('augpt-conversational', args.model)

    # Interact
    if args.seed is not None:
        seed(args.seed)
    interact(pipeline, args)
