#!/bin/env python
import argparse
import os
import sys
import logging
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset, load_backtranslation_transformation  # noqa:E402
from utils import setup_logging  # noqa:E402


def validate_coverage(dataset, backtranslations, logger):
    texts = set()
    for d in dataset:
        for c in d.context:
            texts.add(c)

    missing_texts = 0
    for text in tqdm(texts):
        if text not in backtranslations:
            missing_texts += 1
            logger.debug(f'missing: "{text}"')
    logger.info(f'missing {100 * missing_texts / len(texts):.2f}% of the dataset')


def main():
    setup_logging()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--backtranslations', default=None)
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    transform = load_backtranslation_transformation(args.backtranslations or args.dataset)
    validate_coverage(dataset, transform.dictionary, logger)


if __name__ == '__main__':
    main()
