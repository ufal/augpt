#!/bin/env python
import os
import sys
import argparse
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
old_stdout = sys.stdout
# sys.stdout = open(os.devnull, 'w')
sys.stdout = sys.stderr

import data  # noqa:E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='multiwoz-train+multiwoz-dev+multiwoz-test')
    args = parser.parse_args()

    items = set()
    dataset = data.load_dataset(args.dataset)
    for i, sample in enumerate(tqdm(dataset)):
        for text in sample.context:
            if text not in items:
                items.add(text)
                print(text, file=old_stdout)
