#!/bin/env python
import argparse
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='backtranslations.yaml')
    parser.add_argument('source')
    parser.add_argument('alternative', nargs='+')
    args = parser.parse_args()

    source = open(args.source, 'r').readlines()
    alternatives = [open(x, 'r').readlines() for x in args.alternative]
    dictionary = {x[0].rstrip('\n'): [y.rstrip('\n') for y in x[1:]] for x in zip(source, *alternatives)}
    with open(args.out, 'w+') as f:
        yaml.dump(dictionary, f)
