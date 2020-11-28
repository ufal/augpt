#!/bin/env python
import shutil
import copy
import argparse
import random
import math
import re
import os
from tqdm import tqdm
import requests
import logging
import json
from collections import Counter, defaultdict, OrderedDict


os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.utils import sort_belief, sort_database  # noqa: E402
from ontology_unifier import get_ontology_unifier  # noqa: E402
from utils import setup_logging  # noqa: E402


def download_file(source_url, dest_path):
    response = requests.get(source_url, stream=True, timeout=5)
    response.raise_for_status()
    file_size = int(response.headers.get('content-length', 0))
    if "/" in dest_path:
        dir = "/".join(dest_path.split("/")[0:-1])
        os.makedirs(dir, exist_ok=True)
    pbar = tqdm(
        total=file_size, unit='B', disable=file_size < 1024**2,
        unit_scale=True, desc=source_url.split('/')[-1])

    with open(f'{dest_path}.tmp', "wb") as file:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            pbar.update(1024)
    pbar.close()
    shutil.move(f'{dest_path}.tmp', dest_path)


DATASETS_PATH = os.path.join(os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets')), 'augpt')
SOURCES = [
    'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/self-dialogs.json',
    'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/woz-dialogs.json',
    'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-2-2020/data/flights.json',
    'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-2-2020/data/food-ordering.json',
    'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-2-2020/data/hotels.json',
    'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-2-2020/data/movies.json',
    'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-2-2020/data/music.json',
    'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-2-2020/data/restaurant-search.json',  # noqa:E501
    'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-2-2020/data/sports.json'
]
DB_COUNT_REGEX = re.compile(r'have found (\d+) (hotels|restaurants)', re.IGNORECASE)


def download_raw_data(root):
    for url in SOURCES:
        name = url[url.rfind('/') + 1:]
        dest = os.path.join(root, name)
        if not os.path.exists(dest):
            download_file(url, dest)


FILE_DOMAIN_MAP = OrderedDict([
    ('flights', ['flight_search', 'flight1_detail', 'flight2_detail',
                 'flight3_detail', 'flight4_detail', 'flight_booked']),
    ('food-ordering', ['food_order']),
    ('hotels', ['hotel_search', 'hotel1_detail', 'hotel2_detail', 'hotel4_detail', 'hotel3_detail', 'hotel_booked']),
    ('movies', ['movie_search', 'movie_ticket']),
    ('music', ['music']),
    ('sports', ['epl', 'nba', 'mlb', 'nfl', 'mls']),
    ('restaurant-search', ['restaurant', 'restaurant_reservation'])
])


def load_raw_data(path):
    return {d: json.load(open(os.path.join(path, 'tmp', f'{d}.json'), 'r')) for d in FILE_DOMAIN_MAP.keys()}


def has_overlap(s1, e1, s2, e2):
    if s1 <= s2 <= e1 or s1 <= e2 <= e1:
        return True
    if s2 <= s1 <= e2 or s2 <= e1 <= e1:
        return True
    return False


def delexicalise(ontology, item):
    # First, we greadily remove overlapping segments
    # Shortest segments first, but greedy
    segments = []
    original_segments = sorted(item.get('segments', []), key=lambda x: x['end_index'] - x['start_index'])
    for segment in original_segments:
        for e in segments:
            if has_overlap(e['start_index'], e['end_index'], segment['start_index'], segment['end_index']):
                # Has overlap
                break
        else:
            segments.append(segment)

    segments.sort(key=lambda x: x['start_index'], reverse=True)
    dtext = item['text']
    pointer = -1
    for segment in segments:
        assert pointer == -1 or pointer > segment['end_index']
        domain, slot = segment['annotations'][0]['name'].split('.')[:2]
        slot = ontology.map_slot(slot, domain)
        dtext = dtext[:segment['start_index']] + f'[{slot}]' + dtext[segment['end_index']:]
        pointer = segment['start_index']
    return dtext


def get_database(active_domain, delexicalised_text):
    match = DB_COUNT_REGEX.search(delexicalised_text)
    if any([phrase in delexicalised_text for phrase in
            ['two hotels',
             'two locations',
             'Which one',
             '[name] and [name]']]):
        return {active_domain: 2}
    elif any([phrase in delexicalised_text for phrase in
              ['few options',
               'few places',
               'few hotels',
               'few locations'
               ]]):
        return {active_domain: 3}
    elif 'over a dozen' in delexicalised_text:
        return {active_domain: 12}
    elif match is not None:
        return {active_domain: int(match.group(1))}
    elif any([phrase in delexicalised_text for phrase in
              ['have found a',
               'found [name]',
               'found the [name]'
               'one location']]):
        return {active_domain: 1}
    else:
        return {active_domain: 0}


def build_belief_state_updater(ontology):
    user_belief_state = defaultdict(dict)
    system_belief_state = defaultdict(dict)
    active_domain = None

    def get_belief_state(item):
        nonlocal user_belief_state
        nonlocal system_belief_state
        nonlocal active_domain

        assignment = defaultdict(set)
        for segment in item.get('segments', []):
            domain, slot = segment['annotations'][0]['name'].split('.')[:2]
            slot = ontology.map_slot(slot, domain)
            domain = ontology.map_domain(domain)
            assignment[(domain, slot)].add(segment['text'])

        if assignment:
            active_domain = Counter(sum(([k[0]] * len(v) for k, v in assignment.items()), [])).most_common(1)[0][0]

        # Keep shortest texts in the assignment
        assignment = {k: min(v, key=lambda x: len(x)) for k, v in assignment.items()}

        # User updates system belief_state and vice versa
        if item['speaker'] == 'USER':
            system_belief_state = copy.deepcopy(system_belief_state)
            for (domain, slot), value in assignment.items():
                system_belief_state[domain][slot] = value
            return active_domain, user_belief_state
        else:
            user_belief_state = copy.deepcopy(user_belief_state)
            for (domain, slot), value in assignment.items():
                user_belief_state[domain][slot] = value
            return active_domain, system_belief_state
    return get_belief_state


def transform_taskmaster(raw_data, strip_greetings=False):
    ontology = get_ontology_unifier('taskmaster', None)
    dataset = dict(name='taskmaster', dialogues=[])
    domains = set()
    total_length = sum(len(v) for v in raw_data.values())
    with tqdm(desc='processing', total=total_length) as progress:
        for domain, data in raw_data.items():
            domain = FILE_DOMAIN_MAP[domain][0]
            if ontology is not None:
                domain = ontology.map_domain(domain)
            domains.add(domain)

            for record in data:
                dialogue = dict(name=record['conversation_id'], domains=[domain], items=[])
                get_belief_state = build_belief_state_updater(ontology)
                dataset['dialogues'].append(dialogue)
                current_item = None
                for item in record['utterances']:
                    speaker = 'user' if item['speaker'] == 'USER' else 'system'
                    if current_item is None and strip_greetings:
                        if speaker == 'system':
                            continue  # strip greetings
                    if current_item is None or speaker != current_item['speaker']:
                        current_item = dict(speaker=speaker)
                        dialogue['items'].append(current_item)
                    delexicalised_text = delexicalise(ontology, item)
                    current_item['text'] = (current_item.get('text', '') + ' ' + item['text']).lstrip(' ')
                    current_item['delexicalised_text'] = \
                        (current_item.get('delexicalised_text', '') + ' ' + delexicalised_text).lstrip(' ')
                    active_domain, belief = get_belief_state(item)
                    current_item['active_domain'] = active_domain
                    current_item['belief'] = sort_belief(belief, active_domain)
                    if speaker == 'system':
                        # System has database
                        if active_domain is not None:
                            database = get_database(active_domain, current_item['delexicalised_text'])
                        else:
                            database = OrderedDict()
                        current_item['database'] = sort_database(belief, database)
                progress.update()

    dataset['domains'] = list(domains)
    return dataset


def split_dataset(data, splits=None):
    if splits is None:
        splits = (('train', 0.9), ('dev', 0.1))

    dialogues = data.pop('dialogues')
    indexes = list(range(len(dialogues)))
    random.Random(42).shuffle(indexes)
    offset = 0
    for i, (name, size) in enumerate(splits):
        size = min(len(dialogues) - offset, math.ceil(size * len(dialogues)))
        if i == len(splits) - 1:
            size = len(dialogues) - offset
        idx = sorted(indexes[offset:offset + size])
        dials = [dialogues[x] for x in idx]
        offset += size
        yield name, dict(dialogues=dials, split=name, **data)


def download(logger, args):
    path = os.path.join(DATASETS_PATH, 'taskmaster')
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'tmp'), exist_ok=True)
    logger.info('Downloading original taskmaster data.')
    download_raw_data(os.path.join(path, 'tmp'))
    raw_data = load_raw_data(os.path.join(path))
    logger.info('Starting data processing.')
    dataset = transform_taskmaster(raw_data, strip_greetings=args.strip_greetings)
    total_length = len(dataset['dialogues'])
    for (split, sdataset) in split_dataset(dataset):
        split_len = len(sdataset["dialogues"])
        logger.info(f'Saving {split} with {split_len} dialogues ({100 * split_len / total_length:.2f}%).')
        with open(os.path.join(path, f'{split}.json'), 'w+') as f:
            json.dump(sdataset, f)

    logger.info('Cleaning temporary directory.')
    shutil.rmtree(os.path.join(path, 'tmp'))
    logger.info('Dataset downloaded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-strip-greetings', dest='strip_greetings', action='store_false')
    setup_logging()
    download(logging.getLogger(), parser.parse_args())
