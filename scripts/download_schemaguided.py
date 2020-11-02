#!/bin/env python
import shutil
import argparse
import sys
import os
from tqdm import tqdm
import requests
import logging
import json
from collections import Counter, defaultdict


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
BOOLEAN_SLOTS = ['has_wifi', 'pets_welcome', 'has_laundry_service', 'smoking_allowed', 'additional_luggage',
                 'has_live_music', 'has_seating_outdoors', 'has_vegetarian_options', 'subtitles', 'is_unisex',
                 'offers_cosmetic_services', 'private_visibility', 'add_insurance', 'has_garage', 'in_unit_laundry',
                 'pets_allowed', 'furnish']


def download_raw_data(root):
    os.makedirs(os.path.join(root, 'tmp'), exist_ok=True)
    _BASE_URL = "https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/"
    file_specs = {"dev": 20, "test": 34, "train": 122}
    with tqdm(total=sum(file_specs.values()), desc='downloading') as progress:
        for s, c in file_specs.items():
            for i in range(c):
                file_name = os.path.join(s, f"dialogues_{str(i + 1).zfill(3)}.json")
                dest = os.path.join(root, 'tmp', file_name)
                if not os.path.exists(dest):
                    download_file(_BASE_URL + file_name, dest)
                progress.update()


def load_raw_data(path, split):
    path = os.path.join(path, 'tmp', split)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.json')]
    data = []
    for f in files:
        data.extend(json.load(open(os.path.join(path, f))))
    return data


def delexicalise(ontology, turn):
    substitutions = []
    for frame in turn['frames']:
        if 'slots' not in frame:
            continue
        for slot in frame['slots']:
            start = slot["start"]
            end = slot["exclusive_end"]
            if start == end:
                continue
            key = ontology.map_slot(slot["slot"], frame['service'])
            substitutions.append((start, end, key))

    substitutions.sort(key=lambda x: x[0], reverse=True)
    text = turn['utterance']
    for s, e, k in substitutions:
        text = f'{text[:s]}[{k}]{text[e:]}'
    return text


def try_format_boolean(slot, value):
    if slot in BOOLEAN_SLOTS:
        return 'yes' if value == 'True' else 'no'
    return value


def get_dialogue_act(ontology, turn):
    act = []
    for frame in turn['frames']:
        domain = ontology.map_domain(frame['service'])
        for a in frame['actions']:
            k = a['slot']
            v = a['values']
            v = v[0] if type(v) is list and len(v) == 1 else v
            v = try_format_boolean(k, v)
            if isinstance(v, list):
                v = ','.join(v)
            act.append((map_intent(a['act']), domain, ontology.map_slot(k, domain), v))
    return act


def build_belief_state_updater(ontology):
    old_belief = defaultdict(dict)
    active_domain = None

    def get_belief_state(turn):
        nonlocal old_belief
        nonlocal active_domain

        assignment = defaultdict(set)
        for frame in turn['frames']:
            if 'state' in frame:
                domain = ontology.map_domain(frame['service'])
                for k, v in frame['state']['slot_values'].items():
                    v = v[0] if type(v) is list and len(v) == 1 else v
                    v = try_format_boolean(k, v)
                    if isinstance(v, list):
                        v = ','.join(v)
                    assignment[(domain, ontology.map_slot(k, frame['service']))].add(v)

        if assignment:
            active_domain = Counter(sum(([k[0]] * len(v) for k, v in assignment.items()), [])).most_common(1)[0][0]

        # Keep shortest texts in the assignment
        assignment = {k: min(v, key=lambda x: len(x)) for k, v in assignment.items()}
        new_belief = defaultdict(dict)
        for (domain, slot), value in assignment.items():
            new_belief[domain][slot] = value

        _old_belief = old_belief
        old_belief = new_belief
        return active_domain, _old_belief
    return get_belief_state


def map_intent(intent):
    intent = intent.lower()
    if intent == 'thank_you':
        return 'thank'
    return intent


def get_database(ontology, turn):
    # Database
    db = []
    for frame in turn['frames']:
        domain = frame['service']
        if ontology is not None:
            domain = ontology.map_domain(domain)
        if 'service_results' in frame:
            db.append((domain, len(frame['service_results'])))
        else:
            db.append((domain, 0))
    return dict(db)


def transform_schemaguided(raw_data, split, strip_greetings=False):
    ontology = get_ontology_unifier('schemaguided', None)
    dataset = dict(name='schemaguided', split=split, dialogues=[])
    dataset_domains = set()
    for record in tqdm(raw_data, desc=f'processing {split}'):
        domains = set(map(ontology.map_domain, record['services']))
        dataset_domains.update(domains)
        dialogue = dict(name=record['dialogue_id'], domains=list(domains), items=[])
        dataset['dialogues'].append(dialogue)
        get_belief_state = build_belief_state_updater(ontology)

        for turn in record['turns']:
            text = turn['utterance']

            current_item = dict(speaker=turn['speaker'].lower())
            dialogue['items'].append(current_item)
            current_item['dialogue_act'] = get_dialogue_act(ontology, turn)
            current_item['text'] = text
            current_item['delexicalised_text'] = delexicalise(ontology, turn)

            active_domain, belief = get_belief_state(turn)
            current_item['active_domain'] = active_domain
            current_item['belief'] = sort_belief(belief, active_domain)

            if current_item['speaker'] == 'system':
                # System has database
                database = get_database(ontology, turn)
                current_item['database'] = sort_database(belief, database)

    dataset['domains'] = list(domains)
    return dataset


def download(logger, args):
    path = os.path.join(DATASETS_PATH, 'schemaguided')
    os.makedirs(path, exist_ok=True)
    logger.info('Downloading original schemaguided data.')
    download_raw_data(path)
    logger.info('Starting data processing.')
    for split in ['train', 'test', 'dev']:
        raw_data = load_raw_data(path, split)
        dataset = transform_schemaguided(raw_data, split)
        dataset['split'] = split
        split_len = len(dataset["dialogues"])
        logger.info(f'Saving {split} partition with {split_len} dialogues.')
        with open(os.path.join(path, f'{split}.json'), 'w+') as f:
            json.dump(dataset, f)

    logger.info('Cleaning temporary directory.')
    shutil.rmtree(os.path.join(path, 'tmp'))
    logger.info('Dataset downloaded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    setup_logging()
    download(logging.getLogger(), parser.parse_args())
