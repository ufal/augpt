#!/usr/bin/env python
import sqlite3
import json
import os
import tempfile
import re
import shutil
import requests
import random
import logging
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
import zipfile
import inspect
from collections import OrderedDict, Counter
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging  # noqa: E402


np.set_printoptions(precision=3)
np.random.seed(2)
setup_logging()
logger = logging.getLogger()

# GLOBAL VARIABLES
DATASETS_PATH = os.path.join(os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets')), 'augpt')
DICT_SIZE = 400
MAX_LENGTH = 50
DEFAULT_IGNORE_VALUES = ['not mentioned', 'dont care', 'don\'t care', 'dontcare', 'do n\'t care', 'none']
MW_DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
digitpat = re.compile(r'\d+')
timepat = re.compile(r"\d{1,2}[:]\d{1,2}")
pricepat2 = re.compile(r"\d{1,3}[.]\d{1,2}")
timepat = re.compile(r"\d{1,2}[:]\d{1,2}")
label_regex = re.compile(r'\[([\w\d\s]+)\]')
pricepat = re.compile(r"\d{1,3}[.]\d{1,2}")
fin = open(os.path.join(os.path.dirname(__file__), 'mapping.pair'), 'r')
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))


class Lexicalizer:
    def __init__(self, zipf):
        self.path = zipf.filename

    placeholder_re = re.compile(r'\[(\s*[\w_\s]+)\s*\]')
    number_re = re.compile(r'.*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s$')
    time_re = re.compile(r'((?:\d{1,2}[:]\d{2,3})|(?:\d{1,2} (?:am|pm)))', re.IGNORECASE)

    @staticmethod
    def ends_with_number(s):
        return bool(Lexicalizer.number_re.match(s))

    @staticmethod
    def extend_database_results(database_results, belief):
        # Augment database results from the belief state
        database_results = OrderedDict(database_results)
        if belief is not None:
            for i, (domain, (num_results, results)) in enumerate(database_results.items()):
                if domain not in belief:
                    continue
                if num_results == 0:
                    database_results[domain] = (1, [belief[domain]])
                else:
                    new_results = []
                    for r in results:
                        r = dict(**r)
                        for k, val in belief[domain].items():
                            if k not in r:
                                r[k] = val
                        new_results.append(r)
                    database_results[domain] = (num_results, new_results)
        return database_results

    def __call__(self, text, database_results, belief=None, context=None):
        database_results = Lexicalizer.extend_database_results(database_results, belief)
        result_index = 0
        last_assignment = defaultdict(set)

        def trans(label, span, force=False, loop=100):
            nonlocal result_index
            nonlocal last_assignment
            result_str = None
            current_domain = None
            if '_' in label:
                current_domain = label[:label.index('_')]
                label = label[label.index('_') + 1:]
            if label == 'postcode':
                label = 'post code'

            # No references in the MW 2.0 database
            if label == 'reference':
                return 'YF86GE4J'

            for domain, (count, results) in database_results.items():
                if count == 0:
                    continue
                if current_domain is not None and domain != current_domain and not force:
                    continue
                result = results[result_index % len(results)]
                if label in result:
                    result_str = str(result[label])
                    if result_str == '?':
                        result_str = 'unknown'
                    if label == 'price range' and result_str == 'moderate' and \
                            not text[span[1]:].startswith(' price range') and \
                            not text[span[1]:].startswith(' in price'):
                        result_str = 'moderately priced'
                    if label == 'type':
                        if text[:span[0]].endswith('no ') or text[:span[0]].endswith('any ') or \
                                text[:span[0]].endswith('some ') or Lexicalizer.ends_with_number(text[:span[0]]):
                            if not result_str.endswith('s'):
                                result_str += 's'
                if label == 'time' and ('[leave at]' in text or '[arrive by]' in text) and \
                    belief is not None and 'train' in belief and \
                        any([k in belief['train'] for k in ('leave at', 'arrive by')]):
                    # this is a specific case in which additional [time] slot needs to be lexicalised
                    # directly from the belief state
                    # "The earliest train after [time] leaves at ... and arrives by ..."
                    if 'leave at' in belief['train']:
                        result_str = belief['train']['leave at']
                    else:
                        result_str = belief['train']['arrive by']
                elif force:
                    if label == 'time':
                        if 'leave at' in result or 'arrive by' in result:
                            if 'arrive' in text and 'arrive by' in result:
                                result_str = result['arrive by'].lstrip('0')
                            elif 'leave at' in result:
                                result_str = result['leave at'].lstrip('0')
                        elif context is not None and len(context) > 0:
                            last_utt = context[-1]
                            mtch = Lexicalizer.time_re.search(last_utt)
                            if mtch is not None:
                                result_str = mtch.group(1).lstrip('0')
                if result_str is not None:
                    break
            if force and result_str is None:
                if label == 'reference':
                    result_str = 'YF86GE4J'
                elif label == 'phone':
                    result_str = '01223358966'
                elif label == 'postcode':
                    result_str = 'CB11JG'
                elif label == 'agent':
                    result_str = 'Cambridge Towninfo Centre'
                elif label == 'stars':
                    result_str = '4'

            if result_str is not None and result_str.lower() in last_assignment[label] and loop > 0:
                result_index += 1
                return trans(label, force=force, loop=loop - 1, span=span)

            if result_str is not None:
                last_assignment[label].add(result_str.lower())
            return result_str or f'[{label}]'

        text = Lexicalizer.placeholder_re.sub(lambda m: trans(m.group(1), span=m.span()), text)
        text = Lexicalizer.placeholder_re.sub(lambda m: trans(m.group(1), force=True, span=m.span()), text)
        return text

    def save(self, path):
        shutil.copy(self.path, os.path.join(path, os.path.split(self.path)[-1]))


def clear_whitespaces(text):
    text = re.sub(r'[\s\n\r]+', ' ', text)
    text = ' ' + text + ' '
    text = re.sub(r'\s([,\.:\?\!\']+)', lambda m: m.group(1), text)
    return text.strip()


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall(r'\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall(
        r'([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
        text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub(r'[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    text = re.sub(timepat, ' [value_time] ', text)
    text = re.sub(pricepat, ' [value_price] ', text)
    # text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub(r'$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub(r'[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub(r'^\'', '', text)
    text = re.sub(r'\'$', '', text)
    text = re.sub(r'\'\s', ' ', text)
    text = re.sub(r'\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(r'^\d+$', tokens[i]) and \
                re.match(r'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)
    return text


def fix_active_domain_and_delex(active_domain, text, delex):
    domains = [x.group(1).split('_')[0] for x in label_regex.finditer(delex)]
    domains = [x for x in MW_DOMAINS if x in domains]
    domain_counter = Counter(domains)
    if domain_counter:
        active_domain = domain_counter.most_common(1)[0][0]

    lresponse = text.lower()
    if 'hotel' in lresponse:
        active_domain = 'hotel'
    if 'train' in lresponse or 'arrive' in lresponse or 'leave' in lresponse:
        active_domain = 'train'
    if 'attraction' in lresponse:
        active_domain = 'attraction'
    if 'police' in lresponse:
        active_domain = 'police'
    if 'restaurant' in lresponse or 'food' in lresponse:
        active_domain = 'restaurant'
    if 'hospital' in lresponse:
        active_domain = 'hospital'
    if 'taxi' in lresponse or 'car' in lresponse:
        active_domain = 'taxi'
    taxi_brands = ["toyota", "skoda", "bmw", 'honda', 'ford', 'audi', 'lexus', 'volvo', 'volkswagen', 'tesla']
    if any(t in lresponse for t in taxi_brands):
        active_domain = 'taxi'

    for match in label_regex.finditer(delex):
        domain, slot = match.group(1).split('_')
        if slot == 'reference':
            active_domain = domain

    if active_domain is not None:
        delex = label_regex.sub(lambda x: f'[{active_domain}_{x.group(1).split("_")[1]}]', delex)
    return active_domain, delex


def prepareSlotValuesIndependent(dbzipf, path):
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
    dic = []
    dic_area = []
    dic_food = []
    dic_price = []

    # read databases
    for domain in domains:
        try:
            fin = dbzipf.open(os.path.join('db/' + domain + '_db.json'), 'r')
            db_json = json.load(fin)
            fin.close()

            for ent in db_json:
                for key, val in ent.items():
                    if val == '?' or val == 'free':
                        pass
                    elif key == 'address':
                        dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        if "road" in val:
                            val = val.replace("road", "rd")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "rd" in val:
                            val = val.replace("rd", "road")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "st" in val:
                            val = val.replace("st", "street")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "street" in val:
                            val = val.replace("street", "st")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif key == 'name':
                        dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        if "b & b" in val:
                            val = val.replace("b & b", "bed and breakfast")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "bed and breakfast" in val:
                            val = val.replace("bed and breakfast", "b & b")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "hotel" in val and 'gonville' not in val:
                            val = val.replace("hotel", "")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "restaurant" in val:
                            val = val.replace("restaurant", "")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif key == 'postcode':
                        dic.append((normalize(val), '[' + domain + '_' + 'postcode' + ']'))
                    elif key == 'phone':
                        dic.append((val, '[' + domain + '_' + 'phone' + ']'))
                    elif key == 'trainID':
                        dic.append((normalize(val), '[' + domain + '_' + 'id' + ']'))
                    elif key == 'department':
                        dic.append((normalize(val), '[' + domain + '_' + 'department' + ']'))

                    # NORMAL DELEX
                    elif key == 'area':
                        dic_area.append((normalize(val), '[' + 'value' + '_' + 'area' + ']'))
                    elif key == 'food':
                        dic_food.append((normalize(val), '[' + 'value' + '_' + 'food' + ']'))
                    elif key == 'pricerange':
                        dic_price.append((normalize(val), '[' + 'value' + '_' + 'pricerange' + ']'))
                    else:
                        pass
                    # TODO car type?
        except(Exception):
            pass

        if domain == 'hospital':
            dic.append((normalize('Hills Rd'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('Hills Road'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('CB20QQ'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('0122324515', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize('Addenbrookes Hospital'), '[' + domain + '_' + 'name' + ']'))

        elif domain == 'police':
            dic.append((normalize('Parkside'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('CB11JG'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize('Parkside Police Station'), '[' + domain + '_' + 'name' + ']'))

    # add at the end places from trains
    fin = dbzipf.open(os.path.join('db/' + 'train' + '_db.json'), 'r')
    db_json = json.load(fin)
    fin.close()

    for ent in db_json:
        for key, val in ent.items():
            if key == 'departure' or key == 'destination':
                dic.append((normalize(val), '[' + 'value' + '_' + 'place' + ']'))

    # add specific values:
    for key in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
        dic.append((normalize(key), '[' + 'value' + '_' + 'day' + ']'))

    # more general values add at the end
    dic.extend(dic_area)
    dic.extend(dic_food)
    dic.extend(dic_price)

    return dic


def delexicalise(utt, dictionary):
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?

    return utt


def delexicaliseDomain(utt, dictionary, domain):
    for key, val in dictionary:
        if key == domain or key == 'value':
            utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
            utt = utt[1:-1]  # why this?

    # go through rest of domain in case we are missing something out?
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?
    return utt


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def domain_not_empty(domain_bs):
    return any(len(val) > 0 and val not in DEFAULT_IGNORE_VALUES for val in domain_bs.values())


class BeliefStateTransformation:
    def _process_domain(self, domain_bs):
        return {self._map_slot(slot): self._clear_value(val) for slot, val in domain_bs.items()
                if (len(val) > 0 and val not in DEFAULT_IGNORE_VALUES)}

    def _map_slot(self, slot):
        if slot == 'arriveBy':
            return 'arrive by'
        if slot == 'leaveAt':
            return 'leave at'
        if slot == 'pricerange':
            slot = 'price range'
        return slot

    def _clear_value(self, value):
        value = value.replace('>', ' ')
        if value == 'el shaddia guesthouse':
            value = 'el shaddai'
        if value == 'concerthall':
            value = 'concert hall'
        if value == 'nightclub':
            value = 'night club'
        # BUG in MW2.0
        value = value.lstrip('`')
        return value

    def __call__(self, belief_state, dialogue_act, active_domain):
        clean_belief = dict()
        for domain, domain_bs in belief_state.items():
            new_domain_bs = {}
            if 'semi' in domain_bs:
                new_domain_bs.update(domain_bs['semi'])
            if 'book' in domain_bs:
                new_domain_bs.update({k: v for k, v in domain_bs['book'].items() if k != 'booked'})
            if 'book' in domain_bs and 'booked' in domain_bs['book'] and len(domain_bs['book']['booked']) > 0:
                new_domain_bs['booked'] = 'true'
            elif not domain_not_empty(domain_bs):
                continue
            new_domain_bs = self._process_domain(new_domain_bs)
            if len(new_domain_bs) == 0:
                continue
            if 'internet' in new_domain_bs and new_domain_bs['internet'] == 'no':
                del new_domain_bs['internet']  # no internet by default
            if 'parking' in new_domain_bs and new_domain_bs['parking'] == 'no':
                del new_domain_bs['parking']  # no parking by default
            clean_belief[domain] = new_domain_bs

        for domain in {'Hospital', 'Police'}:
            if any([da[1] == domain for da in dialogue_act]):
                clean_belief[domain.lower()] = {}

        # Sort belief
        clean_belief = {k: OrderedDict(sorted(v.items(), key=lambda x: x[0])) for k, v in clean_belief.items()}
        active_bs = None
        if active_domain is not None:
            active_domain = active_domain.lower()
            active_bs = clean_belief.pop(active_domain, None)
        items = [(active_domain, active_bs)] if active_bs is not None else []
        items += [(k, v) for k, v in sorted(clean_belief.items(), key=lambda x: x[0])]
        result = OrderedDict(items)
        return result


def fixDelex(delex, act):
    for k in act:
        if 'Attraction' == k[1]:
            if 'restaurant_' in delex:
                delex = delex.replace("restaurant", "attraction")
            if 'hotel_' in delex:
                delex = delex.replace("hotel", "attraction")
        if 'Hotel' == k[1]:
            if 'attraction_' in delex:
                delex = delex.replace("attraction", "hotel")
            if 'restaurant_' in delex:
                delex = delex.replace("restaurant", "hotel")
        if 'Restaurant' == k[1]:
            if 'attraction_' in delex:
                delex = delex.replace("attraction", "restaurant")
            if 'hotel_' in delex:
                delex = delex.replace("hotel", "restaurant")

    return delex


def delexicaliseReferenceNumber(sent, turn):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    if turn['metadata']:
        for domain in domains:
            if turn['metadata'][domain]['book']['booked']:
                for slot in turn['metadata'][domain]['book']['booked'][0]:
                    if slot == 'reference':
                        val = '[' + domain + '_' + slot + ']'
                    else:
                        val = '[' + domain + '_' + slot + ']'
                    key = normalize(turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with hashtag
                    key = normalize("#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with ref#
                    key = normalize("ref#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
    return sent


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d['log']) % 2 != 0:
        # print path
        logger.warning('odd # of turns')
        return None  # odd number of turns, wrong dialogue

    for i in range(len(d['log'])):
        if len(d['log'][i]['text'].split()) > maxlen:
            logger.warning('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                logger.warning('not ascii')
                return None
        else:  # sys turn
            if 'database' not in d['log'][i]:
                logger.warning('no db')
                return None  # no db_pointer, probably 2 usr turns in a row, wrong dialogue
            text = d['log'][i]['text']
            if not is_ascii(text):
                logger.warning('not ascii')
                return None
        d['log'][i]['text'] = clear_whitespaces(d['log'][i]['text'])
    return dialogue


def get_dial(dialogue):
    d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    return d_orig


def createDict(word_freqs):
    words = list(word_freqs.keys())
    freqs = list(word_freqs.values())

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    # Extra vocabulary symbols
    _GO = '_GO'
    EOS = '_EOS'
    UNK = '_UNK'
    PAD = '_PAD'
    extra_tokens = [_GO, EOS, UNK, PAD]

    worddict = OrderedDict()
    for ii, ww in enumerate(extra_tokens):
        worddict[ww] = ii
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + len(extra_tokens)

    for key, idx in list(worddict.items()):
        if idx >= DICT_SIZE:
            del worddict[key]

    return worddict


def createDelexData(zipf, path):
    """Main function of the script - loads delexical dictionary,
    goes through each dialogue and does:
    1) data normalization
    2) delexicalization
    3) addition of database pointer
    4) saves the delexicalised data
    """
    transform_belief = BeliefStateTransformation()

    # Load databases
    with zipfile.ZipFile(os.path.join(path, 'database.zip')) as dbzipf:
        db = Database(dbzipf)

        # create dictionary of delexicalied values that then we will search against, order matters here!
        dic = prepareSlotValuesIndependent(dbzipf, path)
    delex_data = OrderedDict()
    with zipfile.ZipFile(os.path.join(path, 'lexicalizer.zip')) as lexzipf:
        lexicalizer = Lexicalizer(lexzipf)

    root = next(iter({n.strip('data.json') for n in zipf.namelist() if n.endswith('data.json')}))
    fin1 = zipf.open(root + 'data.json', 'r')
    data = json.load(fin1)

    fin2 = zipf.open(root + 'dialogue_acts.json', 'r')
    data2 = json.load(fin2)
    ignored_dialogues = 0

    for dialogue_name in tqdm(data):
        dialogue = data[dialogue_name]
        # print dialogue_name

        idx_acts = 1
        active_domain = None
        ignore_dialogue = False

        for idx, turn in enumerate(dialogue['log']):
            try:
                dialogue_act = [tuple(reversed(f.split('-'))) + tuple(x)
                                for f, xs in data2[dialogue_name.strip('.json')][str(idx_acts)].items() for x in xs]
            except(Exception):
                dialogue_act = []
            # normalization, split and delexicalization of the sentence
            sent = normalize(turn['text'])
            text = sent

            words = sent.split()
            sent = delexicalise(' '.join(words), dic)

            # parsing reference number GIVEN belief state
            sent = delexicaliseReferenceNumber(sent, turn)

            # changes to numbers only here
            digitpat = re.compile(r'\d+')
            sent = re.sub(digitpat, '[value_count]', sent)

            dialogue['log'][idx]['dialogue_act'] = dialogue_act
            dialogue['log'][idx]['speaker'] = 'user'

            # delexicalised sentence added to the dialogue
            delex = sent.strip()
            delex = fixDelex(delex, dialogue_act)

            if idx % 2 == 1:  # if it's a system turn
                dialogue['log'][idx]['speaker'] = 'system'
                belief = dialogue['log'][idx]['metadata']
                active_domain, delex = fix_active_domain_and_delex(active_domain, text, delex)
                dialogue['log'][idx]['active_domain'] = active_domain

                belief = transform_belief(belief, dialogue_act, active_domain)
                dialogue['log'][idx]['belief'] = belief
                if 'bus' in belief:
                    # We need to ignore this dialogue
                    # There is no data for the bus domain
                    ignore_dialogue = True
                    break

                dialogue['log'][idx]['database'] = db(belief)

                # Add booked property
                dialogue['log'][idx]['booked_domains'] = sorted(get_booked_domains(dialogue['log'][idx]['metadata']))

                # Test if lexicalizer works
                lexicalizer(delex, db(belief, return_results=True), belief)

            dialogue['log'][idx]['delexicalised_text'] = delex

            idx_acts += 1

        if not ignore_dialogue:
            dialogue['goal'] = parse_goal(dialogue['goal'])
            delex_data[dialogue_name] = dialogue
        else:
            ignored_dialogues += 1
    if ignored_dialogues > 0:
        logger.warning(f'dialogues were ignored {100 * ignored_dialogues / (ignored_dialogues + len(delex_data)):.1f}% due to a missing domain "bus"')  # noqa: E501
    return delex_data


def load_databases(zipf):
    dbs = {}
    sql_dbs = {'attraction', 'hotel', 'restaurant', 'train'}
    for domain in MW_DOMAINS:
        if domain in sql_dbs:
            db = 'db/{}-dbase.db'.format(domain)
            with tempfile.NamedTemporaryFile('rb+') as dbf:
                shutil.copyfileobj(zipf.open(db), dbf)
                dbf.flush()
                fileconn = sqlite3.connect(dbf.name)
                conn = sqlite3.connect(':memory:')
                fileconn.backup(conn)

            def dict_factory(cursor, row):
                d = {}
                for idx, col in enumerate(cursor.description):
                    d[col[0]] = row[idx]
                return d

            conn.row_factory = dict_factory
            c = conn.cursor()
            dbs[domain] = c
        else:
            db = 'db/{}_db.json'.format(domain)
            dbs[domain] = json.load(zipf.open(db))
    return dbs


class Database:
    def __init__(self, zipf, seed=42):
        self.path = zipf.filename
        self.dbs = load_databases(zipf)
        self.ignore_values = ['not mentioned', 'dont care', 'don\'t care', 'dontcare', 'do n\'t care', 'none']
        self.rng = random.Random(seed)

    price_re = re.compile(r'\d+\.\d+')

    @staticmethod
    def translate_to_db_col(s):
        if s == 'leave at':
            return 'leaveAt'
        elif s == 'arrive by':
            return 'arriveBy'
        elif s == 'price range':
            return 'pricerange'
        else:
            return s

    def domain_not_empty(self, domain_bs):
        return any(len(val) > 0 and val not in self.ignore_values for val in domain_bs.values())

    @staticmethod
    def map_database_key(key):
        if key == 'trainID':
            key = 'id'
        key = ''.join([' '+i.lower() if i.isupper()
                       else i for i in key]).lstrip(' ')
        key = key.replace('_', ' ')
        if key == 'pricerange':
            key = 'price range'
        if key == 'taxi phone' or key == 'phone':
            key = 'phone'
        if key == 'taxi colors':
            key = 'color'
        if key == 'taxi types':
            key = 'brand'
        if key == 'ref':
            key = 'reference'
        if key == 'leaveAt':
            key = 'leave at'
        if key == 'arriveBy':
            key = 'arrive by'
        if key == 'entrance fee':
            key = 'fee'
        return key

    @staticmethod
    def map_query_value(value):
        if value == 'concert hall':
            value = 'concerthall'
        if value == 'night club':
            value = 'nightclub'
        return value

    @staticmethod
    def capitalize(val):
        def _mk(v):
            i, v = v
            if i == 0 or v not in {'the', 'an', 'a', 'of', 'in', 'for', 'as', 'these', 'at', 'up', 'on', 'and', 'or'}:
                return v[:1].upper() + v[1:]
            else:
                return v
        return ' '.join(map(_mk, enumerate(val.split())))

    @staticmethod
    def map_database_row(domain, row, query):
        results = dict()
        for k, val in row.items():
            k2 = Database.map_database_key(k)
            if k == 'location':
                continue
            elif k == 'post code' or k == 'postcode':
                val = val.upper()
            elif k == 'name':
                val = Database.capitalize(val)
            elif k == 'type' and val == 'concerthall':
                val = 'concert hall'
            elif k == 'price' and domain == 'hotel' and isinstance(val, dict):
                val = val.get('single', val.get('double', next(iter(val.values()))))
                val = f'{val} pounds'
            if k2 == 'people':
                # BUG in MW2.0
                val = val.lstrip('`')
            results[k2] = val
        if 'color' in results and 'brand' in results:
            results['car'] = f"{results['color']} {results['brand']}"
        if domain == 'train' and 'price' in row and 'people' in query:
            people = int(query['people'])

            def multiply_people(m):
                price = float(m.group(0))
                price *= people
                return format(price, '.2f')
            if people != 1:
                results['price'] = Database.price_re.sub(multiply_people, row['price'])
        return results

    def query_domain(self, domain, query):
        # Handle special domains not in sqlite databases
        # NOTE: this is not a part of multiwoz repo
        # Taken from convlab
        if domain == 'taxi':
            return [{'color': self.rng.choice(self.dbs[domain]['taxi_colors']),
                     'brand': self.rng.choice(self.dbs[domain]['taxi_types']),
                     'phone': ''.join([str(random.randint(1, 9)) for _ in range(11)])}]
        if domain == 'police':
            return deepcopy(self.dbs['police'])
        if domain == 'hospital':
            department = None
            for key, val in query:
                if key == 'department':
                    department = val
            if not department:
                return deepcopy(self.dbs['hospital'])
            else:
                return [deepcopy(x) for x in self.dbs['hospital']
                        if x['department'].lower() == department.strip().lower()]

        sql_query = "select * from {}".format(domain)

        flag = True
        for key, val in query:
            if val == "" or val in self.ignore_values:
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    # change query for trains
                    if key == 'leaveAt':
                        sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

        result = self.dbs[domain].execute(sql_query).fetchall()
        return result

    def __call__(self, belief, return_results=False):
        all_results = OrderedDict()
        for domain, domain_bs in belief.items():
            blocked_slots = {'people', 'booked', 'stay'}
            if domain != 'train' and domain != 'bus':
                blocked_slots.add('day')
                blocked_slots.add('time')
            query = [(Database.translate_to_db_col(slot), Database.map_query_value(val))
                     for slot, val in domain_bs.items() if slot not in blocked_slots]

            result = self.query_domain(domain, query)
            result = [Database.map_database_row(domain, k, domain_bs) for k in result]
            if return_results:
                all_results[domain] = (len(result), result)
            else:
                all_results[domain] = len(result)
        return all_results

    def save(self, path):
        shutil.copy(self.path, os.path.join(path, os.path.split(self.path)[-1]))


def is_booked(raw_belief, domain):
    return domain in raw_belief and 'book' in raw_belief[domain] and \
        'booked' in raw_belief[domain]['book'] and \
        any('reference' in x for x in raw_belief[domain]['book']['booked'])


def get_booked_domains(raw_belief):
    for domain in raw_belief.keys():
        if is_booked(raw_belief, domain):
            yield domain


def parse_goal(dialog_goal):
    belief_transformation = BeliefStateTransformation()
    """Parses user goal into dictionary format."""
    goal = {}
    for domain in MW_DOMAINS:
        if not dialog_goal[domain]:
            continue
        goal[domain] = {}
        goal[domain] = {'informable': [], 'requestable': [], 'booking': {}}
        if 'info' in dialog_goal[domain]:
            # if d['goal'][domain].has_key('info'):
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in dialog_goal[domain]:
                    # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in dialog_goal[domain]:
                    # if d['goal'][domain].has_key('reqt'):
                    if 'trainID' in dialog_goal[domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in dialog_goal[domain]:
                    # if d['goal'][domain].has_key('reqt'):
                    for s in dialog_goal[domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalised
                            goal[domain]['requestable'].append(s)
                if 'book' in dialog_goal[domain]:
                    # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append("reference")

            goal[domain]["informable"] = dialog_goal[domain]['info']
            if 'book' in dialog_goal[domain]:
                # if d['goal'][domain].has_key('book'):
                goal[domain]["booking"] = dialog_goal[domain]['book']

        if 'invalid' in goal[domain]['booking']:
            del goal[domain]['booking']['invalid']
        if 'pre_invalid' in goal[domain]['booking']:
            del goal[domain]['booking']['pre_invalid']
        belief = {domain: {'semi': goal[domain]['informable'], 'book': goal[domain]['booking']}}
        belief = belief_transformation(belief, [], domain).get(domain, dict())
        goal[domain]['informable'] = belief
        del goal[domain]['booking']
    return goal


def map_dialogue_items(log):
    supported_keys = {'text', 'delexicalised_text', 'speaker', 'belief', 'database',
                      'active_domain', 'dialogue_act', 'booked_domains'}
    for item in log:
        yield {k: v for k, v in item.items() if k in supported_keys}


def divideData(data, zipf, path):
    """Given test and validation sets, divide
    the data for three different sets"""
    testListFile = []
    root = next(iter({n.strip('data.json') for n in zipf.namelist() if n.endswith('data.json')}))
    fin = zipf.open(root + 'testListFile.json', 'r')
    for line in fin:
        testListFile.append(line[:-1].decode('utf-8'))
    fin.close()

    valListFile = []
    fin = zipf.open(root + 'valListFile.json', 'r')
    for line in fin:
        valListFile.append(line[:-1].decode('utf-8'))
    fin.close()

    test_dials = []
    val_dials = []
    train_dials = []

    for dialogue_name in tqdm(data):
        # print dialogue_name
        dial = get_dial(data[dialogue_name])
        if dial:
            dialogue = {}
            dialogue['name'] = dialogue_name
            dialogue['items'] = list(map_dialogue_items(dial['log']))
            dialogue['goal'] = dial['goal']

            if dialogue_name in testListFile:
                test_dials.append(dialogue)
            elif dialogue_name in valListFile:
                val_dials.append(dialogue)
            else:
                train_dials.append(dialogue)

    # save all dialogues
    with open(os.path.join(path, 'val.json'), 'w') as f:
        json.dump(dict(domains=MW_DOMAINS, dialogues=val_dials), f, indent=4)

    with open(os.path.join(path, 'test.json'), 'w') as f:
        json.dump(dict(domains=MW_DOMAINS, dialogues=test_dials), f, indent=4)

    with open(os.path.join(path, 'train.json'), 'w') as f:
        json.dump(dict(domains=MW_DOMAINS, dialogues=train_dials), f, indent=4)


def export_database_source(zipf):
    source_code = f"""import sqlite3
import os
import shutil
import re
import random
import json
import zipfile
import tempfile
from copy import deepcopy
from collections import OrderedDict


MW_DOMAINS = {MW_DOMAINS}


{inspect.getsource(load_databases)}

{inspect.getsource(Database)}"""
    with zipf.open('database.py', 'w') as f:
        f.write(source_code.encode('utf-8'))
        f.flush()


def download_file(source_url, dest):
    response = requests.get(source_url, stream=True, timeout=5)
    response.raise_for_status()
    file_size = int(response.headers.get('content-length', 0))
    zipf = None
    if isinstance(dest, tuple):
        zipf, dest_path = dest
    else:
        dest_path = dest
        if "/" in dest_path:
            dir = "/".join(dest_path.split("/")[0:-1])
            os.makedirs(dir, exist_ok=True)
        if os.path.exists(dest_path):
            return

    pbar = tqdm(
        total=file_size, unit='B', disable=file_size < 1024**2,
        unit_scale=True, desc=source_url.split('/')[-1])

    with tempfile.TemporaryFile('rb+') as file:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            pbar.update(1024)
        file.flush()
        file.seek(0)
        pbar.close()
        if zipf is not None:
            with zipf.open(dest_path, 'w') as f:
                shutil.copyfileobj(file, f)
        else:
            with open(dest_path, 'wb+') as f:
                shutil.copyfileobj(file, f)


def export_lexicalizer_source(path):
    source_code = f"""from collections import defaultdict, OrderedDict
import os
import shutil
import re


{inspect.getsource(Lexicalizer)}"""
    with zipfile.ZipFile(os.path.join(path, 'lexicalizer.zip'), 'w') as zipf:
        with zipf.open('lexicalizer.py', 'w') as f:
            f.write(source_code.encode('utf-8'))
            f.flush()


def extract_databases(path, dbzipf, multiwoz_sha):
    with zipfile.ZipFile(os.path.join(path, 'database.zip'), 'w') as dboutf:
        for domain in MW_DOMAINS[:-1]:
            db = f'multiwoz-{multiwoz_sha}/db/{domain}-dbase.db'
            with dbzipf.open(db) as zf, dboutf.open(os.path.join('db', f'{domain}-dbase.db'), 'w') as f:
                shutil.copyfileobj(zf, f)

        # Fix json databases
        # Download from convlab2
        for domain in MW_DOMAINS:
            download_file(
                    f'https://raw.githubusercontent.com/thu-coai/ConvLab-2/b82732eae951b3dc957136f40b992a1904c9cbe5/data/multiwoz/db/{domain}_db.json',  # noqa: E501
                    (dboutf, os.path.join('db', f'{domain}_db.json')))

        # Export database source
        export_database_source(dboutf)


def download(version='2.0'):
    path = os.path.join(DATASETS_PATH, f'multiwoz-{version}')
    multiwoz_sha = 'a24d299fafa00371d03880bce34cb3b0923518fa'
    os.makedirs(path, exist_ok=True)
    download_file(
        f'https://github.com/budzianowski/multiwoz/raw/{multiwoz_sha}/data/MultiWOZ_{version}.zip',
        os.path.join(path, 'original.zip'))
    download_file(
        f'https://github.com/budzianowski/multiwoz/archive/{multiwoz_sha}.zip',
        os.path.join(path, 'repo.zip'))

    with zipfile.ZipFile(os.path.join(path, 'original.zip')) as zipf, \
            zipfile.ZipFile(os.path.join(path, 'repo.zip')) as dbzipf:
        export_lexicalizer_source(path)
        extract_databases(path, dbzipf, multiwoz_sha)
        delex_data = createDelexData(zipf, path)
        divideData(delex_data, zipf, path)

    # Generating blacklist
    logger.info('generating blacklist')
    cwd = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(['python', os.path.join(cwd, 'build_multiwoz_blacklist.py'), '--dataset', 'multiwoz-2.0'], cwd=cwd)


if __name__ == "__main__":
    download()
