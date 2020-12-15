#!/usr/bin/env python
import re
import sys
import json
import shutil
import types
import logging
import subprocess
import zipfile
from io import BytesIO
from itertools import chain
from tqdm import tqdm
from collections import OrderedDict, defaultdict, Counter
import requests
import os
import inspect

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging  # noqa: E402


setup_logging()
logger = logging.getLogger()
DATASETS_PATH = os.path.join(os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets')), 'augpt')
MW_DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
DEFAULT_IGNORE_VALUES = ['not mentioned', 'dont care', 'don\'t care', 'dontcare', 'do n\'t care', 'none']


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

            for domain, (count, results) in database_results.items():
                if count == 0:
                    continue
                result = results[result_index % len(results)]
                if label in result:
                    result_str = result[label]
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


DB_ONTOLOGY = True


class Database:
    def __init__(self, zipf):
        self.path = zipf.filename
        module = types.ModuleType('convlab_dbquery')
        exec(zipf.read('convlab_dbquery.py').decode('utf-8'), module.__dict__)
        convlab_database = getattr(module, 'Database')
        self.ignore_values = DEFAULT_IGNORE_VALUES
        self.supported_domains = MW_DOMAINS
        self._name_map = None
        self._ontology = None
        self._regexp = None

        # Load database files
        def hacked_init(self):
            self.dbs = {}
            for domain in MW_DOMAINS:
                with zipf.open(f'db/{domain}_db.json') as f:
                    self.dbs[domain] = json.load(f)

        setattr(convlab_database, '__init__', hacked_init)
        self.inner = getattr(module, 'Database')()

        # Load ontology
        if globals().get('DB_ONTOLOGY', True):
            with zipf.open('db_ontology.json') as f:
                self._ontology = {tuple(k.split('-')): set(v) for k, v in json.load(f).items()}
            self._build_replace_dict()

    price_re = re.compile(r'\d+\.\d+')

    @staticmethod
    def hack_query(belief):
        new_belief = OrderedDict()
        for domain, bs in belief.items():
            new_bs = OrderedDict()
            new_belief[domain] = new_bs
            for key, val in bs.items():
                val = bs[key]
                if domain == 'restaurant' and key == 'name' and val.lower() == 'charlie':
                    val = 'charlie chan'
                if domain == 'restaurant' and key == 'name' and val.lower() == 'good luck':
                    val = 'the good luck chinese food takeaway'
                # if domain == 'hotel' and key == 'name' and val.lower() == 'el shaddai guesthouse':
                #     val = 'el shaddai'
                new_bs[key] = val
        return new_belief

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

    @staticmethod
    def normalize_for_db(s):
        s = ','.join(s.split(' ,'))
        s = s.replace('swimming pool', 'swimmingpool')
        s = s.replace('night club', 'nightclub')
        s = s.replace('concert hall', 'concerthall')
        return s

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

    def _build_replace_dict(self):
        if self._regexp is not None:
            return
        clear_values = {'the', 'a', 'an', 'food'}
        clear_values.update(self._ontology[('hotel', 'type')])
        clear_values.update(self._ontology[('hotel', 'price range')])
        clear_values.update(self._ontology[('hotel', 'area')])
        clear_values.update(self._ontology[('restaurant', 'price range')])
        clear_values.update(self._ontology[('restaurant', 'food')])
        clear_values.update(self._ontology[('restaurant', 'area')])
        clear_values.update(self._ontology[('attraction', 'type')])
        clear_values = (f' {x} ' for x in clear_values)
        self._regexp = re.compile('|'.join(map(re.escape, clear_values)))
        db_entities = chain(self.inner.dbs['attraction'], self.inner.dbs['hotel'], self.inner.dbs['restaurant'])
        self._name_map = {self._clear_name(r): r['name'].lower() for r in db_entities}

    def _clear_name(self, domain_bs):
        name = ' ' + domain_bs['name'].lower() + ' '
        name = self._regexp.sub(' ', name)
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        return name

    @staticmethod
    def _to_minutes(time):
        hour, minutes = tuple(map(int, time.split(':')))
        return minutes + 60 * hour

    def __call__(self, belief, return_results=False):
        belief = Database.hack_query(belief)
        all_results = OrderedDict()
        for domain, domain_bs in belief.items():
            if domain not in self.supported_domains:
                continue  # skip unsupported domains
            if self.domain_not_empty(domain_bs) or \
                    domain in [d.lower() for d in {'Police', 'Hospital'}]:
                def query_single(domain_bs):
                    blocked_slots = {'people', 'booked', 'stay'}
                    if domain != 'train' and domain != 'bus':
                        blocked_slots.add('day')
                    query_bs = [(Database.translate_to_db_col(slot), Database.normalize_for_db(val))
                                for slot, val in domain_bs.items() if slot not in blocked_slots]
                    result = self.inner.query(domain, query_bs)
                    result = [Database.map_database_row(domain, k, domain_bs) for k in result]

                    # Implement sorting missing in convlab
                    if domain == 'train' and 'arrive by' in domain_bs:
                        result.sort(key=lambda x: self._to_minutes(x['arrive by']), reverse=True)
                    elif domain == 'train' and 'leave at' in domain_bs:
                        result.sort(key=lambda x: self._to_minutes(x['leave at']))
                    return result
                result = query_single(domain_bs)
                if len(result) == 0 and 'name' in domain_bs and self._clear_name(domain_bs) in self._name_map:
                    domain_bs = dict(**domain_bs)
                    domain_bs['name'] = self._name_map[self._clear_name(domain_bs)]
                    result = query_single(domain_bs)

                if return_results:
                    all_results[domain] = (len(result), result)
                else:
                    all_results[domain] = len(result)
        return all_results

    def save(self, path):
        shutil.copy(self.path, os.path.join(path, os.path.split(self.path)[-1]))


class BeliefStateTransformation:
    def __init__(self):
        self.ignore_values = DEFAULT_IGNORE_VALUES

    def _process_domain(self, domain_bs):
        return {self._map_slot(slot): self._clear_value(val) for slot, val in domain_bs.items()
                if (len(val) > 0 and val not in self.ignore_values)}

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

    @staticmethod
    def domain_not_empty(domain_bs, ignore_values):
        return any(len(val) > 0 and val not in ignore_values for val in domain_bs.values())

    def __call__(self, belief_state, dialogue_act, active_domain):
        clean_belief = dict()
        for domain, domain_bs in belief_state.items():
            new_domain_bs = {}
            if 'semi' in domain_bs:
                new_domain_bs.update(domain_bs['semi'])
            if 'book' in domain_bs:
                new_domain_bs.update({k: v for k, v in domain_bs['book'].items() if k != 'booked'})
            if not BeliefStateTransformation.domain_not_empty(domain_bs, self.ignore_values):
                continue
            new_domain_bs = self._process_domain(new_domain_bs)
            # TODO: uncomment for new iteration
            # if len(new_domain_bs) == 0:  # TODO: remove this condition in next iteration
            #     continue
            if 'internet' in new_domain_bs and new_domain_bs['internet'] == 'no':
                del new_domain_bs['internet']  # no internet by default
            if 'parking' in new_domain_bs and new_domain_bs['parking'] == 'no':
                del new_domain_bs['parking']  # no parking by default

            # TODO: comment for new iteration
            if len(new_domain_bs) > 0:  # TODO: remove this condition in next iteration
                clean_belief[domain] = new_domain_bs

        for domain in {'Police', 'Hospital'}:
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


def clear_whitespaces(text):
    text = re.sub(r'[\s\n\r]+', ' ', text)
    text = ' ' + text + ' '
    text = re.sub(r'\s([,\.:\?\!\']+)', lambda m: m.group(1), text)
    return text.strip()


def build_fix_belief_from_database(database_engine):
    _clear_dict = None

    def clear_dict():
        nonlocal _clear_dict
        if _clear_dict is None:
            # Build clear dict
            _clear_dict = dict()
            db_values = set()
            for x in database_engine.inner.dbs['attraction']:
                db_values.add(x['name'])
            _clear_dict = OrderedDict((x.replace("'", ''), x) for x in db_values)
        return _clear_dict

    def call(belief):
        # fix belief state, put back apostrophs
        for domain, bs in belief.items():
            for key, value in bs.items():
                bs[key] = clear_dict().get(value, value)
        return belief
    return call


def normalize(text):
    text = text.replace('swimmingpool', 'swimming pool')
    text = text.replace('nigthclub', 'night club')
    text = text.replace('Shanghi', 'Shanghai')
    return text


DELEX_LABEL_MAP = {
    'Price': 'price range',
    'Fee': None,  # 'fee',
    'Addr': 'address',
    'Area': 'area',
    'Stars': 'stars',
    'Department': None,  # 'department',
    'Stay': None,  # 'stay',
    'Ref': 'reference',
    'Food': 'food',
    'Type': 'type',
    'Choice': None,  # ignore
    'Phone': 'phone',
    'Ticket': 'price',
    'Day': None,  # 'day',
    'Name': 'name',
    'Car': 'car',
    'Leave': 'leave at',
    'Time': 'time',
    'Arrive': 'arrive by',
    'Post': 'postcode',
    'Depart': None,  # 'departure',
    'People': None,  # 'people',
    'Dest':  None,  # 'destination',
    'Open': None,  # ignore
    'Id': 'id',
}


def delexicalise_spans(response, spans, allowed_slots=None):
    allowed_slots = set(allowed_slots or [])
    # First, we clear the spans
    new_spans = []
    for i, span in enumerate(spans):
        if span[1] == 'Fee' and 'vary' in span[-3]:
            pass
        elif DELEX_LABEL_MAP[span[1]] not in allowed_slots:
            pass
        else:
            new_spans.append(span)
    spans = new_spans

    delex = []
    assignment = []
    textlen = 0
    for i, original in enumerate(response.split()):
        for span in spans:
            label = DELEX_LABEL_MAP[span[1]]
            textlen += 1 + len(original)
            if label is None:
                continue  # causes this token to be added
            if label == 'time' and ('minute' in span[-3] or 'hour' in span[-3] or 'day' in span[-3]):

                label = 'duration'
            if original in {',', '.', ':'}:
                if i == span[3]:
                    delex.append(original)
                    delex.append(f'[{label}]')
                    assignment.append((label, None, textlen))
                else:
                    continue
            if i == span[3]:
                if label == 'stars' and '-star' in original:
                    number, ext = original.split('-')
                    delex.append(f'[{label}]-{ext}')
                    original = number
                    assignment.append((label, original, textlen - len(original)))
                elif label == 'area' and original == 'the':
                    delex.append('the')
                    delex.append(f'[{label}]')
                    original = None
                    assignment.append((label, original, textlen))
                elif label == 'area' and original == 'in' and span[-3].startswith('in the '):
                    delex.extend(['in', 'the'])
                    delex.append(f'[{label}]')
                    original = None
                    assignment.append((label, original, textlen))
                elif label == 'time' and original == 'a':
                    delex.append('a')
                    delex.append(f'[{label}]')
                    original = None
                    assignment.append((label, original, textlen))
                elif label == 'stay' and 'day' in original:
                    delex.append(f'[{label}]')
                    delex.append('days' if 'days' in original else 'day')
                    assignment.append((label, original, textlen - len(original)))
                elif label == 'address' and len(delex) >= 2 and delex[-1] == ',' and delex[-2] == '[address]':
                    delex.pop()
                    label, text, index = assignment[-1]
                    assignment[-1] = (label, f'{text} , {original}', index)
                else:
                    delex.append(f'[{label}]')
                    assignment.append((label, original, textlen - len(original)))
                break
            elif span[3] < i <= span[4]:
                # already added the label
                label, text, index = assignment[-1]
                if text is None:
                    text = original
                else:
                    text = f'{text} {original}'
                if i == span[4] and label == 'area' and text.endswith(' of town'):
                    delex.extend(['of', 'town'])
                    text = text[:-len(' of town')]
                if i == span[4] and label == 'time' and text.endswith(' ride'):
                    delex.append('ride')
                    text = text[:-len(' ride')]

                assignment[-1] = (label, text, index)
                break
        else:
            delex.append(original)
    return ' '.join(delex), assignment


def delexicalise(utt, return_replacements=False, database_results=None, belief=None, spans=None):
    database_results = Lexicalizer.extend_database_results(database_results, belief)

    # Delexicalise only the stuff that we can put back
    allowed_keys = {'reference'}  # always delex reference
    for domain, (count, results) in database_results.items():
        if count > 0:
            allowed_keys.update(results[0].keys())
    if 'arrive by' in allowed_keys or 'leave at' in allowed_keys:
        allowed_keys.add('time')

    # First we use the span_info annotations
    spans = sorted(spans, key=lambda x: x[-2])
    utt, replacements = delexicalise_spans(utt, spans, allowed_keys)
    if return_replacements:
        replacements = [x[:2] for x in replacements]
        # replacements.sort(key=lambda k: k[0])
        return utt, replacements
    return utt


def export_convlab_data(path, zipf, commit_sha):
    global DB_ONTOLOGY

    def da2tuples(dialog_act):
        tuples = []
        for domain_intent, svs in dialog_act.items():
            for slot, value in sorted(svs, key=lambda x: x[0]):
                domain, intent = domain_intent.split('-')
                tuples.append([intent, domain, slot, value])
        return tuples

    transform_belief = BeliefStateTransformation()
    DB_ONTOLOGY = False
    with zipfile.ZipFile(os.path.join(path, 'database.zip')) as dbzipf:
        db = Database(dbzipf)
    DB_ONTOLOGY = True
    fix_belief_from_database = build_fix_belief_from_database(db)
    ontology = defaultdict(lambda: set())
    splits = []
    for split in ['train', 'val', 'test']:
        ignored_dialogues = 0
        dialogues = []
        splits.append((split, dialogues))
        with zipfile.ZipFile(BytesIO(zipf.read(f'ConvLab-2-{commit_sha}/data/multiwoz/{split}.json.zip'))) as zsplitf:
            data = json.load(zsplitf.open(f'{split}.json'))
        logger.info('loaded {}, size {}'.format(split, len(data)))
        for sess_id, sess in data.items():
            goal = parse_goal(sess['goal'])
            dialogue = dict(name=sess_id, items=[], goal=goal)
            active_domain = None
            ignore_dialogue = False
            for i, turn in enumerate(sess['log']):
                text = turn['text']
                da = da2tuples(turn['dialog_act'])
                item = dict(
                    speaker='user' if i % 2 == 0 else 'system',
                    text=text,
                    dialogue_act=da
                )

                if item['speaker'] == 'system':
                    belief = turn['metadata']
                    item['span_info'] = turn['span_info']

                    # Detect active domain
                    domain_counter = Counter({x[1].lower() for x in da}.intersection(MW_DOMAINS))
                    if domain_counter:
                        active_domain = domain_counter.most_common(1)[0][0]
                    item['active_domain'] = active_domain

                    belief = transform_belief(belief, da, active_domain)
                    belief = fix_belief_from_database(belief)
                    item['belief'] = belief
                    if 'bus' in belief:
                        # We need to ignore this dialogue
                        # There is no data for the bus domain
                        ignore_dialogue = True
                        break

                    for k, bs in belief.items():
                        for k2, val in bs.items():
                            ontology[(k, k2)].add(val)

                    # Add booked property
                    item['booked_domains'] = sorted(get_booked_domains(turn['metadata']))

                dialogue['items'].append(item)

            if not ignore_dialogue:
                dialogues.append(dialogue)
            else:
                ignored_dialogues += 1
        if ignored_dialogues > 0:
            logger.warning(f'dialogues were ignored {ignored_dialogues * 100 / (ignored_dialogues + len(dialogues)):.2f}% due to a missing domain "bus"')  # noqa: E501

    # Save db ontology
    with zipfile.ZipFile(os.path.join(path, 'database.zip'), 'a') as dbzipf:
        with dbzipf.open('db_ontology.json', 'w') as f:
            f.write(json.dumps({'-'.join(k): list(v) for k, v in ontology.items()}).encode('utf-8'))

        db = Database(dbzipf)

    # Delexicalize loaded data
    for split, dialogues in splits:
        for dialogue in tqdm(dialogues, desc=f'delexicalising {split}'):
            for item in dialogue['items']:
                text = item['text']
                if item['speaker'] == 'system':
                    belief = item['belief']
                    span_info = item['span_info']
                    del item['span_info']

                    database_results = db(belief, return_results=True)
                    delexicalised_text = delexicalise(text, return_replacements=False,
                                                      database_results=database_results,
                                                      belief=belief, spans=span_info)
                    item['delexicalised_text'] = clear_whitespaces(delexicalised_text)
                    database_results = OrderedDict((domain, count) for domain, (count, results)
                                                   in database_results.items())
                    item['database'] = database_results

                text = normalize(text)
                item['text'] = clear_whitespaces(text)

        with open(os.path.join(path, f'{split}.json'), 'w+') as f:
            json.dump(dict(dialogues=dialogues, domains=MW_DOMAINS), f)


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


def get_database_source():
    return f"""from collections import OrderedDict
import re
import os
import importlib
import json
import types
import shutil
import zipfile
from itertools import chain


MW_DOMAINS = {repr(MW_DOMAINS)}
DEFAULT_IGNORE_VALUES = {repr(DEFAULT_IGNORE_VALUES)}


{inspect.getsource(Database)}"""


def extract_databases(path, zipf, commit_sha):
    with zipfile.ZipFile(os.path.join(path, 'database.zip'), 'w') as dbzipf:
        for filename in zipf.namelist():
            if filename.startswith(f'ConvLab-2-{commit_sha}/data/multiwoz/db/'):
                if filename[-1] == '/':
                    continue
                fname = filename[len(f'ConvLab-2-{commit_sha}/data/multiwoz/db/'):]
                fname = os.path.join('db', fname)
                with zipf.open(filename) as zf, dbzipf.open(fname, 'w') as f:
                    shutil.copyfileobj(zf, f)

        # Copy original convlab query file
        with zipf.open(f'ConvLab-2-{commit_sha}/convlab2/util/multiwoz/dbquery.py') as sourcef, \
                dbzipf.open('convlab_dbquery.py', 'w') as f:
            shutil.copyfileobj(sourcef, f)

        with dbzipf.open('database.py', 'w') as f:
            f.write(get_database_source().encode('utf-8'))
            f.flush()


def download_file(source_url, dest_path):
    response = requests.get(source_url, stream=True, timeout=5)
    response.raise_for_status()
    file_size = int(response.headers.get('content-length', 0))
    if "/" in dest_path:
        dir = "/".join(dest_path.split("/")[0:-1])
        os.makedirs(dir, exist_ok=True)
    if os.path.exists(dest_path):
        return

    pbar = tqdm(
        total=file_size, unit='B', disable=file_size < 1024**2,
        unit_scale=True, desc=source_url.split('/')[-1])

    with open(f'{dest_path}.tmp', "wb") as file:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            pbar.update(1024)
    pbar.close()
    shutil.move(f'{dest_path}.tmp', dest_path)


def download():
    path = os.path.join(DATASETS_PATH, 'multiwoz-2.1')
    os.makedirs(path, exist_ok=True)

    # Installing requirements first
    with open(os.path.join(path, 'requirements.txt'), 'w+') as f:
        print('fuzzywuzzy==0.18.0', file=f)
        print('python-Levenshtein==0.12.0', file=f)
        f.flush()

    logger.info('Installing requirements')
    subprocess.check_call([sys.executable, "-m", "pip", "install", '-r', os.path.join(path, 'requirements.txt')])

    # Download the dataset
    commit_sha = 'e368deeb3d405caf19236fb768360a6517a24fcd'
    download_file(
        f'https://github.com/thu-coai/ConvLab-2/archive/{commit_sha}.zip',
        os.path.join(path, 'original.zip'))
    with zipfile.ZipFile(os.path.join(path, 'original.zip')) as zipf:
        extract_databases(path, zipf, commit_sha)
        export_lexicalizer_source(path)
        export_convlab_data(path, zipf, commit_sha)

    # Generating blacklist
    logger.info('Generating blacklist')
    cwd = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(['python', os.path.join(cwd, 'build_multiwoz_blacklist.py'), '--dataset', 'multiwoz-2.1'], cwd=cwd)


if __name__ == "__main__":
    download()
