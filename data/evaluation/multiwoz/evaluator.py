import re
from functools import partial
from tqdm import tqdm
from collections import defaultdict, Counter
from data.loader import load_dataset
from data.utils import BeliefParser


class BaseEvaluator(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp):
        raise NotImplementedError

    def get_report(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


def get_logger():
    import logging  # noqa:F811
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


class MultiWozEvaluator(BaseEvaluator):
    def __init__(self, dataset, is_multiwoz_eval=False, logger=None):
        self.db = dataset.database
        self.dataset = dataset
        self.labels = list()
        self.hyps = list()
        self.belief_parser = BeliefParser()
        self.is_multiwoz_eval = is_multiwoz_eval
        self.logger = logger or get_logger()
        self.label_regex = re.compile(r'\[([\w\d\s]+)\]')

    def _query_original_db(self, domain, belief):
        belief = {domain: belief}
        return self.db(belief, return_results=True)[domain][1]

    def _get_requestables_and_venues(self, beliefs, responses, dialog_booked_domains):
        # for computing corpus success
        requestables = {'phone', 'address', 'postcode', 'reference', 'id'}
        provided_requestables = defaultdict(lambda: set())
        venue_offered = defaultdict(lambda: [])
        for i, (belief, response, booked_domains) in enumerate(zip(beliefs, responses, dialog_booked_domains)):
            database_results = self.db(belief, return_results=True)
            current_requestables = set(self.label_regex.findall(response))
            self.logger.debug(response)
            current_domain = next(iter(belief.keys())) if belief else None
            self.logger.debug(f'domain: {current_domain}, r: {current_requestables}')
            self.logger.debug(f"belief: {belief.get('hotel', None)}")
            self.logger.debug(f"db: {database_results.get('hotel', None)}")

            # Parse multiwoz style requestables first
            legacy_requestables = {x for x in current_requestables if '_' in x}
            current_requestables.difference_update(legacy_requestables)
            for requestable_candidate in legacy_requestables:
                domain, slot = requestable_candidate.split('_')
                if slot not in requestables:
                    continue

                # https://github.com/budzianowski/multiwoz/blob/a24d299fafa00371d03880bce34cb3b0923518fa/evaluate.py#L248
                # if db pointer was allowing for that
                if slot == 'reference' and domain in {'restaurant', 'hotel', 'train'}:
                    if domain not in booked_domains:
                        continue

                provided_requestables[domain].add(slot)

            # New style delexicalization
            for domain, (num_results, results) in database_results.items():
                if not self.is_multiwoz_eval:
                    current_delex_requestables = set(belief.get(domain, dict()).keys())
                    if num_results > 0:
                        current_delex_requestables.update(results[0].keys())

                    matched_requestables = current_requestables.intersection(current_delex_requestables)
                    if 'reference' in matched_requestables and domain in {'restaurant', 'hotel', 'train'}:
                        # https://github.com/budzianowski/multiwoz/blob/a24d299fafa00371d03880bce34cb3b0923518fa/evaluate.py#L248
                        # if db pointer was allowing for that
                        if domain not in booked_domains:
                            matched_requestables.remove('reference')

                    current_requestables -= matched_requestables
                    provided_requestables[domain].update(matched_requestables.intersection(requestables))

                # Venues offered
                if 'name]' in response or 'id]' in response:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        venues = results
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = venues
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0].get('id') == ven.get('id'):
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                venue_offered[domain] = venues
                    else:
                        venue_offered[domain] = domain + '_name'

            # These slots are not lexicalised back, but its is not a concern
            # in multiwoz evaluation which does not take it into account
            if current_domain and self.is_multiwoz_eval:
                provided_requestables[current_domain].update(current_requestables)
        return provided_requestables, venue_offered

    def _evaluate_generated_dialogue(self, real_requestables, provided_requestables,
                                     venue_offered, goal, stats):
        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            # if realDialogue['goal'][domain].has_key('info'):
            if 'informable' in goal[domain]:
                # if realDialogue['goal'][domain]['info'].has_key('name'):
                if 'name' in goal[domain]['informable']:
                    venue_offered[domain] = domain + '_name'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = domain + '_name'

            # if id was not requested but train was found we dont want
            # to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = domain + '_name'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        stat_domain_total, stat_domain_match, stat_domain_success = stats

        # MATCH
        match = 0.0
        for domain in goal.keys():
            domain_success = False
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self._query_original_db(domain, goal[domain]['informable'])

                if isinstance(venue_offered[domain], str):
                    if '_name' in venue_offered[domain]:
                        domain_success = True
                elif len(venue_offered[domain]) > 0:
                    reference = venue_offered[domain][0]['id']
                    if any(isinstance(x, dict) and x.get('id') == reference for x in goal_venues):
                        domain_success = True
            else:
                if domain + '_name' in venue_offered[domain]:
                    domain_success = True
            match += domain_success
            stat_domain_total[domain] += 1
            stat_domain_match[domain] += domain_success

        if match == len(goal.keys()):
            match = 1.0
        else:
            match = 0.0

        # SUCCESS
        success = 0.0
        if match == 1.0:
            for domain in goal.keys():
                # if values in sentences are super set of requestables
                prov_req = provided_requestables[domain].intersection(real_requestables[domain])
                domain_success = len(prov_req) == len(real_requestables[domain])
                # if not domain_success:
                #    # print('HUPS', domain, provided_requestables[domain], real_requestables[domain])
                success += domain_success
                stat_domain_success[domain] += domain_success

            if success >= len(real_requestables):
                success = 1.0
            else:
                success = 0.0

        if success == 0:
            # print((real_requestables, provided_requestables))
            pass
        return success, match

    def _get_goal_and_requestables(self, gt_belief, goal):
        # compute corpus success
        real_requestables = {}
        provided_requestables = {}
        venue_offered = {}
        for domain in goal.keys():
            provided_requestables[domain] = []
            venue_offered[domain] = []
            real_requestables[domain] = goal[domain]['requestable']
        return goal, real_requestables

    def _parse_entities(self, tokens):
        entities = []
        for t in tokens:
            if '[' in t and ']' in t:
                entities.append(t)
        return entities

    def pack_dialogues(self, dataset, beliefs, responses):
        def batch_dialog(dialog):
            (items, goals, beliefs, responses, booked_domains) = tuple(zip(*dialog))
            return items, goals[0], beliefs, responses, booked_domains

        if isinstance(dataset, str):
            dataset = load_dataset(dataset, goal=True)
        current_dialogue = []
        for item, belief, response in zip(dataset, beliefs, responses):
            if len(item.context) == 1:
                if current_dialogue:
                    yield batch_dialog(current_dialogue)
                current_dialogue = []
            # print(item, item.goal)
            current_dialogue.append((item, item.goal, belief, response, item.booked_domains))
            # current_dialogue.append((item, item.goal, item.raw_belief, item.response))
        yield batch_dialog(current_dialogue)

    def evaluate(self, beliefs, responses, progressbar=False):
        dialogues = self.pack_dialogues(self.dataset, beliefs, responses)
        successes, matches = 0, 0
        stats = tuple(Counter() for _ in range(3))
        domain_total, domain_match, domain_success = stats
        total = 0

        offset = 0
        progress = tqdm(total=len(self.dataset),
                        desc=progressbar if isinstance(progressbar, str) else 'evaluating',
                        disable=not progressbar)
        for idx, (items, goal, beliefs, responses, booked_domains) in enumerate(dialogues):
            goal, real_requestables = self._get_goal_and_requestables(items[-1].raw_belief, goal)
            self.logger.debug(f'rr: {real_requestables}, g: {goal}')
            provided_requestables, venue_offered = self._get_requestables_and_venues(beliefs, responses, booked_domains)
            success, match = self._evaluate_generated_dialogue(
                real_requestables, provided_requestables, venue_offered, goal, stats)
            successes += success
            matches += match
            total += 1
            offset += len(items)
            progress.update(len(items))

        domain_results = dict()
        for key in domain_total.keys():
            domain_results[key] = domain_match[key] / \
                float(domain_total[key]), domain_success[key] / float(domain_total[key])

        return successes / float(total), matches / float(total), domain_results


reference_regex = re.compile(r'(?:^|[^a-zA-Z0-9])(?=[A-Z0-9]{8}(?:[^a-zA-Z0-9]|$))([A-Z0-9]*[A-Z][A-Z0-9]*|0{4}\d{4})')  # noqa:E501


def remove_reference(text):
    def rmref(x):
        return x.group(0).replace(x.group(1), 'REFERENCE')
    return partial(reference_regex.sub, rmref)


def compute_bleu_remove_reference(responses, gold_responses):
    import nltk
    responses = map(lambda x: x.lower(), responses)
    gold_responses = map(lambda x: x.lower(), gold_responses)

    responses = list(map(nltk.tokenize.word_tokenize, responses))
    gold_responses = map(nltk.tokenize.word_tokenize, gold_responses)
    gold_responses = list(map(lambda x: [x], gold_responses))
    bleu = nltk.translate.bleu_score.corpus_bleu(gold_responses, responses)
    return bleu
