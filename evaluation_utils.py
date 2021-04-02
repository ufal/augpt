import re
import operator
from functools import reduce, partial
import pipelines  # noqa
import nltk


def compute_bleu(responses, gold_responses):
    responses = map(lambda x: x.lower(), responses)
    responses = list(map(nltk.tokenize.word_tokenize, responses))
    gold_responses = map(lambda x: x.lower(), gold_responses)
    gold_responses = map(nltk.tokenize.word_tokenize, gold_responses)
    gold_responses = list(map(lambda x: [x], gold_responses))
    bleu = nltk.translate.bleu_score.corpus_bleu(gold_responses, responses)
    return bleu


def compute_rouge(responses, gold_responses):
    from rouge import Rouge
    responses = map(lambda x: x.lower(), responses)
    responses = map(nltk.tokenize.word_tokenize, responses)
    responses = list(map(lambda x: ' '.join(x), responses))
    gold_responses = map(lambda x: x.lower(), gold_responses)
    gold_responses = map(nltk.tokenize.word_tokenize, gold_responses)
    gold_responses = list(map(lambda x: ' '.join(x), gold_responses))
    rouge = Rouge().get_scores(responses, gold_responses, avg=True)
    rouge = {f'{k}-{k1}': v for k, sub in rouge.items() for k1, v in sub.items()}
    return rouge


def compute_delexicalized_bleu(responses, gold_responses):
    token_regex = re.compile(r'\[([\w\s\d]+)\]')
    token_sub = partial(token_regex.sub, lambda x: x.group(1).upper().replace(' ', ''))
    responses = map(lambda x: x.lower(), responses)
    responses = map(token_sub, responses)
    gold_responses = map(lambda x: x.lower(), gold_responses)
    gold_responses = map(token_sub, gold_responses)
    return compute_bleu(responses, gold_responses)


def compute_delexicalized_rouge(responses, gold_responses):
    token_regex = re.compile(r'\[([\w\s\d]+)\]')
    token_sub = partial(token_regex.sub, lambda x: x.group(1).upper().replace(' ', ''))
    responses = map(lambda x: x.lower(), responses)
    responses = map(token_sub, responses)
    gold_responses = map(lambda x: x.lower(), gold_responses)
    gold_responses = map(token_sub, gold_responses)
    data = filter(lambda x: all(x), zip(responses, gold_responses))
    data = list(data)
    responses = (x[0] for x in data)
    gold_responses = (x[1] for x in data)
    return compute_rouge(responses, gold_responses)


def compute_sentence_bleu(response, gold_responses):
    responses = nltk.tokenize.sent_tokenize(response.lower())
    responses = list(map(nltk.tokenize.word_tokenize, responses))
    gold_responses = reduce(operator.add, (nltk.tokenize.sent_tokenize(x.lower()) for x in gold_responses))
    gold_responses = list(map(nltk.tokenize.word_tokenize, gold_responses))
    bleu = nltk.translate.bleu_score.corpus_bleu(gold_responses, responses)
    return bleu
