import torch
import random
import dataclasses
from collections import OrderedDict
from .utils import format_belief


class NegativeSamplingDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, inner, transform=None):
        self.inner = inner
        self.transform = transform
        assert hasattr(self.inner, 'ontology')
        assert self.inner.ontology is not None
        self.ontology = {k: sorted(v) for k, v in self.inner.ontology.items()}

    def __len__(self):
        return 2 * len(self.inner)

    def __getitem__(self, i):
        item = self.inner[i // 2]
        negative = i % 2
        if negative:
            negative = False
            belief, response, context = item.belief, item.response, item.context
            raw_belief = item.raw_belief
            negative_type = random.randrange(1, 8)
            use_new_belief = (negative_type // 4) % 2
            modify_belief = (negative_type // 2) % 2
            use_new_response = negative_type % 2

            # Negative resonse
            negative_sample = random.randrange(len(self.inner))
            neg_sample = self.inner[negative_sample]

            if use_new_belief:
                raw_belief = neg_sample.raw_belief
                negative = True
            if modify_belief:
                # Use different slot values
                new_belief = OrderedDict()
                for k, bs in item.raw_belief.items():
                    new_belief[k] = dict()
                    for k2, v in bs.items():
                        choices = [x for x in self.ontology[(k, k2)] if x != v]
                        if len(choices) == 0:
                            new_belief[k][k2] = v
                        else:
                            new_belief[k][k2] = random.choice(choices)
                            negative = True
                raw_belief = new_belief
            if use_new_response:
                # Negative resonse
                negative_sample = random.randrange(len(self.inner))
                neg_sample = self.inner[negative_sample]
                response = neg_sample.response  # negative response
                negative = True
            belief = format_belief(raw_belief)
            item = dataclasses.replace(item, context=context, belief=belief,
                                       raw_belief=raw_belief, response=response, positive=not negative)
        return self.transform(item)


class NegativeSamplerWrapper(torch.utils.data.Sampler):
    def __init__(self, inner):
        self.inner = inner

    @property
    def num_samples(self):
        return 2 * self.inner.num_samples

    def __iter__(self):
        for index in iter(self.inner):
            yield 2 * index
            yield 2 * index + 1

    def set_epoch(self, epoch):
        if hasattr(self.inner, 'set_epoch'):
            self.inner.set_epoch(epoch)

    def __len__(self):
        return self.num_samples
