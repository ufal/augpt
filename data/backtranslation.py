import logging
import yaml
import dataclasses
import random
import datasets
from data.utils import DialogDatasetItem

logger = logging.getLogger()


def load_backtranslations(dataset_names):
    download_config = datasets.DownloadConfig()
    local_file_names = [x for x in dataset_names if x.endswith('.yaml')]
    remote_file_names = [x for x in dataset_names if x not in local_file_names]
    manager = datasets.DownloadManager('jkulhanek/augpt-backtranslations', download_config=download_config)
    urls = {d: datasets.utils.hf_bucket_url('jkulhanek/augpt-backtranslations', f'{d}.yaml') for d in remote_file_names}
    local_file_names.extend(manager.download_and_extract(urls).values())
    backtranslations = dict()
    for filename in local_file_names:
        with open(filename) as f:
            backtranslations.update(yaml.safe_load(f))
    return backtranslations


class BackTranslateAugmentation:
    def __init__(self, dictionary, seed=42):
        self.dictionary = dictionary
        self._rng = random.Random(seed)
        self._num_errors = 0
        self._total = 0

    def _map(self, text: str) -> str:
        self._total += 1
        if text in self.dictionary:
            options = self.dictionary[text] + [text]
            return self._rng.choice(options)

        self._num_errors += 1
        logging.warning(
            'cannot backtranslate, unknown text in the dataset, missing {0:2.2f}%'.format(self._num_errors * 100 / self._total))

        # More than 10% of the translations are missing
        if self._total > 1000 and self._num_errors * 10 > self._total:
            logging.error('more than 10% of translations are missing')
            raise Exception('Backtranslation dictionary is missing')
        return text

    def __call__(self, item: DialogDatasetItem) -> DialogDatasetItem:
        context = [self._map(x) for x in item.context]
        return dataclasses.replace(item, context=context)
