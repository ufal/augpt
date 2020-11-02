import os
import functools
import logging
from data.utils import ConcatDialogDataset, split_name, wrap_dataset_with_blacklist

RESTRICTED_DOMAINS = ['hotel', 'train', 'restaurant', 'attraction', 'taxi',
                      'hospital', 'police', 'rentalcar', 'flight', 'hotels',
                      'restaurant-search', 'flights']
DATASETS_PATH = os.path.join(os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets')), 'augpt')
logger = logging.getLogger()


def load_dataset(name, restrict_domains=False, augment='disabled', use_blacklist=False, **kwargs):
    if restrict_domains:
        return load_dataset(name, domains=RESTRICTED_DOMAINS, **kwargs)

    if '+' in name:
        # This is a concat dataset
        datasets = name.split('+')
        _load_dataset = functools.partial(load_dataset, **kwargs)
        datasets = list(map(_load_dataset, datasets))
        return ConcatDialogDataset(datasets)

    dataset_name, split = split_name(name)

    from data.dataset import load_dataset as load_custom_dataset
    dataset = load_custom_dataset(name, **kwargs)

    if use_blacklist:
        dataset = add_blacklist(dataset, name)
    return dataset


def add_blacklist(dataset, name):
    dataset_name, split = split_name(name)
    with open(os.path.join(DATASETS_PATH, dataset_name, f'{split}-blacklist.txt'), 'r') as f:
        blacklist = sorted(set(int(x.rstrip()) for x in f))
    logging.warning(f'Some examples ({100 * len(blacklist) / len(dataset):.2f}%) were ignored by a blacklist.')
    return wrap_dataset_with_blacklist(dataset, blacklist)


def load_backtranslation_transformation(name):
    import data.backtranslation

    def get_backtranslation_datasets(name):
        if '+' in name:
            datasets = name.split('+')
            return sum(map(get_backtranslation_datasets, datasets), [])

        if name.endswith('.yaml'):
            return [name]

        new_name, split = split_name(name)
        if split in {'dev', 'val', 'train', 'test', 'validation', 'training', 'testing', 'development'}:
            name = new_name
        if name == 'multiwoz-2.0':
            # NOTE: we do not have backtranslations for MultiWOZ 2.0
            return ['multiwoz-2.1']
        return [name]

    backtranslation_dict = data.backtranslation.load_backtranslations(list(set(get_backtranslation_datasets(name))))
    return data.backtranslation.BackTranslateAugmentation(backtranslation_dict)
