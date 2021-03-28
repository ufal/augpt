import os
import sys
import types
import shutil
import logging
import requests
import torch
import transformers
import zipfile
from collections import OrderedDict

# We will fix transformers remote url resolution for older transformer versions
def _fix_transformers_url():
    HUGGINGFACE_CO_PREFIX = "https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
    from packaging import version
    from typing import Optional
    if version.parse(transformers.__version__) < version.parse('3.5'):
        def hf_bucket_url(model_id: str, filename: str, subfolder: Optional[str] = None, revision: Optional[str] = None, mirror=None, **kwargs) -> str:
            if subfolder is not None:
                filename = f"{subfolder}/{filename}"

            if revision is None:
                revision = "main"
            return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)
        setattr(transformers.file_utils, 'hf_bucket_url', hf_bucket_url)

_fix_transformers_url()    
logger = logging.getLogger()
DATASETS_PATH = os.path.join(os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets')), 'augpt')


def seed(r_seed):
    import numpy as np
    import random
    import torch
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)
    random.seed(r_seed)


def get_shape(data):
    if isinstance(data, dict):
        return {key: get_shape(value) for key, value in data.items()}
    if isinstance(data, list):
        return list(map(get_shape, data))
    if isinstance(data, tuple):
        return list(map(get_shape, data))
    return data.shape


def to_tensor(value, dtype=torch.float32):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype)
    return torch.tensor(value, dtype=dtype)


class Metric:
    def __init__(self):
        self.reset_states()

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            return self.update_state(*args, **kwargs)
        else:
            return self.report()


class Mean(Metric):
    def reset_states(self):
        self.cumsum = 0.0
        self.samples = 0.0

    def update_state(self, value, weight=None):
        value = to_tensor(value, dtype=torch.float32)
        weight = to_tensor(weight, dtype=torch.float32)
        if weight is None and hasattr(value, 'shape'):
            weight = torch.prod(to_tensor(value.shape))
            value = value.sum()
        if weight is None:
            weight = to_tensor(1.0, dtype=torch.float32)

        self.samples += weight.item()
        self.cumsum += value.item()

    def report(self):
        if self.samples == 0:
            return 0
        return self.cumsum / self.samples


class F1(Metric):
    def reset_states(self):
        self.tp, self.fp, self.fn = 0, 0, 0

    def update_state(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def report(self):
        if self.tp == 0:
            return 0
        return self.tp / (self.tp + 0.5 * (self.fp + self.fn))


class Accuracy(Mean):
    def update_state(self, y, target):
        y = y.detach()
        target = target.detach()
        with torch.no_grad():
            _, predicted = y.max(-1)
            value = torch.logical_and(
                predicted == target, target != -100).float().sum().cpu()
            weight = (target != -100).float().sum().cpu()
            super().update_state(value, weight)


class LanguageAccuracy(Accuracy):
    def update_state(self, lm_logits, labels):
        lm_logits = lm_logits.detach()
        labels = labels.detach()
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        super().update_state(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class BinaryAccuracy(Mean):
    def update_state(self, y, target, weight=None):
        y = y.detach()
        target = target.detach()
        with torch.no_grad():
            if weight is None:
                weight = torch.ones_like(target).float()
            else:
                wshape = weight.shape
                weight = weight.view(wshape + tuple(1 for _ in y.shape[len(wshape):]))
                weight = weight.expand(wshape + y.shape[len(wshape):])
            predicted = y > 0.5
            target = target > 0.5
            value = ((predicted == target).float() * weight).sum()
            weight = weight.sum()
            super().update_state(value, weight)


class DistributedMetricWrapper(Metric):
    def __init__(self, metric, master, name):
        self.metric = metric
        self.master = master
        self.name = name

    def report(self, *args, **kwargs):
        value = self.metric.report(*args, **kwargs)
        return self.master.collect_metric(self.name, value)

    def update_state(self, *args, **kwargs):
        self.master.invalidate_metrics()
        return self.metric.update_state(*args, **kwargs)

    def reset_states(self, *args, **kwargs):
        self.master.invalidate_metrics()
        return self.metric.reset_states(*args, **kwargs)


class DistributedMetricsDict:
    def __init__(self, **metrics):
        metrics = OrderedDict(**metrics)
        self.metrics = metrics
        self.cached_values = None

    def __getitem__(self, key):
        return DistributedMetricWrapper(self.metrics[key], self, key)

    def items(self):
        for key in self.metrics.keys():
            yield key, self[key]

    def collect_metric(self, key, value):
        if self.cached_values is None:
            values = [x() for x in self.metrics.values()]
            values = torch.tensor(values, dtype=torch.float32).cuda()
            torch.distributed.all_reduce(values)
            values /= torch.distributed.get_world_size()
            values = values.cpu()
            self.cached_values = {key: v.item() for key, v in zip(self.metrics.keys(), values)}
        return self.cached_values[key]

    def invalidate_metrics(self):
        self.cached_values = None


def setup_logging(level=logging.INFO):
    from tqdm import tqdm

    def is_console_handler(handler):
        return isinstance(handler, logging.StreamHandler) and handler.stream in {sys.stdout, sys.stderr}

    class TqdmLoggingHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:  # noqa pylint: disable=bare-except
                self.handleError(record)

    logging.basicConfig(stream=sys.stdout, level=level)
    handler = TqdmLoggingHandler(sys.stdout)
    try:
        import colorlog
        formatter = colorlog.LevelFormatter(fmt={
            'DEBUG': '%(log_color)sdebug: %(message)s (%(module)s:%(lineno)d)%(reset)s',
            'INFO': '%(log_color)sinfo%(reset)s: %(message)s',
            'WARNING': '%(log_color)swarning%(reset)s: %(message)s (%(module)s:%(lineno)d)',
            'ERROR': '%(log_color)serror%(reset)s: %(message)s (%(module)s:%(lineno)d)',
            'CRITICAL': '%(log_color)scritical: %(message)s (%(module)s:%(lineno)d)%(reset)s',
        }, log_colors={
            'DEBUG': 'white',
            'INFO': 'bold_green',
            'WARNING': 'bold_yellow',
            'ERROR': 'bold_red',
            'CRITICAL': 'bold_red',
        })
        handler.setFormatter(formatter)
    except(ModuleNotFoundError):
        # We do not require colorlog to be present
        pass
    logging._acquireLock()
    orig_handlers = logging.root.handlers
    try:
        logging.root.handlers = [x for x in orig_handlers if not is_console_handler(x)] + [handler]
    except Exception:
        logging.root.handlers = orig_handlers
    finally:
        logging._releaseLock()


def download_file(dest_path, source_url, api_key=None):
    from tqdm import tqdm
    auth = None if api_key is None else ('api', api_key)
    response = requests.get(source_url, auth=auth, stream=True, timeout=5)
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


def pull_model(name, overwrite=False):
    if name is None or not name.startswith('wandb:'):
        return name

    # We will load the model from the wandb repository
    import wandb
    name = name[len('wandb:'):]

    def _pull():
        api = wandb.Api()
        root = os.environ.get('MODELS_PATH', os.path.expanduser('~/models'))
        root = os.path.join(root, 'augpt')
        os.makedirs(root, exist_ok=True)
        base_path = f"{os.environ.get('WANDB_ENTITY', api.default_entity)}/{os.environ.get('WANDB_PROJECT', 'dstc9')}"
        model_path = os.path.join(root, name)
        os.makedirs(model_path, exist_ok=True)

        # Try to load artifact
        if ':' in name:
            artifact = api.artifact(f'{base_path}/{name}')
            if wandb.run is not None:
                print(artifact)
                artifact = wandb.run.use_artifact(artifact, type='model')
            # return artifact.download(model_path)
            # Always clean download
            return artifact.download(model_path)

        # Load weights from runs
        run = api.run(f"{base_path}/{name}")
        for f in run.files():
            path = os.path.join(model_path, f.name)
            if not os.path.exists(path):
                download_file(path, f.url, api.api_key)
        return model_path

    if torch.distributed.is_initialized():
        is_master = torch.distributed.get_rank() == 0
        if is_master:
            result = _pull()
            torch.distributed.barrier()
            return result
        else:
            torch.distributed.barrier()
            return _pull()
    else:
        return _pull()


class AutoDatabase:
    @staticmethod
    def load(pretrained_model_name_or_path: str, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if os.path.isdir(pretrained_model_name_or_path):
            database_file = os.path.join(pretrained_model_name_or_path, 'database.zip')
        elif os.path.isfile(pretrained_model_name_or_path) or \
                transformers.file_utils.is_remote_url(pretrained_model_name_or_path):
            database_file = pretrained_model_name_or_path
        else:
            database_file = transformers.file_utils.hf_bucket_url(
                pretrained_model_name_or_path, filename='database.zip'
            )

        try:
            # Load from URL or cache if already cached
            resolved_database_file = transformers.file_utils.cached_path(
                database_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
            )

            # Load config dict
            if resolved_database_file is None:
                raise EnvironmentError

            with zipfile.ZipFile(resolved_database_file) as zipf:
                def _build_database():
                    module = types.ModuleType('database')
                    exec(zipf.read('database.py').decode('utf-8'), module.__dict__)
                    return module.Database(zipf, **kwargs)

                database = _build_database()

        except EnvironmentError:
            msg = (
                f"Can't load config for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"  # noqa: E501
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a database.zip file\n\n"  # noqa: E501
            )
            raise EnvironmentError(msg)

        if resolved_database_file == database_file:
            logger.info("loading database {}".format(database_file))
        else:
            logger.info("loading database {} from cache at {}".format(database_file, resolved_database_file))  # noqa: E501

        return database


class AutoLexicalizer:
    @staticmethod
    def load(pretrained_model_name_or_path: str, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if os.path.isdir(pretrained_model_name_or_path):
            lexicalizer_file = os.path.join(pretrained_model_name_or_path, 'lexicalizer.zip')
        elif os.path.isfile(pretrained_model_name_or_path) or \
                transformers.file_utils.is_remote_url(pretrained_model_name_or_path):
            lexicalizer_file = pretrained_model_name_or_path
        else:
            lexicalizer_file = transformers.file_utils.hf_bucket_url(
                pretrained_model_name_or_path, filename='lexicalizer.zip'
            )

        try:
            # Load from URL or cache if already cached
            resolved_lexicalizer_file = transformers.file_utils.cached_path(
                lexicalizer_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
            )

            # Load config dict
            if resolved_lexicalizer_file is None:
                raise EnvironmentError

            with zipfile.ZipFile(resolved_lexicalizer_file) as zipf:
                def _build_lexicalizer():
                    module = types.ModuleType('lexicalizer')
                    exec(zipf.read('lexicalizer.py').decode('utf-8'), module.__dict__)
                    return module.Lexicalizer(zipf, **kwargs)

                lexicalizer = _build_lexicalizer()

        except EnvironmentError:
            msg = (
                f"Can't load config for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"  # noqa: E501
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a lexicalizer.zip file\n\n"  # noqa: E501
            )
            raise EnvironmentError(msg)

        if resolved_lexicalizer_file == lexicalizer_file:
            logger.info("loading lexicalizer {}".format(lexicalizer_file))
        else:
            logger.info("loading lexicalizer {} from cache at {}".format(lexicalizer_file, resolved_lexicalizer_file))  # noqa: E501

        return lexicalizer
