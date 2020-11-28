import dataclasses
import tempfile
import pytest


@pytest.fixture()
def logger():
    import logging
    return logging.getLogger()


def patch_dataset(dataset):
    return dataclasses.replace(dataset, items=[dataset[i] for i in range(2)], transform=lambda x: x)


@pytest.mark.train
def test_train(logger, monkeypatch):
    import wandb
    import data
    from train import Trainer, parse_args
    import data.evaluation.multiwoz
    from unittest.mock import MagicMock
    import tensorboardX

    with tempfile.TemporaryDirectory() as d:
        with monkeypatch.context() as m:
            m.setattr(wandb, 'run', MagicMock())
            m.setattr(wandb, 'log', MagicMock())
            m.setattr(wandb, 'config', MagicMock())
            m.setattr(wandb, 'tensorboard', MagicMock())
            m.setattr(tensorboardX, 'SummaryWriter', MagicMock())
            wandb.run.dir = d

            old_load_dataset = data.load_dataset
            m.setattr(data, 'load_dataset', lambda name, **kwargs: patch_dataset(old_load_dataset(name, **kwargs)))
            m.setattr("sys.argv", ["train.py"])
            args = parse_args()
            # args.fp16 = True
            args.batch_size = 2
            args.epochs = 1
            args.logging_steps = 1
            args.validation_steps = 1
            args.evaluation_dialogs = 1
            trainer = Trainer(args, logger)

            # Patch prediction
            old_prediction = trainer._run_prediction
            def mock_prediction(*args, **kwargs):  # noqa:E306
                m.setattr(trainer.dev_predictor.pipeline.predictor, 'max_belief_length', 2)
                m.setattr(trainer.dev_predictor.pipeline.predictor, 'max_response_length', 2)
                return old_prediction(*args, **kwargs)
            m.setattr(trainer, '_run_prediction', mock_prediction)

            # Patch evaluation
            # old_evaluation = trainer._run_evaluation
            # def mock_evaluation(*args, **kwargs):  # noqa:E306
            #     # To speed up the generation
            #     old_generate = trainer.dev_predictor.predictor.predictor.model.generate
            #     cached_result = dict()
            #     def mock_generate(input_ids, *args, **kwargs):  # noqa:E306
            #         nonlocal cached_result
            #         assert 'eos_token_id' in kwargs
            #         eos_token_id = kwargs.get('eos_token_id')
            #         if eos_token_id not in cached_result:
            #             r = old_generate(*args, input_ids=input_ids, **kwargs)[:, len(input_ids[0]):]
            #             eos = torch.zeros_like(r).fill_(kwargs.get('eos_token_id'))
            #             cached_result[eos_token_id] = torch.cat([torch.zeros_like(r), r, eos], 1)
            #         return torch.cat([input_ids, cached_result[eos_token_id]], 1)

            #     m.setattr(trainer.dev_predictor.predictor.predictor.model, 'generate', mock_generate)
            #     m.setattr(trainer.dev_predictor.predictor.predictor, 'max_belief_length', 2)
            #     m.setattr(trainer.dev_predictor.predictor.predictor, 'max_response_length', 2)
            #     return old_evaluation(*args, **kwargs)
            # m.setattr(trainer, '_run_evaluation', mock_evaluation)

            # Patch publish artifact
            trainer._publish_artifact = lambda: None

            # Run train
            trainer.train()
