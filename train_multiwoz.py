#!/bin/env python
import os
import shutil
import logging
import torch
import wandb
import nltk

from train import Trainer, parse_args, setup_logging  # noqa:E402
from generate import generate_predictions  # noqa:E402
from data.evaluation.multiwoz import MultiWozEvaluator, compute_bleu_remove_reference  # noqa: E402
from data import load_dataset  # noqa:E402
import data.evaluation.convlab  # noqa:E402
from evaluation_utils import compute_delexicalized_bleu  # noqa:E402
from data.utils import wrap_dataset_with_cache  # noqa:E402


class MultiWozTrainer(Trainer):
    @torch.no_grad()
    def run_convlab_evaluation(self):
        self.logger.info('running convlab evaluation')
        self.model.eval()
        analyzer = data.evaluation.multiwoz.convlab.ConvLabAnalyzer()
        result = analyzer(self.prediction_pipeline, num_dialogs=self.args.evaluation_dialogs)

        # Move the results from evaluator script to wandb
        shutil.move('results', os.path.join(wandb.run.dir, 'results'))

        # Fix synchronize metrics when using different run for other metrics
        online_run = wandb.Api().run(self.wandb_runid)
        evaluation_keys = {'test_inform', 'test_success', 'test_bleu', 'test_delex_bleu'}
        summary = {k: v for k, v in online_run.summary.items() if k in evaluation_keys}
        wandb.run.summary.update(summary)

        # Update results from the analyzer
        wandb.run.summary.update(result)

    @torch.no_grad()
    def run_test_evaluation(self, wandb_runid=None):
        self.logger.info('running multiwoz evaluation')
        self.logger.info('generating responses')
        self.model.eval()
        dataset = load_dataset('multiwoz-2.1-test', use_goal=True)
        dataset = wrap_dataset_with_cache(dataset)
        predictions = generate_predictions(self.prediction_pipeline, dataset, 'test-predictions.txt')
        evaluator = MultiWozEvaluator(dataset, is_multiwoz_eval=True, logger=self.logger)
        success, matches, domain_results = evaluator.evaluate(predictions.beliefs, predictions.delex_responses, progressbar=True)
        self.logger.info('evaluation finished')
        self.logger.info('computing bleu')
        bleu = compute_bleu_remove_reference(predictions.responses, predictions.gold_responses)
        delex_bleu = compute_delexicalized_bleu(predictions.delex_responses, predictions.gold_delex_responses)
        self.logger.info(f'test bleu: {bleu:.4f}')
        self.logger.info(f'delex test bleu: {delex_bleu:.4f}')

        # We will use external run to run in a separate process
        if self.is_master():
            run = wandb.run
            shutil.copy('test-predictions.txt', run.dir)
        else:
            api = wandb.Api()
            run = api.run(self.wandb_runid)
            run.upload_file('test-predictions.txt')
        run.summary.update(dict(
            test_inform=matches,
            test_success=success,
            test_bleu=bleu,
            test_delex_bleu=delex_bleu
        ))

    def run_evaluation(self):
        if self.is_master():
            self.run_convlab_evaluation()
        if self.args.local_rank == -1 or torch.distributed.get_rank() == 1 or torch.distributed.get_world_size() == 1:
            self.run_test_evaluation()
        torch.distributed.barrier()

        if self.is_master() and self.args.local_rank != -1 and torch.distributed.get_world_size() > 1:
            # Fix synchronize metrics when using different run for other metrics
            online_run = wandb.Api().run(self.wandb_runid)
            evaluation_keys = {'test_inform', 'test_success', 'test_bleu', 'test_delex_bleu'}
            summary = {k: v for k, v in online_run.summary.items() if k in evaluation_keys}
            wandb.run.summary.update(summary)


if __name__ == '__main__':
    # Update punkt
    nltk.download('punkt')

    args = parse_args()
    setup_logging()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if args.local_rank != -1:
        torch.distributed.init_process_group('nccl', init_method='env://')
        assert args.device == torch.device('cuda'), "CUDA must be available in distributed training"
        torch.cuda.set_device(args.local_rank)
        logger.info('initialized distributed training with {} nodes, local-rank: {}'.format(
            torch.distributed.get_world_size(), args.local_rank))

    # Start training
    trainer = MultiWozTrainer(args, logger)
    trainer.train()
    if args.local_rank != -1:
        torch.distributed.destroy_process_group()
