import dataclasses
import logging
from dataclasses import dataclass
from transformers import GPT2Tokenizer as AuGPTTokenizer  # noqa
from torch import nn
import transformers
from torch.nn import functional as F
import torch
import data


EOB_TK = '<|eob|>'
EOKB_TK = '<|eokb|>'
EOT_TK = '<|endoftext|>'
SPECIAL_TOKENS = [EOB_TK, EOKB_TK]
logger = logging.getLogger()


def add_custom_tokens(tokenizer, model):
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


# TODO: new transformers version
# @dataclass
# class AuGPTModelOutput(transformers.ModelOutput):
#     """
#     AuGPTModelOutput with consistency detection, split loss between belief state and response
#
#     Args:
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided):
#             Language modeling loss.
#         mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`mc_labels` is provided):
#             Multiple choice classification loss.
#         logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
#             Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#         mc_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
#             Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
#         past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
#             List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
#             :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
#
#             Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
#             :obj:`past_key_values` input) to speed up sequential decoding.
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.
#
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
#
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#     """
#
#     loss: Optional[torch.FloatTensor] = None
#     mc_loss: Optional[torch.FloatTensor] = None
#     logits: torch.FloatTensor = None
#     mc_logits: torch.FloatTensor = None
#     past_key_values: Optional[List[torch.FloatTensor]] = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None


class AuGPTConfig(transformers.GPT2Config):
    def __init__(self,
                 summary_label_smoothing=0.1,
                 response_loss='unlikelihood',
                 **kwargs):
        super().__init__(**kwargs)
        self.summary_label_smoothing = summary_label_smoothing
        self.response_loss = response_loss


class CandidatePenaltyCrossEntropyCriterion(nn.Module):
    def __init__(self, rank_alpha=1.0, ignore_index=-100, checkpoint=False):
        super().__init__()
        self.rank_alpha = rank_alpha
        self.ignore_index = ignore_index

    @torch.no_grad()
    def _negative_targets(self, lprobs, target):
        # E.g. DABCC | D | EFFGD => {A,B,C} are negative targets.
        # Make 'the triangle'.
        # TODO: cuda does not have short kernel for scatter, alternative?
        ntarget = target.add(1).masked_fill_(target == self.ignore_index, 0)
        ctx_cands = ntarget.unsqueeze(1).expand(ntarget.size(0), ntarget.size(1), ntarget.size(1))
        ctx_cands = ctx_cands.tril(-1)

        # Don't include the target for that timestep as a negative target.
        ctx_cands = ctx_cands.masked_fill(ctx_cands == ntarget.unsqueeze(2), 0)
        del ntarget

        negative_targets = lprobs.new_zeros(lprobs.shape[:2] + (lprobs.size(-1) + 1,))
        negative_targets = negative_targets.scatter_(2, ctx_cands, 1)
        return negative_targets[..., 1:]

    def forward(self, logits, target, return_ce=False):
        """Loss which helps model not to predict already appeared tokens.
        Args:
            logits (tensor):
                Torch tensor of shape (bs, seq_len, vocab_size), output language
                model scores.
            target (tensor):
                Torch tensor of shape (bs, seq_len), language model target (model
                input tokens itself).
        Returns:
            Unlikelihood candidates loss-value.
        Notes:
            This loss is based on penalizing of the previous context tokens.
            Original paper - Welleck et al. https://arxiv.org/pdf/1908.04319.pdf.
        """
        lprobs = F.log_softmax(logits, -1)
        del logits
        negative_targets = self._negative_targets(lprobs, target)

        # -- mle loss
        mle_loss = F.nll_loss(
            lprobs.view(-1, lprobs.size(-1)),
            target.view(-1),
            ignore_index=self.ignore_index,
            reduction='none',
        )
        mle_loss = mle_loss.sum()

        # -- custom loss
        # Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))
        # - compute loss
        one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)
        custom_loss = -torch.log(one_minus_probs) * negative_targets
        custom_loss = custom_loss.sum()

        # Scale loss
        loss = mle_loss + self.rank_alpha * custom_loss
        weight = (target != -100).sum()
        loss /= weight
        if return_ce:
            return loss, mle_loss / weight
        return loss


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(-1)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = torch.sum(-true_dist * pred * (target != -100).unsqueeze(-1))
        return loss / (target != -100).sum()


class LabelSmoothingBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target, weight=None):
        smoothed_labels = target.mul(1 - 2 * self.smoothing).add_(self.smoothing)
        return torch.nn.functional.binary_cross_entropy_with_logits(input, smoothed_labels, weight)


class AuGPTModel(transformers.GPT2PreTrainedModel):
    authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias",
                               r"lm\_head\.weight", r"binary\_head\.\w+"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = transformers.GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.consistency_head = nn.Linear(config.n_embd, 1)
        self.auxiliary_dropout = nn.Dropout(config.summary_first_dropout)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self,
                input_ids=None,
                past=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                consistency_token_ids=None,
                consistency_labels=None,
                user_intent_token_ids=None,
                user_intent_labels=None,
                user_intent_mask=None,
                belief_labels=None,
                system_action_token_ids=None,
                system_action_labels=None,
                system_action_mask=None,
                response_labels=None,
                binary_labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                **kwargs
                ):

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        def gather_auxiliary_features(token_ids):
            if token_ids is None:
                token_ids = torch.full_like(hidden_states[..., :1, :],
                                            hidden_states.shape[-2]-1, dtype=torch.long,)
            else:
                token_ids = token_ids.unsqueeze(-1).unsqueeze(-1)
                token_ids = token_ids.expand(
                    (-1,) * (token_ids.dim() - 1) + (hidden_states.size(-1),))

            # shape of binary_token_ids: (bsz, XX, 1, hidden_size)
            # where XX are optional leading dim of hidden_states
            # shape of binary_logits (bsz, XX, hidden_size)
            logits = hidden_states.gather(-2, token_ids).squeeze(-2)
            logits = self.auxiliary_dropout(logits)
            return logits

        consistency_logits = self.consistency_head(gather_auxiliary_features(consistency_token_ids)).squeeze(-1)
        consistency_loss = None
        if consistency_labels is not None:
            # Auxiliary tasks
            aux_criterion = LabelSmoothingBCEWithLogitsLoss(self.config.summary_label_smoothing)
            consistency_loss = aux_criterion(consistency_logits, consistency_labels)

        belief_loss, response_loss = None, None
        if belief_labels is not None:
            assert response_labels is not None

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_belief_labels = belief_labels[..., 1:].contiguous()
            shift_response_labels = response_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            belief_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_belief_labels.view(-1))

            if self.config.response_loss == 'ce':
                response_ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_response_labels.view(-1))
                response_loss = response_ce
            elif self.config.response_loss == 'unlikelihood':
                candidate_ce_fct = CandidatePenaltyCrossEntropyCriterion()
                response_loss, response_ce = candidate_ce_fct(
                    shift_logits,
                    shift_response_labels, return_ce=True)
            else:
                raise ValueError(f'Response loss {self.config.response_loss} is not supported')

        output = (lm_logits, consistency_logits,) + transformer_outputs[1:]
        if consistency_loss is not None:
            output = (consistency_loss,) + output
        return ((belief_loss, response_loss, response_ce) + output) if belief_loss is not None else output


@dataclass
class ModelPredictor:
    model: transformers.PreTrainedModel = None
    tokenizer: transformers.PreTrainedTokenizer = None
    max_belief_length: int = 100
    max_response_length: int = 200
    device: torch.device = torch.device('cpu')

    @staticmethod
    def from_pretrained(model_name):
        config = transformers.GPT2Config.from_pretrained(model_name)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            model_name)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name, config=config)
        if model_name == 'gpt2':
            tokenizer, model = add_custom_tokens(tokenizer, model)
        tokenizer.pad_token = tokenizer.eos_token
        predictor = ModelPredictor(model, tokenizer)
        return predictor

    def predict_belief(self, contexts):
        insert_labels = data.utils.InsertLabelsTransformation()
        tokenize = data.utils.TokenizerTransformation(
            self.tokenizer,
            max_context_length=self.model.config.n_ctx - self.max_belief_length - 1)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(['<|eob|>'])[0]
        beliefs = []
        # TODO: batch generation
        for ctx in contexts:
            sample = insert_labels((ctx, None, None, None, 1))
            sample = tokenize.get_tokens(sample)[0]
            sample = torch.tensor(sample, dtype=torch.int64).to(self.device)
            sample = sample.view(1, *sample.shape)  # (batch, time)
            greedy_output = self.model.generate(
                input_ids=sample,
                max_length=sample.size(1) + self.max_belief_length,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
                do_sample=False)
            # https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py

            prediction = greedy_output[0]
            offset = len(sample[0])
            prediction = prediction[:offset + (prediction[offset:] != eos_token_id).int().sum()]
            prediction = self.tokenizer.decode(prediction, skip_special_tokens=False,
                                               clean_up_tokenization_spaces=True)
            prefix = self.tokenizer.decode(sample[0], clean_up_tokenization_spaces=True) +\
                '=> ' + insert_labels.belief_label
            prediction = prediction[len(prefix):]
            beliefs.append(prediction)
        return beliefs

    def predict_response(self, contexts, beliefs, dbs):
        insert_labels = data.utils.InsertLabelsTransformation()
        tokenize = data.utils.TokenizerTransformation(
            self.tokenizer,
            max_context_length=self.model.config.n_ctx - self.max_response_length)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(['<|endoftext|>'])[0]
        responses = []
        # TODO: batch generation
        for context, belief, db in zip(contexts, beliefs, dbs):
            sample = insert_labels((context, belief, db, None))
            sample = tokenize.get_tokens(sample)[0]
            sample = torch.tensor(sample, dtype=torch.int64).to(self.device)
            sample = sample.view(1, *sample.shape)  # (batch, time)
            greedy_output = self.model.generate(
                input_ids=sample,
                max_length=sample.size(1) + self.max_response_length,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
                do_sample=True,
                top_k=0)
            # https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py
            prediction = greedy_output[0]
            offset = len(sample[0])
            prediction = prediction[:offset + (prediction[offset:] != eos_token_id).int().sum()]
            prediction = self.tokenizer.decode(prediction, skip_special_tokens=False,
                                               clean_up_tokenization_spaces=True)
            prediction = prediction[len(self.tokenizer.decode(sample[0], clean_up_tokenization_spaces=True)):]
            prediction = prediction.lstrip()
            responses.append(prediction)
        return responses

    def to(self, device):
        return dataclasses.replace(self, device=device, model=self.model.to(device))
