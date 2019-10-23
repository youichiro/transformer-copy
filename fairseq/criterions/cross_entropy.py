# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from . import FairseqCriterion, register_criterion


@register_criterion('cross_entropy')
class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, distribution_loss, label_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        copy_alpha = net_output[1]['copy_alpha'].mean().item() if net_output[1]['copy_alpha'] is not None else -1
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'distribution_loss': utils.item(distribution_loss.data) if reduce else distribution_loss.data,
            'label_loss': utils.item(label_loss.data) if reduce else label_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'copy_alpha': copy_alpha,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        if self.args.positive_label_weight != 1 \
            and sample is not None and sample.get('target_label', None) is not None:
            return self.compute_weighted_loss(model, net_output, sample, reduce=True)
        lprobs = model.get_normalized_probs(net_output, log_probs=True, sample=sample)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)
        return loss, loss, loss

    def compute_weighted_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True, sample=sample)  # 8 x 31 x 40031
        lprobs = lprobs.view(-1, lprobs.size(-1))  # 248 x 40031
        target = model.get_targets(sample, net_output).view(-1)

        target_label = sample['target_label'].view(-1).byte()
        neg_target = target.new_tensor(target).masked_fill_(target_label, self.padding_idx)
        pos_target = target.new_tensor(target).masked_fill_(1-target_label, self.padding_idx)

        neg_loss = F.nll_loss(lprobs, neg_target, size_average=False, ignore_index=self.padding_idx,
                              reduce=reduce)
        pos_loss = F.nll_loss(lprobs, pos_target, size_average=False, ignore_index=self.padding_idx,
                              reduce=reduce)

        #loss = neg_loss + self.args.positive_label_weight * pos_loss
        distribution_loss = (1/self.args.positive_label_weight) * neg_loss + pos_loss
        loss = distribution_loss

        """token-level multi-task learning"""
        if self.args.token_labeling_loss_weight > 0:
            assert 0 < self.args.token_labeling_loss_weight <= 1.0

            # get encoder output hidden states
            encoder_out = net_output[1]['encoder_out']  # 31 x 8 x 512
            encoder_out = encoder_out.transpose(0, 1)  # 8 x 31 x 512

            # Map to 2-dim
            project_enc_out_dim = nn.Linear(encoder_out.size(-1), 2, bias=True).cuda()
            encoder_out = project_enc_out_dim(encoder_out)  # 8 x 31 x 2
            encoder_lprobs = F.log_softmax(encoder_out)  # 8 x 31 x 2
            encoder_lprobs = encoder_lprobs.view(-1, 2)  # 248 x 2

            # calculate token labeling loss with positive weight
            source_label = sample['source_label'].view(-1).long()  # 248
            weight = torch.tensor([1., self.args.token_labeling_positive_label_weight]).cuda()
            label_loss = F.nll_loss(encoder_lprobs, source_label, weight=weight, size_average=False, reduce=reduce)

            # combine encoding loss with token labeling loss
            label_weight = self.args.token_labeling_loss_weight
            loss = (1 - label_weight) * distribution_loss + label_weight * label_loss

            return loss, (1 - label_weight) * distribution_loss, label_weight * label_loss

        else:
            return loss, loss, loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        distribution_loss_sum = sum(log.get('distribution_loss', 0) for log in logging_outputs)
        label_loss_sum = sum(log.get('label_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        copy_alpha = sum(log.get('copy_alpha', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'distribution_loss': distribution_loss_sum / sample_size / math.log(2),
            'label_loss': label_loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'copy_alpha': copy_alpha,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
