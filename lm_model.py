import numpy as np
import kenlm
import torch
from fairseq import options, progress_bar, tasks, utils
from fairseq.sequence_scorer import SequenceScorer
from fairseq.utils import import_user_module


class KenLM:
    def __init__(self, data):
        print('| loading KenLM data')
        self.lm = kenlm.Model(data)

    def calc(self, sentence):
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)
        score = self.lm.score(sentence, bos=True, eos=True)
        return score


class TransformerLM:
    def __init__(self, model_path, dict_path):
        parser = options.get_eval_lm_parser()
        parsed_args = options.parse_args_and_arch(
            parser, input_args=[None], parse_known=True
        )[0]
        parsed_args.path = model_path
        parsed_args.dict = dict_path
        parsed_args.max_sentence = 1
        parsed_args.gen_subset = 'test'
        parsed_args.raw_text = True
        parsed_args.no_progress_bar = True
        import_user_module(parsed_args)
        print(parsed_args)

        task = tasks.setup_task(parsed_args)
        print('| loading model(s) from {}'.format(parsed_args.path))
        models, args = utils.load_ensemble_for_inference(
            parsed_args.path.split(':'), task, model_arg_overrides=eval(parsed_args.model_overrides),
        )
        for arg in vars(parsed_args).keys():
            if arg not in {'self_target', 'future_target', 'past_target', 'tokens_per_sample', 'output_size_dictionary'}:
                setattr(args, arg, getattr(parsed_args, arg))
        task = tasks.setup_task(args)

        self.use_cuda = torch.cuda.is_available() and not parsed_args.cpu
        for model in models:
            model.make_generation_fast_()
            if self.use_cuda:
                model.cuda()
        assert len(models) > 0

        scorer = SequenceScorer(task.target_dictionary)

        self.args = args
        self.task = task
        self.models = models
        self.scorer = scorer


    def make_itr(self, sentence):
        self.task.load_sentence(self.args.gen_subset, sentence)
        itr = self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.gen_subset),
            max_tokens=self.args.max_tokens or 36000,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(*[
                model.max_positions() for model in self.models
            ]),
            ignore_invalid_inputs=True,
            num_shards=self.args.num_shards,
            shard_id=self.args.shard_id,
            num_workers=self.args.num_workers,
        ).next_epoch_itr(shuffle=False)

        return itr


    def calc(self, sentence):
        score_sum = 0.
        count = 0
        word_stats = dict()

        itr = self.make_itr(sentence)
        with progress_bar.build_progress_bar(self.args, itr) as t:
            for sample in t:
                sample = utils.move_to_cuda(sample) if self.use_cuda else sample
                if 'net_input' not in sample:
                    continue
                hypos = self.scorer.generate(self.models, sample)

                for hypos_i in hypos:
                    hypo = hypos_i[0]
                    pos_scores = hypo['positional_scores']

                    skipped_toks = 0
                    inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                    if inf_scores.any():
                        print('| Skipping tokens with inf scores:',
                            self.task.target_dictionary.string(hypo['tokens'][inf_scores.nonzero()]))
                        pos_scores = pos_scores[(~inf_scores).nonzero()]
                    score_sum += pos_scores.sum().cpu()
                    count += pos_scores.numel() - skipped_toks

        avg_nll_loss = score_sum / count
        return float(avg_nll_loss)


if __name__ == "__main__":
    model_path = 'out/models_lm/checkpoint_last.pt'
    dict_path = 'out/models_lm/dict.txt'
    lm = TransformerLM(model_path, dict_path)

    sentences = [
        '今 日 は 車 を 買 う 。',
        '今 日 は 車 に 買 う 。',
        '今 日 は 車 で 買 う 。',
    ]
    for s in sentences:
        score = lm.calc(s)
        print(s, score)
