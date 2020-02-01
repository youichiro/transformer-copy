# -*- coding: utf-8 -*-
import os
import re
import argparse
import itertools
from collections import namedtuple
from pprint import pprint

import emoji
import regex
import neologdn
from tqdm import tqdm
import torch
from fairseq import options, tasks, utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.utils import import_user_module

Batch = namedtuple('Batch', 'ids src_tokens src_lengths, src_strs')


def make_batches(lines, args, task, max_positions):
    tokens = [
        task.source_dictionary.encode_line(
            src_str, add_if_not_exist=False, copy_ext_dict=args.copy_ext_dict).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
            src_strs=[lines[i] for i in batch['id']],
        )


class GECModel:
    def __init__(self, model_path, data_raw, option_file, lm=None, lm_weight=0.0, print_hypos=False, reverse=False):
        input_args = open(option_file).readlines()
        input_args = [['--' + arg.split('=')[0], arg.split('=')[1].replace("'", '').strip()]
                     for arg in input_args]
        input_args = list(itertools.chain.from_iterable(input_args))

        parser = options.get_generation_parser(interactive=True)
        args = options.parse_args_and_arch(parser, input_args=input_args, parse_known=True)[0]
        args.data = [data_raw]
        args.path = model_path
        import_user_module(args)
        self.use_cuda = torch.cuda.is_available() and not args.cpu

        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1

        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        print(args)

        task = tasks.setup_task(args)
        print('| loading model(s) from {}'.format(args.path))
        models, _model_args = utils.load_ensemble_for_inference(
            args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
        )
        args.copy_ext_dict = getattr(_model_args, "copy_attention", False)

        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if self.use_cuda:
                model.cuda()

        generator = task.build_generator(args)

        align_dict = utils.load_align_dict(args.replace_unk)
        if align_dict is None and args.copy_ext_dict:
            align_dict = {}

        max_positions = utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        )

        self.args = args
        self.task = task
        self.max_positions = max_positions
        self.generator = generator
        self.models = models
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.align_dict = align_dict
        self.print_hypos = print_hypos
        self.reverse = reverse
        # LM
        self.lm = lm
        self.lm_weight = lm_weight
        assert 0.0 <= self.lm_weight <= 1.0

        print('| finish loading')


    @staticmethod
    def text_clean(text):
        text = text.replace('\u3000', '')
        text = neologdn.normalize(text, repeat=3)
        text = ''.join(['' if c in emoji.UNICODE_EMOJI else c for c in text])
        text = re.sub(r'(\d)([,.])(\d+)', r'\1\3', text)
        text = re.sub(r'\d+', '0', text)
        text = re.sub(r'[!-/:-@[-`{-~]', r'', text)
        text = re.sub(u'[■-♯]', '', text)
        text = regex.sub(r'^(\p{Nd}+\p{Zs})(.*)$', r'\2', text)
        text = text.strip()
        text = text.replace('“', '')
        text = text.replace('…', '')
        text = text.replace('『', '「')
        text = text.replace('』', '」')
        text = text.replace('《', '「')
        text = text.replace('》', '」')
        text = text.replace('〕', '）')
        text = text.replace('〔', '（')
        text = text.replace('〈', '（')
        text = text.replace('〉', '）')
        text = text.replace('→', '')
        text = text.replace(',', '、')
        text = text.replace('，', '、')
        text = text.replace('．', '。')
        text = text.replace('.', '。')
        text = text.replace(' ', '')
        return text


    def sentence_split(self, text):
        if text[-1] != '。':
            text = text + '。'
        text = self.text_clean(text)
        text = ' '.join(text.replace(' ', ''))  # 文字分割
        text = text.replace('。', '。\n')
        lines = re.split('[\t\n]', text)  # 文分割
        lines = [line for line in lines if line]
        return lines


    @staticmethod
    def add_best_hypo(d):
        sorted_hypos = sorted(d['hypos'], key=lambda x:x['score'], reverse=True)
        best_hypo = sorted_hypos[0]
        d['best_hypo'] = best_hypo
        return d


    @staticmethod
    def get_best_hypo(d):
        assert 'best_hypo' in d.keys()
        return d['best_hypo']['hypo_str']


    def rerank_lm(self, d):
        for hypo in d['hypos']:
            score = self.lm.calc(hypo['hypo_str'])
            hypo['score'] = hypo['score'] + self.lm_weight * score
        return d


    @staticmethod
    def reverse_result(d):
        d['src_str'] = d['src_str'][::-1]
        d['src_raw'] = d['src_raw'][::-1]
        for hypo in d['hypos']:
            hypo['hypo_str'] = hypo['hypo_str'][::-1]
            hypo['hypo_raw'] = hypo['hypo_raw'][::-1]
        if 'best_hypo' in d.keys():
            d['best_hypo']['hypo_str'] = d['best_hypo']['hypo_str'][::-1]
            d['best_hypo']['hypo_raw'] = d['best_hypo']['hypo_raw'][::-1]
        return d


    def generate(self, sentence):
        if self.reverse:
            sentence = sentence[::-1]
        start_id = 0
        src_strs = []
        results = []
        res = []
        for batch in make_batches([sentence], self.args, self.task, self.max_positions):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            src_strs.extend(batch.src_strs)

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            sample = utils.move_to_cuda(sample) if self.use_cuda else sample
            translations = self.task.inference_step(self.generator, self.models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
            d = {
                'id': id,
                'src_str': src_str,
                'src_raw': src_str.replace(' ', ''),
                'hypos': []
            }

            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_strs[id],
                    alignment=hypo['alignment'].int().cpu(
                    ) if hypo['alignment'] is not None else None,
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.args.remove_bpe,
                )
                positional_scores = [round(score, 4) for score in hypo['positional_scores'].tolist()]
                alignment = list(map(lambda x: str(utils.item(x)), alignment))
                d['hypos'].append({
                    'hypo_str': hypo_str,
                    'hypo_raw': hypo_str.replace(' ', ''),
                    'score': hypo['score'],
                    # 'positional_scores': positional_scores,
                    # 'alignment': alignment if self.args.print_alignment else None,
                })

            # reranking with language model
            if self.lm:
                d = self.rerank_lm(d)

            d = self.add_best_hypo(d)
            if self.reverse:
                d = self.reverse_result(d)
            if self.print_hypos:
                pprint(d)
            res.append(d)

        return res


    def run_generate(self, sentence, n_round=1):
        for _ in range(n_round):
            res = self.generate(sentence)
            assert len(res) == 1
            sentence = self.get_best_hypo(res[0])
        return sentence


def experiment():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', required=True, help='model path')
    parser.add_argument('--data-raw', default='out/data_raw/naist_clean_char', help='data_raw')
    parser.add_argument('--option-file', default='option_files/exp.txt', help='option file')
    parser.add_argument('--test-data', default='data/naist_clean_char.src', help='test data')
    parser.add_argument('--save-dir', required=True, help='save dir')
    parser.add_argument('--save-file', default='output_gecmodel_last.char.txt', help='save file')
    parser.add_argument('--lm', default=None, choices=['kenlm', 'transformer_lm'], help='choice lm')
    parser.add_argument('--lm-data', type=str, default=None, help='lm data')
    parser.add_argument('--lm-dict', type=str, default=None, help='transformerLM dict')
    parser.add_argument('--lm-weight', type=float, default=0.0, help='lm weight[0.0, 1.0]')
    parser.add_argument('--n-round', type=int, default=1, help='n-round')
    parser.add_argument('--print-hypos', default=False, action='store_true', help='print hypos')
    parser.add_argument('--reverse', default=False, action='store_true', help='reverse')
    args = parser.parse_args()

    if args.lm == 'kenlm':
        from lm_model import KenLM
        lm = KenLM(args.lm_data)
    elif args.lm == 'transformer_lm':
        assert args.lm_dict is not None
        from lm_model import TransformerLM
        lm = TransformerLM(args.lm_data, args.lm_dict)
    else:
        lm = None

    model = GECModel(args.model_path, args.data_raw, args.option_file,
                     lm=lm, lm_weight=args.lm_weight,
                     print_hypos=args.print_hypos, reverse=args.reverse)
    data = open(args.test_data).readlines()

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.save_dir + '/' + args.save_file, 'w') as f:
        for sentence in tqdm(data):
            sentence = sentence.replace('\n', '')
            output = model.run_generate(sentence, args.n_round)
            f.write(output + '\n')



if __name__ == '__main__':
    experiment()
