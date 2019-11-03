import sys
import numpy as np
from collections import defaultdict
from pprint import pprint

if len(sys.argv) != 4:
    print('Usage: <origin_nbest_filename> <rescore_nbest_filename> <beam>')
    exit(-1)

origin_nbest_filename = sys.argv[1]
rescore_nbest_filename = sys.argv[2]
beam = int(sys.argv[3])

i = 0
origin_order = 0
hypo_dic = {}
for line in open(origin_nbest_filename).readlines():
    line = line.replace('\n', '')
    if line[0] == 'H':
        sent_id = int(line.split('\t')[0][2:])
        score = float(line.split('\t')[1])
        hypo = line.split('\t')[2]
        origin_order += 1
        i += 1
        if i > 12:
            i = 1
        uniq_id = sent_id * beam + i
        hypo_dic[origin_order] = {
            'sent_id': sent_id,
            'uniq_id': uniq_id,
            'sub_id': i,
            'hypo': hypo,
            'first_score': score,
            'second_score': 0.0,
        }


for line in open(rescore_nbest_filename).readlines():
    line = line.replace('\n', '')
    if line[0] == 'H':
        sent_id = int(line.split('\t')[0][2:])
        score = float(line.split('\t')[1])
        hypo = line.split('\t')[2]
        hypo_dic[sent_id+1]['second_score'] = score


hypos = sorted(hypo_dic.values(), key=lambda x:x['uniq_id'])


scores = []
for hypo in hypos:
    new_score = (hypo['first_score'] + hypo['second_score']) / 2
    scores.append([new_score, hypo['hypo']])
    if hypo['sub_id'] == beam:
        # output best hypo
        best = max(scores, key=lambda x:x[1])
        print(best[1])
        scores = []

