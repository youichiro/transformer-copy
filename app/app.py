# -*- coding: utf-8 -*-
import os
import sys
import configparser
from pprint import pprint
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_bootstrap import Bootstrap
from align import get_aligned_edits
from utils import text_clean, sentence_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gec_model import GECModel

mode = 'docker'  # ('local', 'docker')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
bootstrap = Bootstrap(app)

ini = configparser.ConfigParser()
ini.read('./config.ini', 'UTF-8')
model_path = ini.get(mode, 'model_path')
second_model_path = ini.get(mode, 'second_model_path')
data_raw = ini.get(mode, 'data_raw')
option_file = ini.get(mode, 'option_file')
url_prefix = ini.get(mode, 'url_prefix')

base_model = GECModel(model_path, data_raw, option_file)
second_model = GECModel(second_model_path, data_raw, option_file)


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('checker.html', prefix=url_prefix)


@app.route('/api', methods=['GET'])
def api():
    text = request.args.get('input_text')
    if not text:
        return ''
    lines = sentence_split(text)
    results = []
    for line in lines:
        if not line or line == ' ' or line.replace(' ', '') == '。':
            continue
        res = generate(line, times=2, mode=mode)[0]
        res['src_str'] = line
        res['src_raw'] = line.replace(' ', '')
        res['edits'] = get_aligned_edits(res['src_raw'], res['best_hypo']['hypo_raw'])
        results.append(res)
    return jsonify({'res': results})


def generate_single(sentence, model, times, mode):
    scores = []
    for _ in range(times):
        res = model.generate(sentence)
        hypos = res[0]['hypos']
        for hypo in hypos:
            scores.append([hypo['score'], hypo['hypo_raw']])
    scores = sorted(scores, reverse=True)
    if mode == 'local':
        pprint(scores)
    best_hypo_score, best_hypo_raw = scores[0]
    best_hypo_str = ' '.join(best_hypo_raw)
    # update best_hypo
    res[0]['best_hypo']['hypo_raw'] = best_hypo_raw
    res[0]['best_hypo']['score'] = best_hypo_score
    res[0]['best_hypo']['hypo_str'] = best_hypo_str
    return res, best_hypo_str


def generate(sentence, times=2, mode='local'):
    res, best = generate_single(sentence, base_model, times, mode)
    res, best = generate_single(best, base_model, times, mode)
    res, best = generate_single(best, second_model, times, mode)
    return res


if __name__ == '__main__':
    app.debug = False
    if mode == 'local':
        app.run(host='localhost', port=5003)
    else:
        app.run(host='0.0.0.0', port=5003)
