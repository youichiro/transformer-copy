# -*- coding: utf-8 -*-
import os
import sys
import configparser
from pprint import pprint

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_bootstrap import Bootstrap

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gec_model import GECModel

mode = 'docker'  # ('local', 'docker')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
bootstrap = Bootstrap(app)

ini = configparser.ConfigParser()
ini.read('./config.ini', 'UTF-8')
model_path = ini.get(mode, 'model_path')
data_raw = ini.get(mode, 'data_raw')
option_file = ini.get(mode, 'option_file')
url_prefix = ini.get(mode, 'url_prefix')

gec_model = GECModel(model_path, data_raw, option_file)


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('checker.html', prefix=url_prefix)


@app.route('/api', methods=['GET'])
def api():
    text = request.args.get('input_text')
    if not text:
        return ''
    lines = gec_model.sentence_split(text)
    results = []
    for line in lines:
        res, best_hypo_raw = generate(line)
        d = res[0]
        d['best_hypo_raw'] = best_hypo_raw
        results.append(d)
    return jsonify({'res': results})


def generate(sentence, times=4):
    scores = []
    for _ in range(times):
        res = gec_model.generate(sentence)
        hypos = res[0]['hypos']
        for hypo in hypos:
            scores.append([hypo['score'], hypo['hypo_raw']])
    scores = sorted(scores, reverse=True)
    best_hypo_raw = scores[0][1]
    return res, best_hypo_raw


if __name__ == '__main__':
    app.debug = False
    if mode == 'local':
        app.run(host='localhost', port=5003)
    else:
        app.run(host='0.0.0.0', port=5003)
