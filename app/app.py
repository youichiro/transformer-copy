# -*- coding: utf-8 -*-
import os
import sys

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_bootstrap import Bootstrap

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from gec_model import GECModel

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
bootstrap = Bootstrap(app)

MODEL_PATH = '../out/models/checkpoint_last.pt'
DATA_RAW = '../out/data_raw/naist_clean_char'
OPTION_FILE = 'model_options.txt'

gec_model = GECModel(MODEL_PATH, DATA_RAW, OPTION_FILE)


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('checker.html')


@app.route('/api', methods=['GET'])
def api():
    text = request.args.get('input_text')
    if not text:
        return ''
    res = gec_model.generate(text)
    print(res)
    return jsonify({'res': res})


if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost', port=5002)
