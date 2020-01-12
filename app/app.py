# -*- coding: utf-8 -*-
import os
import sys
import configparser

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
    res = gec_model.generate(text)
    return jsonify({'res': res})


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5003)
