# -*- coding: utf-8 -*-
import re
import emoji
import regex
import neologdn


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


def sentence_split(text):
    if text[-1] != '。':
        text = text + '。'
    text = text_clean(text)
    text = ' '.join(text)  # 文字分割
    text = text.replace('。', '。\n')
    lines = re.split('[\t\n]', text)  # 文分割
    lines = [line.strip() for line in lines if line]
    return lines
