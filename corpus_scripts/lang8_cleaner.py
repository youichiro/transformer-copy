# -*- coding: utf-8 -*-
import re
import regex
import MySQLdb
from pprint import pprint
import neologdn
import emoji
from tqdm import tqdm


OUTPUT_FILE = "output.txt"
MIN_CHAR = 5
MAX_CHAR = 50

conn = MySQLdb.connect(
    db="lang_8",
    user="root",
    passwd="",
    charset='utf8mb4',
)
c = conn.cursor()
print("MySQL connected\n")


def execute(sql):
    """sql文を実行してその結果を返す"""
    c.execute(sql)
    return list(c.fetchall())


def output_result(res):
    """結果をファイルに出力する"""
    with open(OUTPUT_FILE, 'w') as f:
        for i in range(len(res)):
            f.write("\t".join(res[i]) + "\n")
    print(f"\nWritten to {OUTPUT_FILE}")


def text_clean(text):
    """クリーニング"""
    # 全角スペース\u3000をスペースに変換
    text.replace('\u3000', ' ')
    # 全角→半角，重ね表現の除去
    text = neologdn.normalize(text, repeat=3)
    # 絵文字を削除
    text = ''.join(['' if c in emoji.UNICODE_EMOJI else c for c in text])
    # 桁区切りの除去と数字の置換
    text = re.sub(r'(\d)([,.])(\d+)', r'\1\3', text)
    text = re.sub(r'\d+', '0', text)
    # 半角記号の置換
    text = re.sub(r'[!-/:-@[-`{-~]', r' ', text)
    # 全角記号の置換 (ここでは0x25A0 - 0x266Fのブロックのみを除去)
    text = re.sub(u'[■-♯]', ' ', text)
    # 文頭の「数字列+スペース」を削除
    text = regex.sub(r'^(\p{Nd}+\p{Zs})(.*)$', r'\2', text)
    # 文頭行末の空白は削除
    text = text.strip()
    # 複数の空白を1つにまとめる
    text = re.sub(r'\s+', ' ', text)
    return text


def text_removal(text):
    """削除の場合Trueを返す"""
    # 日本語と記号のみの文を残す
    if not re.match(r'^[一-龠ぁ-んァ-ヶ、。!！?？，．,.\[\]「」ー\-\s\d]+$', text):
        return True
    # 文字数の制限
    if len(text) > MAX_CHAR or len(text) < MIN_CHAR:
        return True
    return False


def main():
    sql = f"""
        SELECT learner_sentence, correct_sentence
        FROM data
        WHERE learning_language='Japanese'
        AND (insert_edit_distance < 6 AND delete_edit_distance < 6)
        AND CHAR_LENGTH(correct_sentence) < {MAX_CHAR}
        AND CHAR_LENGTH(correct_sentence) > {MIN_CHAR}
    """
    res = execute(sql)
    print(f'original size: {len(res)}')
    corpus = []
    for row in tqdm(res):
        learner_sentence = text_clean(row[0])
        correct_sentence = text_clean(row[1])
        if text_removal(learner_sentence) or text_removal(correct_sentence):
            continue
        corpus.append((learner_sentence, correct_sentence))
    # 重複を削除してシャッフル
    corpus = list(set(corpus))
    print(f'cleaned size: {len(corpus)}')

    output_result(corpus)


if __name__ == "__main__":
    main()

conn.close()

