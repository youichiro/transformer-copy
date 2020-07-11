import sys
import json
import MySQLdb
import editdistance
from tqdm import tqdm

from naist_edit_distance import levenshtein_distance as naist_editdistance


LANG8_DATA = "/path/to/lang-8-20111007-L1-v2_clean.json"

conn = MySQLdb.connect(
    db="lang_8",
    user="root",
    passwd="",
    charset="utf8mb4"
)
c = conn.cursor()


def create_table():
    # lang8を保存するデータベースを作成する
    sql = """
        CREATE TABLE data(
            id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            journal_id INT NOT NULL,
            sentence_id INT NOT NULL,
            sentence_number INT NOT NULL,
            correction_number INT NOT NULL,
            learning_language VARCHAR(255) NOT NULL,
            native_language VARCHAR(255) NOT NULL,
            learner_sentence TEXT,
            correction TEXT,
            correct_sentence TEXT,
            edit_distance INT NOT NULL,
            delete_edit_distance INT NOT NULL,
            insert_edit_distance INT NOT NULL,
            UNIQUE(journal_id, sentence_id, sentence_number, correction_number)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    try:
        c.execute(sql)
    except:
        return False
    return True


def clean_text(text):
    # テキストのクリーニング
    text = str(text)
    text = text.replace('\'', '’')
    text = text.replace('\"', '”')
    text = text.replace('\\', '')
    return text


def annotated_to_correct(text):
    # タグに基づいて訂正した文を作成する
    if text.lower() == 'good' or text.lower() == 'ok':
        return ''
    if '[sline]' in text:
        if text.count('[sline]') != text.count('[/sline]'):
            return ''
        while '[sline]' in text:
            b_idx = text.find('[sline]')
            e_idx = text.rfind('[/sline]')
            text = text[0:b_idx] + text[e_idx+8:]
    text = text.replace('[f-red]', '').replace('[/f-red]', '')
    text = text.replace('[f-blue]', '').replace('[/f-blue]', '')
    text = text.replace('[f-bold]', '').replace('[/f-bold]', '')
    text = text[-4:] if text[-4:].lower() == 'good' else text
    text = text[-2:] if text[-2:].lower() == 'ok' else text
    return text


def make_value(data):
    # INSERTするための1レコードを作る
    return "({}, {}, {}, {}, '{}', '{}', '{}', '{}', '{}', {}, {}, {})".format(
        data['journal_id'],
        data['sentence_id'],
        data['sentence_number'],
        data['correction_number'],
        data['learning_language'],
        data['native_language'],
        data['learner_sentence'],
        data['correction'],
        data['correct_sentence'],
        data['edit_distance'],
        data['delete_edit_distance'],
        data['insert_edit_distance'],
    )


def insert_values(values):
    # 全レコードをINSERTする
    sql = """
        INSERT INTO data (
            journal_id,
            sentence_id,
            sentence_number,
            correction_number,
            learning_language,
            native_language,
            learner_sentence,
            correction,
            correct_sentence,
            edit_distance,
            delete_edit_distance,
            insert_edit_distance
        ) VALUES {};
    """.format(', '.join(values))
    c.execute(sql)
    conn.commit()


def main():
    if create_table():  # テーブルの作成
        print('Success creating table.')
    else:
        print('Table exists already.')

    src_file = open(LANG8_DATA).readlines()

    for line in tqdm(src_file):
        data = json.loads(line.replace('\n', ''))
        assert len(data['learner_sentence']) == len(data['correction'])
        values = []
        for i in range(len(data['learner_sentence'])):
            if len(data['correction'][i]) > 0:
                for j in range(len(data['correction'][i])):
                    learner_sentence = clean_text(data['learner_sentence'][i])
                    correction = clean_text(data['correction'][i][j])
                    correct_sentence = annotated_to_correct(correction)
                    edit_distance = editdistance.eval(learner_sentence, correct_sentence)
                    delete_edit_distance, insert_edit_distance = naist_editdistance(learner_sentence, correct_sentence)
                    value = make_value({
                        'journal_id': int(data['journal_id'].replace('.', '')),
                        'sentence_id': int(data['sentence_id']),
                        'sentence_number': i+1,
                        'correction_number': j+1,
                        'learning_language': data['learning_language'],
                        'native_language': data['native_language'],
                        'learner_sentence': learner_sentence,
                        'correction': correction,
                        'correct_sentence': correct_sentence,
                        'edit_distance': editdistance.eval(learner_sentence, correct_sentence),
                        'delete_edit_distance': delete_edit_distance,
                        'insert_edit_distance': insert_edit_distance
                    })
                    values.append(value)
            else:
                # 訂正がない場合は訂正文を学習者文とし，編集距離は0とする
                learner_sentence = clean_text(data['learner_sentence'][i])
                value = make_value({
                      'journal_id': int(data['journal_id'].replace('.', '')),
                      'sentence_id': int(data['sentence_id']),
                      'sentence_number': i+1,
                      'correction_number': 1,
                      'learning_language': data['learning_language'],
                      'native_language': data['native_language'],
                      'learner_sentence': learner_sentence,
                      'correction': learner_sentence,
                      'correct_sentence': learner_sentence,
                      'edit_distance': 0,
                      'delete_edit_distance': 0,
                      'insert_edit_distance': 0
                })
                values.append(value)

        insert_values(values)
    print('Success inserting data.')


if __name__ == '__main__':
    main()
    conn.close()

