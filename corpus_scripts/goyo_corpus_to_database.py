import sys
import copy
import re
import MySQLdb
import editdistance
from tqdm import tqdm

CORPUS_PATH = '/path/to/naist-goyo-corpus'

# regex
text_id_regex = r'.*<id>(.*)<\/id>.*'
native_language_regex = r'.*<nationality>(.*)<\/nationality>.*'
tag_regex = r'<goyo .*?>'
crr_regex = r"<goyo (?!crr1).*?crr='(.*?)'( .*?)?>.*?<\/goyo>"
crr_regex_expanded = r"<goyo .*?crr1='(.*?)' crr2='(.*?)'( .*?)?>.*?<\/goyo>"

# mysql
conn = MySQLdb.connect(
    db='naist_goyo_corpus',
    user='root',
    passwd='',
    charset='utf8mb4'
)
c = conn.cursor()

def create_table():
    sql = """
        CREATE TABLE data (
            id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            text_id VARCHAR(255) NOT NULL,
            native_language VARCHAR(255) NOT NULL,
            sentence_id INT NOT NULL,
            correction_id INT NOT NULL,
            annotated_sentence TEXT,
            learner_sentence TEXT,
            correct_sentence TEXT,
            edit_distance INT NOT NULL,
            UNIQUE(text_id, sentence_id, correction_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    try:
        c.execute(sql)
    except:
        return False
    return True


def base_data():
    return {
        'text_id': None,
        'native_language': None,
        'sentence_id': None,
        'correction_id': 1,
        'annotated_sentence': None,
        'learner_sentence': None,
        'correct_sentence': None,
        'edit_distance': None,
    }


def make_value(data):
    return "('{}', '{}', {}, {}, '{}', '{}', '{}', {})".format(
        data['text_id'], data['native_language'], data['sentence_id'],
        data['correction_id'], data['annotated_sentence'], data['learner_sentence'],
        data['correct_sentence'], data['edit_distance']
    )

def insert_value(value):
    sql = """
        INSERT INTO data (text_id, native_language, sentence_id,
        correction_id, annotated_sentence, learner_sentence,
        correct_sentence, edit_distance) VALUES {};
    """.format(value)
    c.execute(sql)
    conn.commit()


def make_value_and_insert(data):
    data['edit_distance'] = editdistance.eval(data['learner_sentence'], data['correct_sentence'])
    value = make_value(data)
    insert_value(value)


def main():
    if create_table():
        print('Success creating table.')
    else:
        print('Table exists already.')

    with open(CORPUS_PATH, 'r') as f:
        src_file = f.read()

    for xml in tqdm(src_file.split('<corpus>')[1:]):
        data = base_data()
        xml = xml.replace('\n', '')
        # text_id
        match = re.match(text_id_regex, xml)
        data['text_id'] = match.groups()[0]
        # native_language
        match = re.match(native_language_regex, xml)
        data['native_language'] = match.groups()[0]

        text = xml[xml.find('<text>') + 6:xml.find('</text>')]
        text = text.replace('<p>', '').replace('</p>', '')
        sentence_id = 0
        for s in text.split('<s>'):
            annotated_sent = s.replace('</s>', '').replace('\u3000', ' ').strip()
            if not annotated_sent:
                continue
            sentence_id += 1
            data['sentence_id'] = sentence_id
            data['annotated_sentence'] = annotated_sent.replace('\'', '\\\'')
            sent_removed_tags = re.sub(tag_regex, '', annotated_sent)
            data['learner_sentence'] = sent_removed_tags.replace('</goyo>', '').replace('\'', '\\\'')

            if 'crr2' in annotated_sent:
                data1 = copy.deepcopy(data)
                data2 = copy.deepcopy(data)

                correct_sent_1 = re.sub(crr_regex, r'\1', annotated_sent)
                correct_sent_2 = re.sub(crr_regex_expanded, r'\1', correct_sent_1)
                data1['correct_sentence'] = correct_sent_2.replace('\'', '\\\'')
                make_value_and_insert(data1)

                data2['correction_id'] = 2
                correct_sent_1 = re.sub(crr_regex, r'\1', annotated_sent)
                correct_sent_2 = re.sub(crr_regex_expanded, r'\2', correct_sent_1)
                data2['correct_sentence'] = correct_sent_2.replace('\'', '\\\'')
                make_value_and_insert(data2)
            else:
                data['correct_sentence'] = re.sub(crr_regex, r'\1', annotated_sent).replace('\'', '\\\'')
                make_value_and_insert(data)
    print('Success inserting data.')


if __name__ == '__main__':
    main()
    conn.close()
