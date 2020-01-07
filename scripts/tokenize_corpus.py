import argparse
from tqdm import tqdm
from mecab import Mecab

dict_path = '/tools/env/lib/mecab/dic/unidic'
mecab = Mecab(dict_path)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--corpus', required=True, help='Input raw corpus')
    parser.add_argument('--pref', required=True, help='Prefix of outputs')
    args = parser.parse_args()

    corpus = open(args.corpus).readlines()
    tokenized_file = args.pref + '_unidic.tgt'
    plabel_file = args.pref + '_unidic.plabels'
    with open(tokenized_file, 'w') as f1, open(plabel_file, 'w') as f2:
        for line in tqdm(corpus):
            line = line.replace('\n', '').replace(' ', '')
            if not line:
                continue
            words, parts = mecab.tagger(line)
            plabels = ['1' if p[:2] == '助詞' else '0' for p in parts]
            f1.write(' '.join(words) + '\n')
            f2.write(' '.join(plabels) + '\n')


if __name__ == '__main__':
    main()

