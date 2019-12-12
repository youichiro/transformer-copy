import regex
import argparse
import itertools
import numpy as np
from tqdm import tqdm


class NoiseInjector:
    def __init__(self, corpus, pset,
                 shuffle_sigma=0.5,
                 replace_mean=0.1, replace_var=0.03,
                 replace_p_mean=0.3, replace_p_var=0.03,
                 replace_p_choice_ratio=0.7,
                 delete_mean=0.1, delete_var=0.03,
                 delete_p_mean=0.15, delete_p_var=0.03,
                 delete_okurikana_mean=0.2, delete_okurikana_var=0.03,
                 add_mean=0.1, add_var=0.03,
                 add_p_choice_ratio=0.7):
        self.pset = pset
        self.corpus = self.to_word_list(corpus)
        self.shuffle_sigma = shuffle_sigma
        self.replace_a, self.replace_b = self.solve_ab(replace_mean, replace_var)
        self.replace_p_a, self.replace_p_b = self.solve_ab(replace_p_mean, replace_p_var)
        self.replace_p_choice_ratio = replace_p_choice_ratio
        self.delete_a, self.delete_b = self.solve_ab(delete_mean, delete_var)
        self.delete_p_a, self.delete_p_b = self.solve_ab(delete_p_mean, delete_p_var)
        self.delete_okurikana_a, self.delete_okurikana_b = self.solve_ab(delete_okurikana_mean, delete_okurikana_var)
        self.add_a, self.add_b = self.solve_ab(delete_mean, delete_var)
        self.add_p_choice_ratio=add_p_choice_ratio

    @staticmethod
    def solve_ab(mean, var):
        a = mean * mean * (1. - mean) / var - mean
        b = (1. - mean) * (mean * (1. - mean) / var - 1.)
        return a, b

    def to_word_list(self, corpus):
        """コーパスの単語を1次元配列に変換する"""
        black_list = ['、', '。', '「', '」', '（', '）', '》', '《', '’', '‘', '”', '“', 'ー']
        word_list = []
        for words in corpus:
            word_list += [w for w in words if w not in self.pset and w not in black_list]  # 助詞リストとブラックリストを除く
        return word_list

    @staticmethod
    def is_included_okurikana(word):
        """送り仮名を含む単語かどうかを判定する (ex. 教える -> True)"""
        pattern = regex.compile(r'\p{Script=Han}[\u3041-\u309F]+')
        m = pattern.fullmatch(word)
        return True if m else False

    @staticmethod
    def to_char(words):
        """単語リストを文字分割する"""
        return ' '.join(''.join(words))

    def shuffle(self, words, plabels, chunks):
        """文節の中で単語の順番をシャッフルする"""
        # chunks: [[今日, は], [いい, 天気], [です, ね]]
        if self.shuffle_sigma < 1e-6:
            return list(itertools.chain.from_iterable(chunks))
        ret = []
        ret_plabels = []
        ntoken = 0
        for chunk in chunks:
            shuffle_key = [i + np.random.normal(loc=0, scale=self.shuffle_sigma) for i in range(len(chunk))]
            new_idx = np.argsort(shuffle_key)
            # new_chunk = [chunk[i] for i in new_idx]
            new_chunk = [words[ntoken:ntoken+len(chunk)][i] for i in new_idx]
            new_plabels = [plabels[ntoken:ntoken+len(chunk)][i] for i in new_idx]
            ret += new_chunk
            ret_plabels += new_plabels
            ntoken += len(chunk)
        return ret, ret_plabels

    def replace(self, words, plabels):
        """(1)助詞の置換の割合を多くする (2)助詞は助詞セットの中から置換するようにする"""
        replace_ratio = np.random.beta(self.replace_a, self.replace_b)
        replace_p_ratio = np.random.beta(self.replace_p_a, self.replace_p_b)  # 助詞に対しての確率
        ret = []
        ret_plabels = []
        rnd = np.random.random(len(words))
        for i, (word, plabel) in enumerate(zip(words, plabels)):
            ratio = replace_p_ratio if plabel == 1 else replace_ratio
            if rnd[i] < ratio:
                if np.random.random() < self.replace_p_choice_ratio:  # p_choice_ratioの確率で助詞セットから置換
                    # 助詞セットからランダムに置換
                    pset = [p for p in self.pset if p != word[1]]  # 自分自身を除く
                    rnd_p = pset[np.random.randint(len(pset))]
                    ret.append((-1, rnd_p))
                    ret_plabels.append(1)
                else:
                    # vocabularyからランダムに置換
                    rnd_word = self.corpus[np.random.randint(len(self.corpus))]
                    if rnd_word == word[1]:
                        rnd_word = self.corpus[np.random.randint(len(self.corpus))]  # もう1回ランダム
                    ret.append((-1, rnd_word))
                    ret_plabels.append(0)
            else:
                ret.append(word)
                ret_plabels.append(plabel)
        return ret, ret_plabels

    def delete(self, words, plabels):
        """(1)助詞の削除の割合を多くする (2)送り仮名の削除の割合を多くする"""
        delete_ratio = np.random.beta(self.delete_a, self.delete_b)
        delete_p_ratio = np.random.beta(self.delete_p_a, self.delete_p_b)  # 助詞に対しての確率
        delete_okurikana_ratio = np.random.beta(self.delete_okurikana_a, self.delete_okurikana_b)  # 送り仮名に対しての確率
        ret = []
        ret_plabels = []
        rnd = np.random.random(len(words))
        for i, (word, plabel) in enumerate(zip(words, plabels)):
            ratio = delete_p_ratio if plabel == 1 else delete_ratio
            is_included_okurikana = self.is_included_okurikana(word[1])
            ratio = delete_okurikana_ratio if is_included_okurikana else ratio
            if rnd[i] < ratio:
                if is_included_okurikana:
                    # 送り仮名を1文字削除する
                    drop_idx = np.random.randint(len(word) - 1) + 1  # 漢字を除くため 1 ~ (len(w)-1)
                    dropped_word = ''.join([w for i, w in enumerate(word[1]) if i != drop_idx])
                    ret.append((-1, dropped_word))
                else:
                    continue
            ret.append(word)
            ret_plabels.append(plabel)
        return ret, ret_plabels

    def add(self, words, plabels, chunks=None):
        """助詞が挿入されやすくする"""
        add_ratio = np.random.beta(self.add_a, self.add_b)
        ret = []
        ret_plabels = []
        rnd = np.random.random(len(words))
        for i, (word, plabel) in enumerate(zip(words, plabels)):
            if rnd[i] < add_ratio:
                if np.random.random() < self.add_p_choice_ratio:
                    # 助詞セットからランダムに挿入
                    rnd_p = self.pset[np.random.randint(len(self.pset))]
                    ret.append((-1, rnd_p))
                    ret_plabels.append(1)
                else:
                    # vocabularyからランダムに挿入
                    rnd_word = self.corpus[np.random.randint(len(self.corpus))]
                    ret.append((-1, rnd_word))
                    ret_plabels.append(0)
            ret.append(word)
            ret_plabels.append(plabel)
        return ret, ret_plabels

    def parse(self, pairs):
        pairs = [(i, w, new_i) for new_i, (i, w) in enumerate(pairs)]
        orig_idx = np.argsort([i for i, w, ni in pairs])
        n = 0
        char_pairs = []
        for oi in orig_idx:
            i, word, new_i = pairs[oi]
            chars = ' '.join(word).split(' ')
            tmp = [new_i]
            for char in chars:
                if i >= 0:
                    tmp.append((n, char))
                    n += 1
                else:
                    tmp.append((-1, char))
            char_pairs.append(tmp)
        char_pairs = sorted(char_pairs)
        new_pairs = []
        for p in char_pairs:
            new_pairs += list(p[1:])

        align = []
        art = []
        for si in range(len(new_pairs)):
            ti = new_pairs[si][0]
            c = new_pairs[si][1]
            art.append(c)
            if ti >= 0:
                align.append('{}-{}'.format(si, ti))
        return art, align

    def inject_noise(self, words, plabels, chunks):
        funcs = [self.replace, self.delete, self.add]
        np.random.shuffle(funcs)
        pairs = [(i, w) for (i, w) in enumerate(words)]
        pairs, plabels = self.shuffle(pairs, plabels, chunks)
        for f in funcs:
            pairs, plabels = f(pairs, plabels)
        return self.parse(pairs)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--chunked-corpus', default='data/bccwj_clean_unidic.chunk',
                        help='tokenized and chunked corpus')
    parser.add_argument('-l', '--plabel-file', default='data/bccwj_clean_unidic.plabels',
                        help='particle labeled file')
    parser.add_argument('-p', '--pset-file', default='data/pset.txt', help='particle set file')
    parser.add_argument('-o', '--output-dir', default='data_art/ja_bccwj_clean', help='output directory')
    parser.add_argument('-e', '--epoch', type=int, default=10, help='epoch')
    parser.add_argument('-s', '--seed', type=int, default=2468, help='seed value')
    args = parser.parse_args()
    np.random.seed(args.seed)

    print(f"epoch={args.epoch}, seed={args.seed}")
    filename = args.output_dir.split('/')[-1]
    ofile_prefix = f"{args.output_dir}/{filename}_{args.epoch}"

    lines = open(args.chunked_corpus, encoding='utf-8').readlines()
    chunk_corpus = [[chunk.split(' ') for chunk in line.replace('\n', '').split('|')] for line in lines]
    # chunk_corpus: [[[今日, は], [いい, 天気], [です, ね]], ...]
    corpus = [line.replace('\n', '').replace('|', ' ').split(' ') for line in lines]
    # corpus: [[今日, は, いい, 天気, です, ね], ...]

    lines = open(args.plabel_file, encoding='utf-8').readlines()
    plabel_list = [[int(i) for i in line.replace('\n', '').split(' ')] for line in lines]
    # plabel_list: [[0, 1, 0, 0, 1, 0], ...]

    lines = open(args.pset_file, encoding='utf-8').readlines()
    pset = [line.replace('\n', '') for line in lines]

    assert len(chunk_corpus) == len(corpus) == len(plabel_list)

    noise_injector = NoiseInjector(corpus, pset)

    with open(ofile_prefix + '.src', 'w') as fs, \
         open(ofile_prefix + '.tgt', 'w') as ft, \
         open(ofile_prefix + '.forward', 'w') as fa:
        for words, plabels, chunks in zip(tqdm(corpus), plabel_list, chunk_corpus):
            tgt = noise_injector.to_char(words)
            src, align = noise_injector.inject_noise(words, plabels, chunks)
            fs.write(' '.join(src) + '\n')
            ft.write(tgt + '\n')
            fa.write(' '.join(align) + '\n')


if __name__ == '__main__':
    main()
