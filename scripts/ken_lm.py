import kenlm


class KenLM:
    def __init__(self, data):
        print('| loading KenLM data')
        self.lm = kenlm.Model(data)

    def calc(self, sentence):
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)
        score = self.lm.score(sentence, bos=True, eos=True)
        return score

