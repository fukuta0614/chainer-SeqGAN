import re
import unicodedata
import numpy as np
from collections import Counter
import os
import json
import pickle


class Arasuji(object):
    def __init__(self, raw_data, vocab_size, seq_length):

        self.vocab_size = vocab_size
        self.seq_length = seq_length

        self.raw_text=[]
        for i, j in enumerate(raw_data):
            x = self.clean(j['summary'])
            if x and len(x) < seq_length:
                self.raw_text.append(x)

        self.data_num = len(self.raw_text)
        print('data num',self.data_num)

        words = []
        for line in self.raw_text:
            words.extend(line)
            words.append(' ')

        counter = Counter(words)
        self.word_freq = {word: cnt for word, cnt in counter.most_common(vocab_size-3)}
        self.vocab = ['_START'] + ['<EOS>'] + sorted(list(self.word_freq)) + ['  ']
        self.word2idx = {word:i for i, word in enumerate(self.vocab)}

        print('word num',len(self.vocab))

        self.data = np.ones((self.data_num, self.seq_length), np.int32) * (vocab_size-1)
        for i in range(self.data_num):
            for j in range(len(self.raw_text[i])):
                w = self.raw_text[i][j]
                if w in self.vocab:
                    self.data[i][j] = self.word2idx[w]
            else:
                self.data[i][len(self.raw_text[i])] = 1

        perm = np.random.permutation(self.data_num)
        self.test_idx = perm[:11700]
        self.train_idx = perm[11700:]

    def clean(self, string):
        SpecialLetters = r"""＊|¥|￥|#|＃|？|×|＋|†|:|;|~|¨|\xad|°|´|'̈|゙ ゚
                         |×|ĵ|α|β|π|σ|φ|ω|м|о|х|٩|۶|ก|ค|ง|จ|ณ|ท|\||
                         |น|ฟ|ม|ย|ร|ส|ห|ั|า|ิ|ี|ุ|เ|แ|ไ|่|‐|–|─|—|•|‥|′| '́|̈'
                         |…|※|‼|⁇|⁈|⁉|⁺|℃|ⅰ|ⅱ|ⅲ|←|↑|→|↓|⇒|⇔|−|〜 |〝|\〟|〜|〟
                         |∞|≒|≧|≪|≫|①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑪|⑫|\^
                         |━|│|┌|┐|■|□|△|▼|▽|◆|◇|○|◎|●|◒|◯|〇|◼|〓|★|☆|♀|♂|♥|♡|♦|♪|♬|♯|⚪|⚫|✕|✖|✳|〃
                         |\x81|\x8d|«|·|»|â|ä|è|é|ö|ø|ə|ɪ|ɲ|ʕ|̀|́|̄|̈|ά|γ|δ|ς|υ|д|з|щ|я|ᅠ|\u200f|―|‹|›|∀|√|∠|∮|∵|∽
                         |≋|≓|≔|≕|≖|≠|≡|≣|⊂|⊰|⊱|⊴|⋆|⋛|⋯|⌒|┏|┓|├|┤|╋|═|▄|◥|◻|◽|☓|☝|☪|☺|
                         ♉|♠|♢|♤|♧|♭|⚠|✤|✩|✴|✽|❁|❕|❗|❪|❫|❮|❯|➖|➡|⬆|⭐|؈|'ฺ|∽|♉|,"""

        string = string.split('。')[0] + '。'
        string = ' '.join(string.split())

        string = re.sub(r'{|\{|\[|『|【|《|〈|〔|〖','「',string)
        string = re.sub(r'{|\}|\]|』|】|》|〉|〕|〗','」', string)
        string = re.sub(r'[‘’“”″`\']', '"', string)
        string = re.sub(r"（.+?）", '', string)
        string = re.sub(r"\(.+?\)", '', string)
        string = re.sub(r"<.+?>", '', string)

        string = unicodedata.normalize('NFKC', string).lower()
        string = re.sub('https?://(\S)+', '', string)
        string = re.sub(SpecialLetters, '', string)
        string = ' '.join(string.split())

        if len(set(string)) < 5:
            return ''
        return string

    def get_train_data(self,batch_size):
        idx = np.random.choice(self.train_idx, batch_size, replace=False)
        return self.data[idx]

    def get_test_data(self, batch_size):
        idx = np.random.choice(self.test_idx, batch_size, replace=False)
        return self.data[idx]


if __name__ == '__main__':
    json_data = []
    for i, f_name in enumerate(os.listdir('dataset')):
        try:
            with open('dataset/' + f_name) as f:
                json_data.append(json.load(f))
        except:
            pass
    Loader = Arasuji(json_data, vocab_size=3000, seq_length=40)

    with open('arasuji.dat', 'wb') as f:
        pickle.dump(Loader, f)