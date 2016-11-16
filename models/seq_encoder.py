import chainer
import chainer.functions as F
import chainer.links as L
from chainer.cuda import cupy as xp


class SeqEncoder(chainer.Chain):
    def __init__(self, vocab_size, emb_dim, hidden_dim,
                 sequence_length):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        super(SeqEncoder, self).__init__(
            embed=L.EmbedID(self.vocab_size, self.emb_dim),
            lstm1=L.LSTM(self.emb_dim, self.hidden_dim),
            out=L.Linear(self.hidden_dim, self.vocab_size),
        )

    def reset_state(self):
        if hasattr(self, "lstm1"):
            self.lstm1.reset_state()
        if hasattr(self, "lstm2"):
            self.lstm2.reset_state()

    def encode(self, x_input, train=True):
        """
        inputを逆順にいれる
        """
        self.reset_state()
        for i in range(self.sequence_length):
            x = chainer.Variable(xp.asanyarray(x_input[:, self.sequence_length-i-1], 'int32'))
            h0 = self.embed(x)
            h1 = self.lstm1(F.dropout(h0, train=train))

        return h1