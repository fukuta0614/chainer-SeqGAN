# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.cuda import cupy as xp
import numpy as np


class SeqEncoder(chainer.Chain):
    def __init__(self, vocab_size, emb_dim, hidden_dim, sequence_length, tag_num=0, latent_dim=None):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim if latent_dim else hidden_dim
        self.sequence_length = sequence_length
        self.input_dim = self.hidden_dim + tag_num

        if tag_num > 0:
            super(SeqEncoder, self).__init__(
                tag_embed=L.EmbedID(tag_num, tag_num, initialW=np.random.normal(scale=0.1, size=(tag_num, tag_num))),
                embed=L.EmbedID(self.vocab_size, self.emb_dim, initialW=np.random.normal(scale=0.1, size=(self.vocab_size, self.emb_dim))),
                lstm1=L.LSTM(self.emb_dim, self.hidden_dim),
                linear_mu=L.Linear(self.input_dim, self.latent_dim),
                linear_ln_var=L.Linear(self.input_dim, self.latent_dim)
            )
        else:
            super(SeqEncoder, self).__init__(
                embed=L.EmbedID(self.vocab_size, self.emb_dim,
                                initialW=np.random.normal(scale=0.1, size=(self.vocab_size, self.emb_dim))),
                lstm1=L.LSTM(self.emb_dim, self.hidden_dim),
                linear_mu=L.Linear(self.input_dim, self.latent_dim),
                linear_ln_var=L.Linear(self.input_dim, self.latent_dim)
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
            x = chainer.Variable(xp.asanyarray(x_input[:, self.sequence_length-i-1], 'int32'), volatile=not train)
            h0 = self.embed(x)
            h1 = self.lstm1(F.dropout(h0, train=train))

        mu = self.linear_mu(h1)
        ln_var = self.linear_ln_var(h1)

        return h1, mu, ln_var

    def encode_with_tag(self, x_input, tag, train=True):
        """
        inputを逆順にいれる
        """
        self.reset_state()
        for i in range(self.sequence_length):
            x = chainer.Variable(xp.asanyarray(x_input[:, self.sequence_length-i-1], 'int32'), volatile=not train)
            h0 = self.embed(x)
            h1 = self.lstm1(F.dropout(h0, train=train))
        tag_ = self.tag_embed(chainer.Variable(self.xp.array(tag, 'int32'), volatile=not train))
        h1 = F.concat((h1, tag_))

        mu = self.linear_mu(h1)
        ln_var = self.linear_ln_var(h1)

        return h1, mu, ln_var
