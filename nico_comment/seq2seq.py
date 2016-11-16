
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np
from chainer.cuda import cupy as xp
# import temptemp


class Encoder(chainer.Chain):
    def __init__(self, vocab_size, emb_dim, hidden_dim,
                 sequence_length):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        super(Encoder, self).__init__(
            embed=L.EmbedID(self.vocab_size, self.emb_dim),
            lstm1=L.LSTM(self.emb_dim, self.hidden_dim),
            lstm2=L.LSTM(self.hidden_dim, self.hidden_dim),
            out=L.Linear(self.hidden_dim, self.vocab_size),
        )

    def reset_state(self):
        if hasattr(self, "lstm1"):
            self.lstm1.reset_state()
        if hasattr(self, "lstm2"):
            self.lstm2.reset_state()

    def encode(self, x_input, train=True):
        self.reset_state()
        batch_size = len(x_input)
        for i in range(self.sequence_length):
            x = chainer.Variable(xp.asanyarray(x_input[:, i], 'int32'))
            h0 = self.embed(x)
            h1 = self.lstm1(F.dropout(h0, train=train))
            h2 = self.lstm2(F.dropout(h1, train=train))
        return h2


class SeqGAN(chainer.Chain):
    def __init__(self, vocab_size, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 reward_gamma=0.95, lstm_layer=1):

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.0

        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, sequence_length)

        if lstm_layer == 2:
            super(SeqGAN, self).__init__(
                embed=L.EmbedID(self.vocab_size, self.emb_dim),
                lstm1=L.LSTM(self.emb_dim, self.hidden_dim),
                lstm2=L.LSTM(self.hidden_dim, self.hidden_dim),
                # lstm3=L.LSTM(self.hidden_dim, self.hidden_dim),
                out=L.Linear(self.hidden_dim, self.vocab_size),
            )
        else:
            super(SeqGAN, self).__init__(
                embed=L.EmbedID(self.vocab_size, self.emb_dim),
                lstm1=L.LSTM(self.emb_dim, self.hidden_dim),
                out=L.Linear(self.hidden_dim, self.vocab_size),
            )

    def reset_state(self):
        if hasattr(self, "lstm1"):
            self.lstm1.reset_state()
        if hasattr(self, "lstm2"):
            self.lstm2.reset_state()
        # if hasattr(self, "lstm3"):
        #     self.lstm3.reset_state()

    def decode_one_step(self, x, train=True):
        h0 = self.embed(x)
        h1 = self.lstm1(F.dropout(h0, train=train))
        h2 = self.lstm2(F.dropout(h1, train=train))
        y = self.out(h2)
        return y

    def generate(self, batch_size, train=False):
        """
        :return: (batch_size, self.seq_length)
        """

        self.reset_state()
        x = chainer.Variable(xp.asanyarray([self.start_token] * batch_size, 'int32'), volatile=True)
        gen_x = np.zeros((batch_size, self.sequence_length), 'int32')
        # gen_p = np.zeros((batch_size, self.sequence_length))

        for i in range(self.sequence_length):
            scores = self.decode_one_step(x, train)
            pred = F.softmax(scores)
            pred = cuda.to_cpu(pred.data) - np.finfo(np.float32).epsneg

            generated = []
            for j in range(batch_size):
                histogram = np.random.multinomial(1, pred[j])
                generated.append(int(np.nonzero(histogram)[0]))

            gen_x[:, i] = generated
            x = chainer.Variable(xp.asanyarray(generated, 'int32'), volatile=True)

        return gen_x

    def pretrain_step_autoencoder(self, x_input, context):
        """
        Maximum likelihood Estimation

        :param x_input:
        :return: loss
        """
        self.reset_state()
        batch_size = len(x_input)
        accum_loss = 0
        self.lstm1.h = context[0]
        self.lstm2.h = context[1]

        for i in range(self.sequence_length):
            if i == 0:
                x = chainer.Variable(xp.asanyarray([self.start_token] * batch_size, 'int32'))
            else:
                x = chainer.Variable(xp.asanyarray(x_input[:, i - 1], 'int32'))

            scores = self.decode_one_step(x)
            loss = F.softmax_cross_entropy(scores, chainer.Variable(xp.asanyarray(x_input[:, i], 'int32')))
            accum_loss += loss

        return accum_loss / self.sequence_length

    def pretrain_step(self, x_input):
        """
        Maximum likelihood Estimation

        :param x_input:
        :return: loss
        """
        self.reset_state()
        batch_size = len(x_input)
        accum_loss = 0
        for i in range(self.sequence_length):
            if i == 0:
                x = chainer.Variable(xp.asanyarray([self.start_token] * batch_size, 'int32'))
            else:
                x = chainer.Variable(xp.asanyarray(x_input[:, i - 1], 'int32'))

            scores = self.decode_one_step(x)
            loss = F.softmax_cross_entropy(scores, chainer.Variable(xp.asanyarray(x_input[:, i], 'int32')))
            accum_loss += loss

        return accum_loss / self.sequence_length

    def reinforcement_step(self, x_input, rewards):
        """
        :param x_input: (batch_size, seq_length)
        :param rewards: (batch_size, seq_length)
        :return:
        """
        self.reset_state()
        batch_size = len(x_input)
        accum_loss = 0
        for j in range(self.sequence_length):
            if j == 0:
                x = chainer.Variable(xp.asanyarray([self.start_token] * batch_size, 'int32'))
            else:
                x = chainer.Variable(xp.asanyarray(x_input[:, j - 1], 'int32'))

            scores = self.decode_one_step(x)
            log_prob = F.log_softmax(scores)  # (batch_size, vocab_size)
            loss = 0
            for i in range(batch_size):
                loss += log_prob[i, x_input[i, j]] * rewards[i, j]
            accum_loss += loss

        return -accum_loss

    def get_rewards(self, samples, dis, rollout_num=16):
        """
        get reward from generated sample FOR ROLLOUT

        :param samples: generated_sample (batch, seq_length)
        :param dis: discriminator
        :param rollout_num: num of roll out

        :return: (batch, seq_length) rewards[i,j] means rewards of a_{j-1} of batch i
        """

        batch_size = len(samples)
        reward_mat = np.zeros((batch_size, self.sequence_length), 'float32')
        for given in range(1, 20):
            rewards = self.roll_out(samples, given, dis, rollout_num)
            reward_mat[:, given - 1] = rewards

        reward_mat[:, 19] = dis.get_reward(samples)
        return reward_mat

    def roll_out(self, samples, given, dis, rollout_num):
        """
        compute expected rewards

        :param samples: generated_sample
        :param given: use x_0 ~ x_given as generated (state)
        :param dis: discriminator
        :param rollout_num: num of roll out

        :return: rewards (batch_size)
        """

        batch_size = len(samples)
        self.reset_state()
        gen_x = np.zeros((batch_size, self.sequence_length), 'int32')

        x = chainer.Variable(xp.asanyarray([self.start_token] * batch_size, 'int32'), volatile=True)
        self.decode_one_step(x, False)
        for i in range(given):
            gen_x[:, i] = samples[:, i]
            x = chainer.Variable(xp.asanyarray(samples[:, i], 'int32'), volatile=True)
            scores = self.decode_one_step(x, False)

        scores_ = scores
        c1, h1 = self.lstm1.c, self.lstm1.h
        c2, h2 = self.lstm2.c, self.lstm2.h

        rewards = []
        for _ in range(rollout_num):
            self.lstm1.set_state(chainer.Variable(c1.data.copy(), volatile=True), chainer.Variable(h1.data.copy(), volatile=True))
            self.lstm2.set_state(chainer.Variable(c2.data.copy(), volatile=True), chainer.Variable(h2.data.copy(), volatile=True))

            scores = chainer.Variable(scores_.data.copy(), volatile=True)
            for i in range(given, self.sequence_length):

                pred = F.softmax(scores)
                pred = cuda.to_cpu(pred.data) - np.finfo(np.float32).epsneg

                generated = []
                for j in range(batch_size):
                    histogram = np.random.multinomial(1, pred[j])
                    generated.append(int(np.nonzero(histogram)[0]))

                gen_x[:, i] = generated
                x = chainer.Variable(xp.asanyarray(generated, 'int32'), volatile=True)
                scores = self.decode_one_step(x)

            rewards.append(dis.get_reward(gen_x))

        return np.mean(rewards, axis=0)

    def target_loss(self, target_lstm, generated_num, batch_size, writer, summary_op, test_count, sess):

        #  Generated Samples
        generated_samples = []
        supervised_g_losses = []

        for _ in range(int(generated_num / batch_size)):
            gen = list(self.generate(batch_size))
            generated_samples.extend(gen)
            g_loss, summary = sess.run([target_lstm.pretrain_loss, summary_op], {target_lstm.x: gen})
            supervised_g_losses.append(g_loss)
            writer.add_summary(summary, test_count)
            test_count += 1

        return np.mean(supervised_g_losses), test_count




