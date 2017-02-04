import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np
# import temptemp


def choice(t):
    return np.random.choice(t[0], p=t[1])


class SeqGAN(chainer.Chain):
    def __init__(self, sequence_length, vocab_size, emb_dim, hidden_dim, start_token, reward_gamma=0.95, lstm_layer=1,
                 dropout=False, oracle=False, free_pretrain=False, encoder=None, latent_dim=None, tag_dim=0):

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.x0 = None
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.oracle = oracle
        self.dropout = dropout
        self.temperature = 1.0
        self.state = {}
        self.free_pretrain = free_pretrain
        self.encoder = encoder
        self.latent_dim = latent_dim if latent_dim else 0
        self.input_dim = self.emb_dim + self.latent_dim
        self.dropout_ratio = 0.5

        layers = dict()
        layers['embed'] = L.EmbedID(self.vocab_size, self.emb_dim,
                                    initialW=np.random.normal(scale=0.1, size=(self.vocab_size, self.emb_dim)))
        if tag_dim > 0:
            if encoder:
                self.tag_num = tag_dim
                layers['tag_embed'] = L.EmbedID(self.tag_num, self.tag_num,
                                                initialW=np.random.normal(scale=0.1, size=(self.tag_num, self.tag_num)))
                dec_input = self.tag_num + self.hidden_dim
                layers['dec_input'] = L.Linear(dec_input, self.hidden_dim)
            else:
                self.tag_num = tag_dim
                layers['tag_embed'] = L.EmbedID(self.tag_num, self.hidden_dim)

        if self.oracle:
            layers['lstm1'] = L.LSTM(self.input_dim, self.hidden_dim,
                                     lateral_init=chainer.initializers.normal.Normal(0.1),
                                     upward_init=chainer.initializers.normal.Normal(0.1),
                                     bias_init=chainer.initializers.normal.Normal(0.1),
                                     forget_bias_init=chainer.initializers.normal.Normal(0.1)
                                     )
        else:
            layers['lstm1'] = L.LSTM(self.input_dim, self.hidden_dim)
        if lstm_layer >= 2:
            layers['lstm2'] = L.LSTM(self.hidden_dim, self.hidden_dim)
        if lstm_layer >= 3:
            layers['lstm3'] = L.LSTM(self.hidden_dim, self.hidden_dim)
        if lstm_layer >= 4:
            layers['lstm4'] = L.LSTM(self.hidden_dim, self.hidden_dim)
        layers['out'] = L.Linear(self.hidden_dim, self.vocab_size, initialW=np.random.normal(scale=0.1, size=(self.vocab_size, self.hidden_dim)))

        super(SeqGAN, self).__init__(**layers)

    def reset_state(self):
        if hasattr(self, "lstm1"):
            self.lstm1.reset_state()
        if hasattr(self, "lstm2"):
            self.lstm2.reset_state()
        if hasattr(self, "lstm3"):
            self.lstm3.reset_state()
        if hasattr(self, "lstm4"):
            self.lstm4.reset_state()

    def save_state(self):
        if hasattr(self, "lstm1"):
            self.state['c1'] = self.lstm1.c
            self.state['h1'] = self.lstm1.h
        if hasattr(self, "lstm2"):
            self.state['c2'] = self.lstm2.c
            self.state['h2'] = self.lstm2.h
        if hasattr(self, "lstm3"):
            self.state['c3'] = self.lstm3.c
            self.state['h3'] = self.lstm3.h
        if hasattr(self, "lstm4"):
            self.state['c4'] = self.lstm4.c
            self.state['h4'] = self.lstm4.h

    def set_state(self):
        if hasattr(self, "lstm1"):
            self.lstm1.set_state(chainer.Variable(self.state['c1'].data.copy(), volatile=True),
                                 chainer.Variable(self.state['h1'].data.copy(), volatile=True))
        if hasattr(self, "lstm2"):
            self.lstm2.set_state(chainer.Variable(self.state['c2'].data.copy(), volatile=True),
                                 chainer.Variable(self.state['h2'].data.copy(), volatile=True))
        if hasattr(self, "lstm3"):
            self.lstm3.set_state(chainer.Variable(self.state['c3'].data.copy(), volatile=True),
                                 chainer.Variable(self.state['h3'].data.copy(), volatile=True))
        if hasattr(self, "lstm4"):
            self.lstm4.set_state(chainer.Variable(self.state['c4'].data.copy(), volatile=True),
                                 chainer.Variable(self.state['h4'].data.copy(), volatile=True))

    def decode_one_step(self, x, train=True, z=None):
        if self.dropout:
            if z is not None:
                h0 = F.concat((self.embed(x), z))
            elif len(x.data.shape) == 2:
                h0 = x
            else:
                h0 = self.embed(x)

            h = self.lstm1(F.dropout(h0, ratio=self.dropout_ratio, train=train))
            if hasattr(self, "lstm2"):
                h = self.lstm2(F.dropout(h, ratio=self.dropout_ratio, train=train))
            if hasattr(self, "lstm3"):
                h = self.lstm3(F.dropout(h, ratio=self.dropout_ratio, train=train))
            if hasattr(self, "lstm4"):
                h = self.lstm4(F.dropout(h, ratio=self.dropout_ratio, train=train))
            y = self.out(h)
            return y
        else:
            if z is not None:
                h0 = F.concat((self.embed(x), z))
            elif len(x.data.shape) == 2:
                h0 = x
            else:
                h0 = self.embed(x)

            h = self.lstm1(h0)
            if hasattr(self, "lstm2"):
                h = self.lstm2(h)
            if hasattr(self, "lstm3"):
                h = self.lstm3(h)
            if hasattr(self, "lstm4"):
                h = self.lstm4(h)
            y = self.out(h)
            return y

    def generate_use_tag(self, tag, x=None, pool=None, train=False):
        """
        :return: (batch_size, self.seq_length)
        """

        self.reset_state()
        batch_size = len(tag)
        if train:
            self.lstm1.h = self.tag_embed(chainer.Variable(self.xp.array(tag, 'int32')))

        else:
            if x is not None:
                _, mu_z, ln_var_z = self.encoder.encode_with_tag(x, tag, train)
                z = F.gaussian(mu_z, ln_var_z)
            else:
                z = chainer.Variable(self.xp.asanyarray(np.random.normal(scale=1, size=(batch_size, self.emb_dim)), 'float32'), volatile=not train)

            if self.encoder:
                tag_ = self.tag_embed(chainer.Variable(self.xp.array(tag, 'int32'), volatile=not train))
                self.lstm1.h = self.dec_input(F.concat((z, tag_)))
            else:
                self.lstm1.h = self.tag_embed(chainer.Variable(self.xp.array(tag, 'int32'), volatile=not train))

        x = chainer.Variable(self.xp.asanyarray([self.start_token] * batch_size, 'int32'), volatile=not train)

        gen_x = np.zeros((batch_size, self.sequence_length), 'int32')

        for i in range(self.sequence_length):
            scores = self.decode_one_step(x, train)
            pred = F.softmax(scores)
            pred = cuda.to_cpu(pred.data)
            # pred = cuda.to_cpu(pred.data) - np.finfo(np.float32).epsneg

            if pool:
                generated = pool.map(choice, [(self.vocab_size, p) for p in pred])
            else:
                generated = [np.random.choice(self.vocab_size, p=pred[j]) for j in range(batch_size)]
                # generated = []
                # for j in range(batch_size):
                #     # histogram = np.random.multinomial(1, pred[j])
                #     # generated.append(int(np.nonzero(histogram)[0]))

            gen_x[:, i] = generated
            x = chainer.Variable(self.xp.asanyarray(generated, 'int32'), volatile=not train)

        return gen_x

    def generate(self, batch_size, train=False, pool=None, random_input=False, random_state=False):
        """
        :return: (batch_size, self.seq_length)
        """

        self.reset_state()
        z = None

        if self.latent_dim:
            z = chainer.Variable(self.xp.asanyarray(np.random.normal(scale=1, size=(batch_size, self.latent_dim)), 'float32'), volatile=True)
        if random_input:
            self.x0 = np.random.normal(scale=1, size=(batch_size, self.emb_dim))
            x = chainer.Variable(self.xp.asanyarray(self.x0, 'float32'), volatile=True)
        elif random_state:
            self.lstm1.h = chainer.Variable(self.xp.asanyarray(np.random.normal(scale=1, size=(batch_size, self.emb_dim)), 'float32'), volatile=True)
            x = chainer.Variable(self.xp.asanyarray([self.start_token] * batch_size, 'int32'), volatile=True)
        else:
            x = chainer.Variable(self.xp.asanyarray([self.start_token] * batch_size, 'int32'), volatile=True)

        gen_x = np.zeros((batch_size, self.sequence_length), 'int32')

        for i in range(self.sequence_length):
            scores = self.decode_one_step(x, train, z)
            pred = F.softmax(scores)
            pred = cuda.to_cpu(pred.data)
            # pred = cuda.to_cpu(pred.data) - np.finfo(np.float32).epsneg

            if pool:
                generated = pool.map(choice, [(self.vocab_size, p) for p in pred])
            else:
                generated = [np.random.choice(self.vocab_size, p=pred[j]) for j in range(batch_size)]
                # generated = []
                # for j in range(batch_size):
                #     # histogram = np.random.multinomial(1, pred[j])
                #     # generated.append(int(np.nonzero(histogram)[0]))

            gen_x[:, i] = generated
            x = chainer.Variable(self.xp.asanyarray(generated, 'int32'), volatile=True)

        return gen_x

    def pretrain_step_vrae_tag(self, x_input, tag, word_drop_ratio=0.0, train=True):
        """
        Maximum likelihood Estimation

        :param x_input:
        :return: loss
        """
        batch_size = len(x_input)
        _, mu_z, ln_var_z = self.encoder.encode_with_tag(x_input, tag, train)

        self.reset_state()

        if self.latent_dim:
            z = F.gaussian(mu_z, ln_var_z)
        else:
            latent = F.gaussian(mu_z, ln_var_z)
            tag_ = self.tag_embed(chainer.Variable(self.xp.array(tag, 'int32'), volatile=not train))
            self.lstm1.h = self.dec_input(F.concat((latent, tag_)))
            z = None

        accum_loss = 0
        for i in range(self.sequence_length):
            if i == 0:
                x = chainer.Variable(self.xp.asanyarray([self.start_token] * batch_size, 'int32'), volatile=not train)
            else:
                if np.random.random() < word_drop_ratio and train:
                    x = chainer.Variable(self.xp.asanyarray([self.start_token] * batch_size, 'int32'), volatile=not train)
                else:
                    x = chainer.Variable(self.xp.asanyarray(x_input[:, i - 1], 'int32'), volatile=not train)

            scores = self.decode_one_step(x, z=z)
            loss = F.softmax_cross_entropy(scores, chainer.Variable(self.xp.asanyarray(x_input[:, i], 'int32'), volatile=not train))
            accum_loss += loss

        dec_loss = accum_loss
        kl_loss = F.gaussian_kl_divergence(mu_z, ln_var_z) / batch_size
        return dec_loss, kl_loss

    def pretrain_step_vrae(self, x_input, word_drop_ratio=0.0, train=True):
        """
        Maximum likelihood Estimation

        :param x_input:
        :return: loss
        """
        batch_size = len(x_input)
        _, mu_z, ln_var_z = self.encoder.encode(x_input, train)
        self.reset_state()

        if self.latent_dim:
            z = F.gaussian(mu_z, ln_var_z)
        else:
            self.lstm1.h = F.gaussian(mu_z, ln_var_z)
            z = None

        accum_loss = 0
        for i in range(self.sequence_length):
            if i == 0:
                x = chainer.Variable(self.xp.asanyarray([self.start_token] * batch_size, 'int32'), volatile=not train)
            else:
                if np.random.random() < word_drop_ratio and train:
                    x = chainer.Variable(self.xp.asanyarray([self.start_token] * batch_size, 'int32'), volatile=not train)
                else:
                    x = chainer.Variable(self.xp.asanyarray(x_input[:, i - 1], 'int32'), volatile=not train)

            scores = self.decode_one_step(x, z=z)
            loss = F.softmax_cross_entropy(scores, chainer.Variable(self.xp.asanyarray(x_input[:, i], 'int32'), volatile=not train))
            accum_loss += loss

        dec_loss = accum_loss
        kl_loss = F.gaussian_kl_divergence(mu_z, ln_var_z) / batch_size
        return dec_loss, kl_loss

    def pretrain_step_autoencoder(self, x_input):
        """
        Maximum likelihood Estimation

        :param x_input:
        :return: loss
        """

        h, _, _ = self.encoder.encode(x_input)

        self.reset_state()
        batch_size = len(x_input)
        accum_loss = 0
        self.lstm1.h = h

        for i in range(self.sequence_length):
            if i == 0:
                x = chainer.Variable(self.xp.asanyarray([self.start_token] * batch_size, 'int32'))
            else:
                x = chainer.Variable(self.xp.asanyarray(x_input[:, i - 1], 'int32'))

            scores = self.decode_one_step(x)
            loss = F.softmax_cross_entropy(scores, chainer.Variable(self.xp.asanyarray(x_input[:, i], 'int32')))
            accum_loss += loss

        return accum_loss / self.sequence_length

    def pretrain_step(self, x_input, tag=None):
        """
        Maximum likelihood Estimation

        :param x_input:
        :return: loss
        """
        self.reset_state()
        batch_size = len(x_input)
        accum_loss = 0
        if tag is not None:
            self.lstm1.h = self.tag_embed(chainer.Variable(self.xp.array(tag, 'int32')))

        for i in range(self.sequence_length):
            if i == 0:
                if self.free_pretrain:
                    x = chainer.Variable(self.xp.asanyarray(np.random.normal(scale=0.1, size=(batch_size, self.emb_dim)), 'float32'))
                else:
                    x = chainer.Variable(self.xp.asanyarray([self.start_token] * batch_size, 'int32'))
            else:
                x = chainer.Variable(self.xp.asanyarray(x_input[:, i - 1], 'int32'))

            scores = self.decode_one_step(x)
            loss = F.softmax_cross_entropy(scores, chainer.Variable(self.xp.asanyarray(x_input[:, i], 'int32')))
            accum_loss += loss

        return accum_loss  # / self.sequence_length

    def reinforcement_step(self, x_input, rewards, g_steps, tag=None, random_input=False):
        """
        :param x_input: (batch_size, seq_length)
        :param rewards: (batch_size, seq_length)
        :param g_steps: g_steps
        :return:
        """
        self.reset_state()
        batch_size = len(x_input)
        accum_loss = 0
        if tag is not None:
            self.lstm1.h = self.tag_embed(chainer.Variable(self.xp.array(tag, 'int32')))

        for j in range(self.sequence_length):
            if j == 0:
                if random_input:
                    x = chainer.Variable(self.xp.asanyarray(self.x0, 'float32'))
                else:
                    x = chainer.Variable(self.xp.asanyarray([self.start_token] * batch_size, 'int32'))
            else:
                x = chainer.Variable(self.xp.asanyarray(x_input[:, j - 1], 'int32'))

            scores = self.decode_one_step(x)
            log_prob = F.log_softmax(scores)  # (batch_size, vocab_size)
            loss = 0
            for i in range(batch_size):
                loss += log_prob[i, x_input[i, j]] * rewards[i, j]
            accum_loss += loss

        return -accum_loss / g_steps

    def get_rewards(self, samples, dis, rollout_num=16, tag=None, pool=None, gpu=0):
        """
        get reward from generated sample for ROLLOUT

        :param samples: generated_sample (batch, seq_length)
        :param dis: discriminator
        :param pool: multiprocess.Pool
        :param rollout_num: num of roll out
        :param gpu: gpu_id

        :return: (batch, seq_length) rewards[i,j] means rewards of a_{j-1} of batch i
        """

        batch_size = len(samples)

        if pool:
            dis.to_cpu()
            rewards = pool.map(self.roll_out, [(samples, given, dis, rollout_num, gpu) for given in range(1, self.sequence_length)])
            dis.to_gpu()
            rewards.append(dis.get_reward(samples))
            return np.array(rewards).T
        else:
            reward_mat = np.zeros((batch_size, self.sequence_length), 'float32')
            for given in range(1, self.sequence_length):
                if tag is not None:
                    rewards = self.roll_out((samples, given, dis, rollout_num, gpu, tag))
                else:
                    rewards = self.roll_out((samples, given, dis, rollout_num))

                reward_mat[:, given - 1] = rewards

            reward_mat[:, 19] = dis.get_reward(samples)
            return reward_mat

    def roll_out(self, args):
        """
        compute expected rewards

        :param samples: generated_sample
        :param given: use x_0 ~ x_given as generated (state)
        :param dis: discriminator
        :param pool: multiprocess.Pool
        :param rollout_num: num of roll out

        :return: rewards (batch_size)
        """
        tag = None
        if len(args) == 4:
            samples, given, dis, rollout_num = args
        elif len(args) == 5:
            samples, given, dis, rollout_num, gpu = args
            cuda.get_device(gpu).use()
            dis.to_gpu()
            self.to_gpu()
        elif len(args) == 6:
            samples, given, dis, rollout_num, gpu, tag = args
        else:
            raise AssertionError('undesired argument')

        batch_size = len(samples)
        self.reset_state()

        if tag is not None:
            self.lstm1.h = self.tag_embed(chainer.Variable(self.xp.array(tag, 'int32')))

        gen_x = np.zeros((batch_size, self.sequence_length), 'int32')

        x = chainer.Variable(self.xp.asanyarray([self.start_token] * batch_size, 'int32'), volatile=True)
        self.decode_one_step(x, False)
        for i in range(given):
            gen_x[:, i] = samples[:, i]
            x = chainer.Variable(self.xp.asanyarray(samples[:, i], 'int32'), volatile=True)
            scores = self.decode_one_step(x, False)

        scores_ = scores
        self.save_state()

        rewards = []
        for _ in range(rollout_num):
            self.set_state()
            scores = chainer.Variable(scores_.data.copy(), volatile=True)
            for i in range(given, self.sequence_length):

                pred = F.softmax(scores)
                pred = cuda.to_cpu(pred.data)

                generated = [np.random.choice(self.vocab_size, p=pred[j]) for j in range(batch_size)]

                # pred = cuda.to_cpu(pred.data) - np.finfo(np.float32).epsneg
                # generated = []
                # for j in range(batch_size):
                #     histogram = np.random.multinomial(1, pred[j])
                #     generated.append(int(np.nonzero(histogram)[0]))

                gen_x[:, i] = generated
                x = chainer.Variable(self.xp.asanyarray(generated, 'int32'), volatile=True)
                scores = self.decode_one_step(x, False)

            rewards.append(dis.get_reward(gen_x))

        return np.mean(rewards, axis=0)

    def target_loss(self, target_lstm, generated_num, batch_size, sess):

        #  Generated Samples
        supervised_g_losses = []

        for _ in range(int(generated_num / batch_size)):
            gen = self.generate(batch_size)
            g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: gen})
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)




