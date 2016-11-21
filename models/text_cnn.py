
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class TextCNN(chainer.Chain):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters):

        self.num_emb = vocab_size
        self.emb_dim = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.temperature = 1.0

        layers = dict()
        # embedding
        layers['embed'] = L.EmbedID(self.num_emb, self.emb_dim)
        # conv-filter
        layers.update({
            'conv-{}'.format(i): L.Convolution2D(1, num_filter, (filter_size, self.emb_dim))
            for i, (filter_size, num_filter) in enumerate(zip(self.filter_sizes, self.num_filters))
            })
        # highway-architecture
        layers['highway_out'] = L.Linear(sum(self.num_filters), sum(self.num_filters), nobias=True)
        layers['highway_gate'] = L.Linear(sum(self.num_filters), sum(self.num_filters), nobias=True)
        # output-layer
        layers['out'] = L.Linear(sum(self.num_filters), num_classes)

        super(TextCNN, self).__init__(**layers)

    def forward(self, x_input, ratio=0.5, train=True):

        try:
            batch_size, seq_length = x_input.shape
        except:
            batch_size = len(x_input)
            seq_length = len(x_input[0])

        x = chainer.Variable(self.xp.asarray(x_input, 'int32'))

        # embedding
        h1 = self.embed(x)[:, None, :, :]

        # conv-pooling
        pooled = []
        for i, (filter_size, num_filter) in enumerate(zip(self.filter_sizes, self.num_filters)):
            h2 = F.max_pooling_2d(F.relu(getattr(self, 'conv-{}'.format(i))(h1)), ksize=(seq_length - filter_size + 1, 1), stride=1)
            pooled.append(F.reshape(h2, (batch_size, -1)))
        h3 = F.concat(pooled)

        # highway network
        t = F.sigmoid(self.highway_gate(h3))
        h4 = t * F.relu(self.highway_out(h3)) + (1 - t) * h3
        h5 = F.dropout(h4, ratio=ratio,  train=train)

        return self.out(h5)

    def get_reward(self, x_input):
        pred = F.softmax(self.forward(x_input, train=False)).data
        return np.array([float(item[1]) for item in pred])

    def __call__(self, x_input, t, dis_dropout_keep_prob=0.5, train=True):

        y = self.forward(x_input, ratio=1-dis_dropout_keep_prob, train=train)
        t = chainer.Variable(self.xp.asarray(t, 'int32'))

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
