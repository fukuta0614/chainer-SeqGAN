import argparse
import os
import pickle
import random
import time
import sys
import datetime
import copy
import numpy as np
import chainer
from chainer import optimizers
from chainer import cuda
from chainer import serializers
import tensorflow as tf
import multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SeqGAN, TextCNN, SeqEncoder
from optimizer_hook import NamedWeightDecay

os.environ['PATH'] += ':/usr/local/cuda/bin'

parser = argparse.ArgumentParser()
parser.add_argument("--out", default='')
parser.add_argument("--gen", default='')
parser.add_argument("--dis", default='')
parser.add_argument("--enc", default='')
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--parallel', '-p', default=0, type=int)

#  Generator  Hyper-parameters
parser.add_argument("--gen_emb_dim", type=int, default=128)
parser.add_argument("--gen_hidden_dim", type=int, default=128)
parser.add_argument("--gen_grad_clip", type=int, default=5)
parser.add_argument("--gen_lr", type=float, default=1e-3)
parser.add_argument("--num_lstm_layer", type=int, default=1)
parser.add_argument("--no-dropout", dest='dropout', action='store_false', default=True)

#  Training  Hyper-parameters
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--total_epoch", type=int, default=800)
parser.add_argument("--gen_pretrain_epoch", type=int, default=100)


args = parser.parse_args()

# multiprocess worker
if args.parallel > 0:
    pool = mp.Pool(16)
else:
    pool = None

batch_size = args.batch_size
assert args.sample_per_iter % batch_size == 0

if args.out:
    out = args.out
else:
    out = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_vrae", out))
os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)

with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
    for k, v in args._get_kwargs():
        print('{} = {}'.format(k, v))
        f.write('{} = {}\n'.format(k, v))


cuda.get_device(args.gpu).use()

SEED = 88
random.seed(SEED)
np.random.seed(SEED)

# load nico-comment dataset loader
with open('nico_comment_processed.dat', 'rb') as f:
    train_comment_data, test_comment_data, train_tag_data, test_tag_data, vocab, tag_id = pickle.load(f)

train_num = len(train_comment_data)
test_num = len(test_comment_data)
vocab_size = 3000
seq_length = 30
start_token = 0


encoder = SeqEncoder(vocab_size=vocab_size, emb_dim=args.gen_emb_dim, hidden_dim=args.gen_hidden_dim,
                     sequence_length=seq_length)

if args.enc:
    serializers.load_hdf5(args.enc, encoder)

# generator
generator = SeqGAN(vocab_size=vocab_size, emb_dim=args.gen_emb_dim, hidden_dim=args.gen_hidden_dim,
                   sequence_length=seq_length, start_token=start_token, lstm_layer=args.num_lstm_layer,
                   dropout=args.dropout, free_pretrain=args.free_pretrain, encoder=encoder).to_gpu()
if args.gen:
    serializers.load_hdf5(args.gen, generator)


# set optimizer
enc_optimizer = optimizers.Adam(alpha=args.gen_lr)
enc_optimizer.setup(encoder)
enc_optimizer.add_hook(chainer.optimizer.GradientClipping(args.gen_grad_clip))

gen_optimizer = optimizers.Adam(alpha=args.gen_lr)
gen_optimizer.setup(generator)
gen_optimizer.add_hook(chainer.optimizer.GradientClipping(args.gen_grad_clip))

# summaries
sess = tf.Session()
sess.run(tf.initialize_all_variables())

summary_dir = os.path.join(out_dir, "summaries")

loss_ = tf.placeholder(tf.float32)
train_loss_summary = tf.scalar_summary('train_loss', loss_)
test_loss_summary = tf.scalar_summary('test_loss', loss_)

summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

dis_train_count = 0
gen_train_count = 0
test_count = 0

with open(os.path.join(out_dir, "generated_sample_pretrain.txt"), 'w') as f:
    f.write('')

with open(os.path.join(out_dir, "generated_sample.txt"), 'w') as f:
    f.write('')


def progress_report(count, start_time, batchsize):
    duration = time.time() - start_time
    throughput = count * batchsize / duration
    sys.stderr.write(
        '\r{} updates ({} samples) time: {} ({:.2f} samples/sec)'.format(
            count, count * batchsize, str(datetime.timedelta(seconds=duration)).split('.')[0], throughput
        )
    )


if not args.gen:
    print('Start pre-training generator...')
    start = time.time()
    for epoch in range(args.gen_pretrain_epoch):

        # pre-train
        pre_train_loss = []
        perm = np.random.permutation(train_num)

        for i in range(0, train_num, batch_size):
            batch = train_comment_data[perm[i:i+batch_size]]
            if args.ae_pretrain:
                g_loss = generator.pretrain_step_autoencoder(batch)
                enc_optimizer.zero_grads()
                gen_optimizer.zero_grads()
                g_loss.backward()
                enc_optimizer.update()
                gen_optimizer.update()
                pre_train_loss.append(float(g_loss.data))
            else:
                g_loss = generator.pretrain_step(batch)
                gen_optimizer.zero_grads()
                g_loss.backward()
                gen_optimizer.update()
                pre_train_loss.append(float(g_loss.data))

            # progress report
            gen_train_count += 1
            progress_report(gen_train_count, start, batch_size)

        # test
        test_loss = []
        perm = np.random.permutation(test_num)

        for i in range(0, test_num, batch_size):
            batch = test_comment_data[perm[i:i + batch_size]]
            g_loss = generator.pretrain_step(batch)
            test_loss.append(float(g_loss.data))
        test_count += 1

        print('\npre-train epoch {}  train_loss {}  test_loss {}'.format(epoch, np.mean(pre_train_loss),
                                                                         np.mean(test_loss)))
        summary = sess.run(train_loss_summary, feed_dict={loss_: np.mean(pre_train_loss)})
        summary_writer.add_summary(summary, test_count)
        summary = sess.run(test_loss_summary, feed_dict={loss_: np.mean(test_loss)})
        summary_writer.add_summary(summary, test_count)
        samples = generator.generate(10, train=False)
        with open(os.path.join(out_dir, "generated_sample_pretrain.txt"), 'a', encoding='utf-8') as f:
            f.write('\npre-train epoch {}  train_loss {} test_loss {} \n'.format(epoch, np.mean(pre_train_loss),
                                                                                 np.mean(test_loss)))
            for x in samples:
                f.write(''.join([vocab[w] for w in x]) + '\n')

        if epoch > 0 and epoch % 10 == 0:

            serializers.save_hdf5(os.path.join(out_dir, "models", "gen_pretrain_{}.model".format(epoch)), generator)
            serializers.save_hdf5(os.path.join(out_dir, "models", "enc_pretrain_{}.model".format(epoch)), encoder)
