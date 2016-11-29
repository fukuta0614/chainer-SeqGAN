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
# from optimizer_hook import NamedWeightDecay

os.environ['PATH'] += ':/usr/local/cuda/bin'

parser = argparse.ArgumentParser()
parser.add_argument("--out", default='')
parser.add_argument("--gen", default='')
parser.add_argument("--dis", default='')
parser.add_argument("--enc", default='')
parser.add_argument("--init", default=0, type=int)
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--parallel', '-p', default=0, type=int)

#  Generator  Hyper-parameters
parser.add_argument("--gen_emb_dim", type=int, default=256)
parser.add_argument("--gen_hidden_dim", type=int, default=256)
parser.add_argument("--gen_grad_clip", type=int, default=5)
parser.add_argument("--gen_lr", type=float, default=1e-3)
parser.add_argument("--num_lstm_layer", type=int, default=1)
parser.add_argument("--no-dropout", dest='dropout', action='store_false', default=True)
parser.add_argument("--kl_anneal", dest='kl_anneal', action='store_true', default=False)
parser.add_argument("--kl_anneal_ratio", type=float, default=1e-3)
parser.add_argument("--kl_initial", type=float, default=0)
parser.add_argument("--word_drop", type=float, default=0)
parser.add_argument("--concat_z", dest='latent_dim', action='store_const', const=int(128))
parser.add_argument("--concat_z_dim", dest='latent_dim', type=int)
parser.add_argument('--use_tag', dest='use_tag', action='store_true', default=False)

#  Training  Hyper-parameters
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--total_epoch", type=int, default=800)
parser.add_argument("--gen_pretrain_epoch", type=int, default=100)

parser.add_argument("--no_vae", dest='vae', action='store_false', default=True)

args = parser.parse_args()

# multiprocess worker
if args.parallel > 0:
    pool = mp.Pool(16)
else:
    pool = None

batch_size = args.batch_size

out = datetime.datetime.now().strftime('%m%d')
if args.out:
    out += '_'+args.out
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
with open('nico_comment_processed3.dat', 'rb') as f:
    train_comment_data, test_comment_data, train_tag_data, test_tag_data, vocab, tag_id = pickle.load(f)

train_num = len(train_comment_data)
test_num = len(test_comment_data)
tag_dim = len(tag_id) if args.use_tag else 0
vocab_size = 3000
seq_length = 20
start_token = 0

# encoder
encoder = SeqEncoder(vocab_size=vocab_size, emb_dim=args.gen_emb_dim, hidden_dim=args.gen_hidden_dim,
                     latent_dim=args.latent_dim, sequence_length=seq_length, tag_num=tag_dim).to_gpu()

if args.enc:
    serializers.load_hdf5(args.enc, encoder)

# generator
generator = SeqGAN(vocab_size=vocab_size, emb_dim=args.gen_emb_dim, hidden_dim=args.gen_hidden_dim,
                   sequence_length=seq_length, start_token=start_token, lstm_layer=args.num_lstm_layer,
                   dropout=args.dropout, encoder=encoder, latent_dim=args.latent_dim, tag_dim=tag_dim).to_gpu()
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
train_g_loss_summary = tf.scalar_summary('rec_loss', loss_)
train_kl_loss_summary = tf.scalar_summary('kl_loss', loss_)
test_loss_summary = tf.scalar_summary('test_loss', loss_)
test_g_loss_summary = tf.scalar_summary('test_rec_loss', loss_)
test_kl_loss_summary = tf.scalar_summary('test_kl_loss', loss_)

summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

dis_train_count = 0
gen_train_count = 0
test_count = args.init

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


print('Start pre-training generator...')
start = time.time()
C = args.kl_initial
for epoch in range(args.init, args.gen_pretrain_epoch):

    # pre-train
    pre_train_loss = []
    sum_g_loss = []
    sum_kl_loss = []
    perm = np.random.permutation(train_num)
    if args.kl_anneal:
        C += args.kl_anneal_ratio

    for i in range(0, train_num, batch_size):
        x_batch = train_comment_data[perm[i:i + batch_size]]
        tag_batch = train_tag_data[perm[i:i + batch_size]]
        if args.vae:
            if args.use_tag:
                g_loss, kl_loss = generator.pretrain_step_vrae_tag(x_batch, tag_batch, args.word_drop)
            else:
                g_loss, kl_loss = generator.pretrain_step_vrae(x_batch, args.word_drop)

            loss = g_loss + C * kl_loss

            enc_optimizer.zero_grads()
            gen_optimizer.zero_grads()
            loss.backward()
            enc_optimizer.update()
            gen_optimizer.update()

            pre_train_loss.append(float(loss.data))
            sum_g_loss.append(float(g_loss.data))
            sum_kl_loss.append(float(kl_loss.data))
        else:
            g_loss = generator.pretrain_step_autoencoder(x_batch)
            enc_optimizer.zero_grads()
            gen_optimizer.zero_grads()
            g_loss.backward()
            enc_optimizer.update()
            gen_optimizer.update()
            pre_train_loss.append(float(g_loss.data))

        # progress report
        gen_train_count += 1
        progress_report(gen_train_count, start, batch_size)

    # test
    test_loss = []
    sum_test_g_loss = []
    sum_test_kl_loss = []
    perm = np.random.permutation(test_num)

    for i in range(0, test_num, batch_size):
        x_batch = test_comment_data[perm[i:i + batch_size]]
        tag_batch = test_tag_data[perm[i:i + batch_size]]
        if args.vae:
            if args.use_tag:
                g_loss, kl_loss = generator.pretrain_step_vrae_tag(x_batch, tag_batch, args.word_drop, train=False)
            else:
                g_loss, kl_loss = generator.pretrain_step_vrae(x_batch, args.word_drop, train=False)

            loss = g_loss + kl_loss
            sum_test_g_loss.append(float(g_loss.data))
            sum_test_kl_loss.append(float(kl_loss.data))
            test_loss.append(float(loss.data))

        else:
            loss = generator.pretrain_step_(x_batch)
            test_loss.append(float(loss.data))

    test_count += 1

    if args.vae:
        print('\npre-train epoch {}'.format(epoch))
        print('  train : rec_loss {}  kl_loss {} loss {}'.format(np.mean(sum_g_loss), np.mean(sum_kl_loss), np.mean(pre_train_loss)))
        print('   test : rec_loss {}  kl_loss {} loss {}'.format(np.mean(sum_test_g_loss), np.mean(sum_test_kl_loss), np.mean(test_loss)))

        summary = sess.run(train_loss_summary, feed_dict={loss_: np.mean(pre_train_loss)})
        summary_writer.add_summary(summary, test_count)
        summary = sess.run(train_g_loss_summary, feed_dict={loss_: np.mean(sum_g_loss)})
        summary_writer.add_summary(summary, test_count)
        summary = sess.run(train_kl_loss_summary, feed_dict={loss_: np.mean(sum_kl_loss)})
        summary_writer.add_summary(summary, test_count)

        summary = sess.run(test_loss_summary, feed_dict={loss_: np.mean(test_loss)})
        summary_writer.add_summary(summary, test_count)
        summary = sess.run(test_g_loss_summary, feed_dict={loss_: np.mean(sum_test_g_loss)})
        summary_writer.add_summary(summary, test_count)
        summary = sess.run(test_kl_loss_summary, feed_dict={loss_: np.mean(sum_test_kl_loss)})
        summary_writer.add_summary(summary, test_count)

        if args.use_tag:
            tag_batch_1 = np.repeat(range(len(tag_id)), 5)
            samples = generator.generate_use_tag(tag_batch_1, train=False)
            perm = np.random.permutation(test_num)
            x_batch = test_comment_data[perm[:50]]
            tag_batch_2 = test_tag_data[perm[:50]]
            autoencode_samples = generator.generate_use_tag(tag_batch_2, x_batch, train=False)
            with open(os.path.join(out_dir, "generated_sample_pretrain.txt"), 'a', encoding='utf-8') as f:
                f.write('\npre-train epoch {}  train_loss {} test_loss {} \n'.format(epoch, np.mean(pre_train_loss),
                                                                                     np.mean(test_loss)))
                sentences = [''.join([vocab[w] for w in x]) for i, x in enumerate(samples)]
                for x, y in zip(sentences, tag_batch_1):
                    f.write(tag_id[y] + '     ' + x + '\n')

                sentences = [''.join([vocab[w] for w in x]) for i, x in enumerate(autoencode_samples)]
                for x, y in zip(sentences, tag_batch_2):
                    f.write(tag_id[y] + '     ' + x + '\n')
        else:
            samples = generator.generate(10, train=False)
            with open(os.path.join(out_dir, "generated_sample_pretrain.txt"), 'a', encoding='utf-8') as f:
                f.write('\npre-train epoch {}  train_loss {} test_loss {} \n'.format(epoch, np.mean(pre_train_loss),
                                                                                     np.mean(test_loss)))
                for x in samples:
                    f.write(''.join([vocab[w] for w in x]) + '\n')
    else:
        print('\npre-train epoch {}  train_loss {}  test_loss {}'.format(epoch, np.mean(pre_train_loss), np.mean(test_loss)))
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

    if epoch > 0 and epoch % 5 == 0:
        serializers.save_hdf5(os.path.join(out_dir, "models", "gen_pretrain_{}.model".format(epoch)), generator)
        serializers.save_hdf5(os.path.join(out_dir, "models", "enc_pretrain_{}.model".format(epoch)), encoder)
