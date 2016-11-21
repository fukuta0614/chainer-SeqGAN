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

from models import SeqGAN, TextCNN
from optimizer_hook import NamedWeightDecay

from dataset.arasuji import Arasuji

# cuda.cudnn_enable = False

os.environ['PATH'] += ':/usr/local/cuda/bin'

parser = argparse.ArgumentParser()
parser.add_argument("--out", default='')
parser.add_argument("--gen", default='')
parser.add_argument("--dis", default='')
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--parallel', '-p', default=0, type=int)

#  Generator  Hyper-parameters
parser.add_argument("--gen_emb_dim", type=int, default=128)
parser.add_argument("--gen_hidden_dim", type=int, default=128)
parser.add_argument("--gen_grad_clip", type=int, default=5)
parser.add_argument("--gen_lr", type=float, default=1e-3)
parser.add_argument("--num_lstm_layer", type=int, default=2)

#  Discriminator  Hyper-parameters
parser.add_argument("--dis_embedding_dim", type=int, default=64)
parser.add_argument("--dis_filter_sizes", default="1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20")
parser.add_argument("--dis_num_filters", default="100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160")
parser.add_argument("--dis_dropout_keep_prob", type=float, default=0.75)
parser.add_argument("--dis_l2_reg_lambda", type=float, default=0.2)
parser.add_argument("--dis_lr", type=float, default=1e-4)

#  Training  Hyper-parameters
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--total_epoch", type=int, default=800)
parser.add_argument("--gen_pretrain_epoch", type=int, default=100)
parser.add_argument("--dis_pretrain_epoch", type=int, default=50)

parser.add_argument("--g_steps", type=int, default=1)
parser.add_argument("--d_steps", type=int, default=5)
parser.add_argument("--K", type=int, default=5)

parser.add_argument("--rollout_update_ratio", type=float, default=0.8)
parser.add_argument("--sample_per_iter", type=int, default=10000)

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
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", out))
os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)

with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
    for k, v in args._get_kwargs():
        print('{} = {}'.format(k, v))
        f.write('{} = {}\n'.format(k, v))

cuda.get_device(args.gpu).use()

SEED = 88
random.seed(SEED)
np.random.seed(SEED)

# load data arasuji loader
with open('dataset/arasuji.dat', 'rb') as f:
    arasuji = pickle.load(f)

train_num = len(arasuji.train_idx)
test_num = len(arasuji.test_idx)
vocab_size = 3000
seq_length = 40
start_token = 0

# generator
generator = SeqGAN(vocab_size=vocab_size, emb_dim=args.gen_emb_dim, hidden_dim=args.gen_hidden_dim,
                   sequence_length=seq_length, start_token=start_token, lstm_layer=args.num_lstm_layer,
                   dropout=True).to_gpu()
if args.gen:
    serializers.load_hdf5(args.gen, generator)

# discriminator
discriminator = TextCNN(num_classes=2, vocab_size=vocab_size, embedding_size=args.dis_embedding_dim,
                        filter_sizes=[int(n) for n in args.dis_filter_sizes.split(',')],
                        num_filters=[int(n) for n in args.dis_num_filters.split(',')]
                        ).to_gpu()
if args.dis:
    serializers.load_hdf5(args.dis, discriminator)

# set optimizer
gen_optimizer = optimizers.Adam(alpha=args.gen_lr)
gen_optimizer.setup(generator)
gen_optimizer.add_hook(chainer.optimizer.GradientClipping(args.gen_grad_clip))

dis_optimizer = optimizers.Adam(alpha=args.dis_lr)
dis_optimizer.setup(discriminator)
dis_optimizer.add_hook(NamedWeightDecay(args.dis_l2_reg_lambda, '/out/'))

# summaries
sess = tf.Session()
sess.run(tf.initialize_all_variables())

summary_dir = os.path.join(out_dir, "summaries")

loss_ = tf.placeholder(tf.float32)
train_loss_summary = tf.scalar_summary('train_loss', loss_)
test_loss_summary = tf.scalar_summary('test_loss', loss_)
dis_loss_summary = tf.scalar_summary('dis_loss', loss_)
dis_acc_summary = tf.scalar_summary('dis_acc', loss_)

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
        for _ in range(train_num // batch_size):
            batch = arasuji.get_train_data(batch_size)
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
        for _ in range(test_num // batch_size):
            batch = arasuji.get_test_data(batch_size)
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
                f.write(''.join([arasuji.vocab[w] for w in x]) + '\n')

    serializers.save_hdf5(os.path.join(out_dir, "models", "gen_pretrain.model"), generator)

else:
    # test
    test_loss = []
    for _ in range(test_num // batch_size):
        batch = arasuji.get_test_data(batch_size)
        g_loss = generator.pretrain_step(batch)
        test_loss.append(float(g_loss.data))
    print('\npre-trained test_loss {}'.format(np.mean(test_loss)))
    test_count = args.gen_pretrain_epoch
    summary = sess.run(test_loss_summary, feed_dict={loss_: np.mean(test_loss)})
    summary_writer.add_summary(summary, test_count)

# discriminator pre-train
if not args.dis:
    train_count = 0
    start = time.time()
    print('Start pre-training discriminator...')

    for epoch in range(args.dis_pretrain_epoch):

        # negative = np.vstack([generator.generate(batch_size, pool=pool) for x in range(args.sample_per_iter // batch_size)])
        negative = generator.generate(args.sample_per_iter)

        for k in range(args.K):
            positive = arasuji.get_train_data(args.sample_per_iter)

            x = np.vstack([positive, negative])
            y = np.array([1] * args.sample_per_iter + [0] * args.sample_per_iter)
            sum_train_loss = []
            sum_train_accuracy = []
            perm = np.random.permutation(len(y))

            for i in range(0, len(y), batch_size):
                x_batch = x[perm[i:i + batch_size]]
                y_batch = y[perm[i:i + batch_size]]
                loss, acc = discriminator(x_batch, y_batch, args.dis_dropout_keep_prob)
                dis_optimizer.zero_grads()
                loss.backward()
                dis_optimizer.update()
                sum_train_loss.append(float(loss.data))
                sum_train_accuracy.append(float(acc.data))

                train_count += 1
                progress_report(train_count, start, batch_size)

        print('\ndis-train epoch ', epoch, 'train_loss ', np.mean(sum_train_loss), 'train_acc ',
              np.mean(sum_train_accuracy))
        dis_train_count += 1
        summary = sess.run(dis_loss_summary, feed_dict={loss_: np.mean(sum_train_loss)})
        summary_writer.add_summary(summary, dis_train_count)
        summary = sess.run(dis_acc_summary, feed_dict={loss_: np.mean(sum_train_accuracy)})
        summary_writer.add_summary(summary, dis_train_count)

    serializers.save_hdf5(os.path.join(out_dir, "models", "dis_pretrain.model"), discriminator)

gen_optimizer = optimizers.Adam(alpha=args.gen_lr)
gen_optimizer.setup(generator)
gen_optimizer.add_hook(chainer.optimizer.GradientClipping(args.gen_grad_clip))

# roll out generator
rollout_generator = copy.deepcopy(generator).to_cpu()
rollout_params = np.asanyarray(tuple(param.data for param in rollout_generator.params()))

print('#########################################################################\n')
print('Start Reinforcement Training ...')

start = time.time()
for epoch in range(1, args.total_epoch):

    print('total epoch ', epoch)
    tmp = time.time()
    # g-step
    mean_time = 0
    for step in range(args.g_steps):
        samples = generator.generate(batch_size, train=True)
        rewards = rollout_generator.get_rewards(samples, discriminator, pool=pool, gpu=args.gpu)
        loss = generator.reinforcement_step(samples, rewards, g_steps=args.g_steps)
        gen_optimizer.zero_grads()
        loss.backward()
        gen_optimizer.update()

        duration = time.time() - start
        step_time = time.time() - tmp
        mean_time += step_time
        tmp = time.time()
        sys.stderr.write('\rreinforce step {}/{} time: {} ({:.2f} sec/step)'.format(
            step + 1, args.g_steps, str(datetime.timedelta(seconds=duration)).split('.')[0], step_time))
    else:
        print('\rreinforce step {}/{} time: {} ({:.2f} sec/step)'.format(
            step + 1, args.g_steps, str(datetime.timedelta(seconds=duration)).split('.')[0], mean_time/args.g_steps))

    # update rollout generator
    for i, param in enumerate(generator.params()):
        rollout_params[i] += args.rollout_update_ratio * (cuda.to_cpu(param.data) - rollout_params[i])

    # d-step
    mean_time = 0
    for step in range(args.d_steps):

        # negative = np.vstack([generator.generate(batch_size, pool=pool) for x in range(args.sample_per_iter // batch_size)])
        negative = generator.generate(args.sample_per_iter)

        for i in range(args.K):
            positive = arasuji.get_train_data(args.sample_per_iter)

            x = np.vstack([positive, negative])
            y = np.array([1] * args.sample_per_iter + [0] * args.sample_per_iter)
            sum_train_loss = []
            sum_train_accuracy = []
            perm = np.random.permutation(len(y))

            for k in range(0, len(y), batch_size):
                x_batch = x[perm[i:i + batch_size]]
                y_batch = y[perm[i:i + batch_size]]
                loss, acc = discriminator(x_batch, y_batch, args.dis_dropout_keep_prob)
                dis_optimizer.zero_grads()
                loss.backward()
                dis_optimizer.update()
                sum_train_loss.append(float(loss.data))
                sum_train_accuracy.append(float(acc.data))

        dis_train_count += 1
        summary = sess.run(dis_loss_summary, feed_dict={loss_: np.mean(sum_train_loss)})
        summary_writer.add_summary(summary, dis_train_count)
        summary = sess.run(dis_acc_summary, feed_dict={loss_: np.mean(sum_train_accuracy)})
        summary_writer.add_summary(summary, dis_train_count)

        duration = time.time() - start
        step_time = time.time() - tmp
        mean_time += step_time
        tmp = time.time()
        sys.stderr.write('\rdis-train step {}/{} time: {} ({:.2f} sec/step)'.format(
            step + 1, args.d_steps, str(datetime.timedelta(seconds=duration)).split('.')[0], step_time))
    else:
        print('\rdis-train step {}/{} time: {} ({:.2f} sec/step)'.format(
            step + 1, args.d_steps, str(datetime.timedelta(seconds=duration)).split('.')[0], mean_time / args.d_steps))

    test_loss = []
    for _ in range(test_num // batch_size):
        batch = arasuji.get_test_data(batch_size)
        g_loss = generator.pretrain_step(batch)
        test_loss.append(float(g_loss.data))

    print('test_loss {}'.format(np.mean(test_loss)))
    test_count += 1
    summary = sess.run(test_loss_summary, feed_dict={loss_: np.mean(test_loss)})
    summary_writer.add_summary(summary, test_count)
    samples = generator.generate(10, train=False)
    with open(os.path.join(out_dir, "generated_sample.txt"), 'a', encoding='utf-8') as f:
        f.write('\ntotal epoch {} test_loss {} \n'.format(epoch, np.mean(test_loss)))
        for x in samples:
            f.write(''.join([arasuji.vocab[w] for w in x]) + '\n')

    if epoch % 10 == 0:
        serializers.save_hdf5(os.path.join(out_dir, "models/gen_{:03d}.model".format(epoch)), generator)
        serializers.save_hdf5(os.path.join(out_dir, "models/dis_{:03d}.model".format(epoch)), discriminator)
