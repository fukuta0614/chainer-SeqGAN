import argparse
import os

import sys
import pickle
import random
import time
import copy
import multiprocessing as mp
import numpy as np
import chainer
from chainer import optimizers
from chainer import cuda
from chainer import serializers
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_oracle import TARGET_LSTM, Gen_Data_loader, Dis_dataloader, Likelihood_data_loader
from models import SeqGAN, TextCNN
from optimizer_hook import NamedWeightDecay


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

os.environ['PATH'] += ':/usr/local/cuda/bin'

parser = argparse.ArgumentParser()
parser.add_argument("--out", default='')
parser.add_argument("--gen", default='')
parser.add_argument("--genopt", default='')
parser.add_argument("--dis", default='')
parser.add_argument('--gpu', '-g', default=0, type=int)
parser.add_argument('--parallel', '-p', default=0, type=int)
args = parser.parse_args()

if args.parallel > 0:
    pool = mp.Pool(16)
else:
    pool = None

if args.out:
    out = args.out
else:
    out = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", out))
os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)

cuda.get_device(args.gpu).use()


#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
gen_emb_dim = 32
gen_hidden_dim = 32
gen_grad_clip = 5
gen_batch_size = 64
#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64
#########################################################################################
#  Training  Hyper-parameters
#########################################################################################
total_epoch = 800
gen_pretrain_epoch = 200
dis_pretrain_epoch = 50

rollout_update_ratio = 0.8

g_steps = 1
d_steps = 5
K = 3

SEED = 88
#########################################################################################


def generate_samples_pos(session, trainable_model, batch_size, gen_num, output_file):
    #  Generated Samples
    generated_samples = []
    for _ in range(int(gen_num / batch_size)):
        generated_samples.extend(trainable_model.generate(session))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buf = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buf)


def generate_samples_neg(trainable_model, batch_size, gen_num, output_file):
    #  Generated Samples
    # generated_samples = []
    # for _ in range(int(gen_num / batch_size)):
    #     generated_samples.extend(trainable_model.generate(batch_size, train=False))
    generated_samples = trainable_model.generate(gen_num, train=False)

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buf = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buf)


def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def significance_test(session, target_lstm, data_loader, output_file):
    loss = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = session.run(target_lstm.out_loss, {target_lstm.x: batch})
        loss.extend(list(g_loss))
    with open(output_file, 'w')as fout:
        for item in loss:
            buffer = str(item) + '\n'
            fout.write(buffer)

random.seed(SEED)
np.random.seed(SEED)

positive_file = 'save/real_data.txt'
negative_file = os.path.join(out_dir, 'generator_sample.txt')
eval_file = os.path.join(out_dir, 'eval.txt')
generated_num = 10000

gen_data_loader = Gen_Data_loader(gen_batch_size)
likelihood_data_loader = Likelihood_data_loader(gen_batch_size)
dis_data_loader = Dis_dataloader()
vocab_size = 5000
seq_length = 20
best_score = 1000
target_params = pickle.load(open('save/target_params.pkl', 'rb'), encoding='bytes')
target_lstm = TARGET_LSTM(vocab_size, 64, 32, 32, 20, 0, target_params)
start_token = 0

# generator
generator = SeqGAN(seq_length, vocab_size, gen_emb_dim, gen_hidden_dim, start_token, oracle=True).to_gpu()
if args.gen:
    print(args.gen)
    serializers.load_hdf5(args.gen, generator)

# discriminator
discriminator = TextCNN(num_classes=2, vocab_size=vocab_size,
                        embedding_size=dis_embedding_dim,
                        filter_sizes=dis_filter_sizes, num_filters=dis_num_filters).to_gpu()
if args.dis:
    serializers.load_hdf5(args.dis, discriminator)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
gen_data_loader.create_batches(positive_file)
generate_samples_pos(sess, target_lstm, 64, 10000, positive_file)

# summaries
summary_dir = os.path.join(out_dir, "summaries")

loss_ = tf.placeholder(tf.float32)
target_loss_summary = tf.scalar_summary('target_loss', loss_)
dis_loss_summary = tf.scalar_summary('dis_loss', loss_)
dis_acc_summary = tf.scalar_summary('dis_acc', loss_)

summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

dis_train_count = 0
test_count = 0

gen_optimizer = optimizers.Adam(alpha=1e-3)
gen_optimizer.setup(generator)
gen_optimizer.add_hook(chainer.optimizer.GradientClipping(gen_grad_clip))

dis_optimizer = optimizers.Adam(alpha=1e-4)
dis_optimizer.setup(discriminator)
dis_optimizer.add_hook(NamedWeightDecay(dis_l2_reg_lambda, '/out/'))

if not args.gen:

    print('Start pre-training generator...')

    for epoch in range(gen_pretrain_epoch):

        pre_train_loss = []
        for _ in range(gen_data_loader.num_batch):
            batch = gen_data_loader.next_batch()
            g_loss = generator.pretrain_step(batch)
            gen_optimizer.zero_grads()
            g_loss.backward()
            gen_optimizer.update()
            pre_train_loss.append(g_loss.data)

        # print('pre-train epoch ', epoch, 'train_loss ', np.mean(pre_train_loss))
        generate_samples_neg(generator, gen_batch_size, 1000, eval_file)
        likelihood_data_loader.create_batches(eval_file)
        test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
        print('pre-train epoch: {}  test loss: {}  lr: {}'.format(epoch, test_loss, gen_optimizer.lr))
        test_count += 1
        summary = sess.run(target_loss_summary, feed_dict={loss_: test_loss})
        summary_writer.add_summary(summary, test_count)

    serializers.save_hdf5(os.path.join(out_dir, "models", "gen_pretrain.model"), generator)
    serializers.save_hdf5(os.path.join(out_dir, "models", "gen_pretrain.opt"), gen_optimizer)

else:
    test_count = gen_pretrain_epoch
    test_loss = generator.target_loss(target_lstm, 1000, gen_batch_size, sess)
    summary = sess.run(target_loss_summary, feed_dict={loss_: test_loss})
    summary_writer.add_summary(summary, test_count)
    print('After pre-training:' + ' ' + str(test_loss))

gen_optimizer = optimizers.Adam(alpha=1e-3)
gen_optimizer.setup(generator)
gen_optimizer.add_hook(chainer.optimizer.GradientClipping(gen_grad_clip))

if not args.dis:

    for epoch in range(dis_pretrain_epoch):

        generate_samples_neg(generator, gen_batch_size, generated_num, negative_file)

        #  train discriminator
        dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
        dis_batches = dis_data_loader.batch_iter(zip(dis_x_train, dis_y_train), dis_batch_size, K)

        sum_train_loss = []
        sum_train_accuracy = []

        for batch in dis_batches:
            try:
                x_batch, y_batch = zip(*batch)
                loss, acc = discriminator(x_batch, y_batch, dis_dropout_keep_prob)
                dis_optimizer.zero_grads()
                loss.backward()
                dis_optimizer.update()
                sum_train_loss.append(float(loss.data))
                sum_train_accuracy.append(float(acc.data))
            except ValueError:
                pass

        print('dis-train epoch ', epoch, 'train_loss ', np.mean(sum_train_loss), 'train_acc ', np.mean(sum_train_accuracy))
        dis_train_count += 1
        summary = sess.run(dis_loss_summary, feed_dict={loss_: np.mean(sum_train_loss)})
        summary_writer.add_summary(summary, dis_train_count)
        summary = sess.run(dis_acc_summary, feed_dict={loss_: np.mean(sum_train_accuracy)})
        summary_writer.add_summary(summary, dis_train_count)
    serializers.save_hdf5(os.path.join(out_dir, "models", "dis_pretrain.model"), discriminator)
    serializers.save_hdf5(os.path.join(out_dir, "models", "dis_pretrain.opt"), dis_optimizer)

# roll out generator
rollout_generator = copy.deepcopy(generator)
if pool:
    rollout_generator.to_cpu()
rollout_params = np.asanyarray(tuple(param.data for param in rollout_generator.params()))

print('#########################################################################')
print('Start Reinforcement Training ...')

for epoch in range(1, total_epoch):

    print('total batch: ', epoch)

    for step in range(g_steps):
        samples = generator.generate(gen_batch_size, train=True, random_input=True)
        rewards = rollout_generator.get_rewards(samples, discriminator, rollout_num=16, pool=pool, gpu=args.gpu)
        print(rewards[:30])
        loss = generator.reinforcement_step(samples, rewards, g_steps=g_steps, random_input=True)
        gen_optimizer.zero_grads()
        loss.backward()
        gen_optimizer.update()
        print(' Reinforce step {}/{}'.format(step+1, g_steps))

    for i, param in enumerate(generator.params()):
        if pool:
            rollout_params[i] += rollout_update_ratio * (cuda.to_cpu(param.data) - rollout_params[i])
        else:
            rollout_params[i] += rollout_update_ratio * (param.data - rollout_params[i])

    for step in range(d_steps):

        # generate for discriminator
        generate_samples_neg(generator, gen_batch_size, generated_num, negative_file)

        # train discriminator
        dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
        dis_batches = dis_data_loader.batch_iter(zip(dis_x_train, dis_y_train), dis_batch_size, K)
        sum_train_loss = []
        sum_train_accuracy = []
        flag = True
        for batch in dis_batches:
            try:
                x_batch, y_batch = zip(*batch)
                loss, acc = discriminator(np.array(x_batch), y_batch, dis_dropout_keep_prob)
                if flag:
                    print(float(acc.data))
                    flag=False
                dis_optimizer.zero_grads()
                loss.backward()
                dis_optimizer.update()
                sum_train_loss.append(float(loss.data))
                sum_train_accuracy.append(float(acc.data))
            except ValueError:
                pass

        dis_train_count += 1
        summary = sess.run(dis_loss_summary, feed_dict={loss_: np.mean(sum_train_loss)})
        summary_writer.add_summary(summary, dis_train_count)
        summary = sess.run(dis_acc_summary, feed_dict={loss_: np.mean(sum_train_accuracy)})
        summary_writer.add_summary(summary, dis_train_count)
        print(' dis-train step: {}/{}  train_loss: {}  train_acc: {}'.format(step+1, d_steps, np.mean(sum_train_loss), np.mean(sum_train_accuracy)))

    test_loss = generator.target_loss(target_lstm, 1000, gen_batch_size, sess)
    print(' test_loss: ', test_loss)
    test_count += 1
    summary = sess.run(target_loss_summary, feed_dict={loss_: test_loss})
    summary_writer.add_summary(summary, test_count)
