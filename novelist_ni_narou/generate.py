# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import chainer.serializers
import os
import sys
import pickle
import numpy as np
from model import SeqGAN
import time
import datetime
import multiprocessing as mp

pool = mp.Pool()

generator = SeqGAN(vocab_size=3000, emb_dim=128, hidden_dim=128,
                   sequence_length=40, start_token=0, lstm_layer=2
                   ).to_gpu()

batch_size = 10000


def progress_report(count, start_time, batch_size):
    duration = time.time() - start_time
    throughput = count * batch_size / duration
    sys.stderr.write(
        '\rtrain {} updates ({} samples) time: {} ({:.2f} samples/sec)'
            .format(count, count * batch_size,
                    str(datetime.timedelta(seconds=duration)).split('.')[0], throughput))

negative = []

# pool=None
st = time.time()
for x in range(30000 // batch_size):
    negative.append(generator.generate(batch_size))
    progress_report(x, st, batch_size)
t = time.time()
print()
print(t - st)
for x in range(30000 // batch_size):
    negative.append(generator.generate(batch_size, pool))
    progress_report(x, t, batch_size)
print(time.time() - t)
