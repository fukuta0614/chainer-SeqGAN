import tensorflow as tf
import numpy as np
import os
from tensorflow.python.summary.event_accumulator import EventAccumulator


def scalar2arrays(scalarEvents):
    """
    converts scalarEvent to set of numpy.array
    """
    wall_times = []
    steps = []
    values = []

    for event in scalarEvents:
        wall_times.append(event.wall_time)
        steps.append(event.step)
        values.append(event.value)

    return np.array(wall_times), np.array(steps), np.array(values)


sess = tf.Session()
sess.run(tf.initialize_all_variables())


if __name__ == '__main__':

    for d in os.listdir('runs_vrae'):
        di = 'runs_vrae/' + d
        print(di)
        if os.path.isdir(di):
            summaries = di + '/summaries/'
            for f in os.listdir(summaries):
                accumulator = EventAccumulator(summaries + f)
                accumulator.Reload()  # load event files
                try:
                    _, steps, kl_values = scalar2arrays(accumulator.Scalars('test_loglikelihood'))
                    os.system('mv {}{} old/'.format(summaries, f))
                except:
                    continue

    for d in os.listdir('runs_vrae'):
        di = 'runs_vrae/' + d
        print(di)
        if os.path.isdir(di):

            sess = tf.Session()
            sess.run(tf.initialize_all_variables())

            summaries = di + '/summaries/'
            accumulator = EventAccumulator(summaries + os.listdir(summaries)[0])
            accumulator.Reload()  # load event files

            loss_ = tf.placeholder(tf.float32)
            ll = tf.scalar_summary('test_loglikelihood', loss_)
            summary_writer = tf.train.SummaryWriter(summaries, sess.graph)
            try:
                _, steps, kl_values = scalar2arrays(accumulator.Scalars('test_kl_loss'))
                _, steps, rec_values = scalar2arrays(accumulator.Scalars('test_rec_loss'))
            except:
                continue
            for step, kl, rec in zip(steps, kl_values, rec_values):
                summary = sess.run(ll, feed_dict={loss_: np.float32(kl + 30*rec)})
                summary_writer.add_summary(summary, step)
            print('ok')
