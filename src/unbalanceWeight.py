import tensorflow as tf

feature_desc = {
    'x': tf.io.VarLenFeature(tf.float32),
    'y': tf.io.VarLenFeature(tf.float32)
}

TIME_STEPS = 5000     # backpropagation through time 的 time_steps
BATCH_SIZE = 256
INPUT_SIZE = 8      # sin 数据输入 size
OUTPUT_SIZE = 55     # cos 数据输出 size


def _parse_function(example_proto):
    record = tf.io.parse_single_example(example_proto, feature_desc)
    label = record['y'].values
    return label


class MODEL(object):
    def __init__(self, n_steps, input_size, output_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
        self.sample_sum = tf.Variable(tf.zeros([self.output_size], dtype=tf.float32), name='sample_weight')
        self.sample_count = tf.Variable(0, dtype=tf.float32, name='sample_count')
        self.CountOp = tf.assign(self.sample_sum, tf.add(self.sample_sum, tf.reduce_sum(self.ys, 0)))
        self.AddCounterOp = tf.assign(self.sample_count, tf.add(self.sample_count, BATCH_SIZE))
        self.sample_weight = self.sample_sum / self.sample_count
        p = self.sample_weight
        n = 1 - p
        self.w = n / (p + 1e-10)


def Main():
    model = MODEL(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tfRecordPaths = tf.io.match_filenames_once('data*.tfrecord')

    with tf.Session(config=config) as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        files = sess.run(tfRecordPaths)
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_function)
        dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=((OUTPUT_SIZE)), drop_remainder=True)
        dataset = dataset.shuffle(20000)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        feed_dict = {}
        for i in range(1000):
            y = sess.run(next_element)
            feed_dict[model.ys] = y
            sess.run([model.AddCounterOp, model.CountOp], feed_dict=feed_dict)

        samplePositiveCount, postiveWeight, sampleCount, weight = sess.run([model.sample_sum, model.sample_weight, model.sample_count, model.w], feed_dict=feed_dict)
        print('sample samplePositiveCount = ', samplePositiveCount)
        print('sample postiveWeight = ', postiveWeight)
        print('sample sampleCount = ', sampleCount)
        print('sample weight = ', weight)

Main()
