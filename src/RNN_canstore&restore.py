import tensorflow as tf

feature_desc = {
    'x': tf.io.VarLenFeature(tf.float32),
    'y': tf.io.VarLenFeature(tf.float32)
}

TIME_STEPS = 5000     # backpropagation through time 的 time_steps
BATCH_SIZE = 80
INPUT_SIZE = 8      # sin 数据输入 size
OUTPUT_SIZE = 55     # cos 数据输出 size
CELL_SIZE = 128      # RNN 的 hidden unit size
LR = 0.1         # learning rate
CELL_DEEP = 2
MODEL_DIR = 'models/gru2lay'
TRAIN_TARGET_ACCURACY = 0.98
IS_TRAINING = True

def _parse_function(example_proto):
    record = tf.io.parse_single_example(example_proto, feature_desc)
    data = record['x'].values
    label = record['y'].values
    return tf.reshape(data, [-1, 8]), label


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
        Ws_in1 = self._weight_variable([self.input_size, self.cell_size], name='Ws_in1')
        bs_in1 = self._bias_variable([self.cell_size, ], name='bs_in1')
        Ws_in2 = self._weight_variable([self.cell_size, self.cell_size], name='Ws_in2')
        bs_in2 = self._bias_variable([self.cell_size, ], name='bs_in2')
        r1 = tf.matmul(l_in_x, Ws_in1) + bs_in1
        l_in_y = tf.matmul(r1, Ws_in2) + bs_in2
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        gru_cell = tf.nn.rnn_cell.GRUCell(self.cell_size)
        res_cell = tf.nn.rnn_cell.ResidualWrapper(cell=gru_cell)
        cells = tf.nn.rnn_cell.MultiRNNCell([res_cell] * CELL_DEEP)
        _, self.cell_final_state = tf.nn.dynamic_rnn(cells, inputs=self.l_in_y, dtype=tf.float32, time_major=False)

    def add_output_layer(self):
        # shape = (batch, cell_size)
        l_out_x1 = tf.reshape(self.cell_final_state[-1], [-1, self.cell_size], name='2_2D1')
        Ws_out1 = self._weight_variable([self.cell_size, self.cell_size], name='Ws_out1')
        bs_out1 = self._bias_variable([self.cell_size, ], name='bs_out1')
        Ws_out2 = self._weight_variable([self.cell_size, self.output_size], name='Ws_out2')
        bs_out2 = self._bias_variable([self.output_size, ], name='bs_out2')
        p1 = tf.matmul(l_out_x1, Ws_out1) + bs_out1
        self.pred = tf.matmul(p1, Ws_out2) + bs_out2

    def compute_cost(self):
        ys = tf.reshape(self.ys, [-1], name='reshape_target')
        pred = tf.reshape(self.pred, [-1], name='reshape_pred')
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=pred))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.greater_equal(tf.sigmoid(pred), 0.5), tf.cast(ys, tf.bool)), tf.float32))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


def Main():
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tfRecordPaths = tf.io.match_filenames_once('data*.tfrecord')

    print('BATCH_SIZE = %d, CELL_SIZE = %d, CELL_DEEP = %d, LR = %f' % (BATCH_SIZE, CELL_SIZE, CELL_DEEP, LR))

    with tf.Session(config=config) as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        if not IS_TRAINING:
            saver = tf.train.Saver()
            saver.restore(sess, MODEL_DIR)

        files = sess.run(tfRecordPaths)
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_function)
        dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, INPUT_SIZE), (OUTPUT_SIZE)), drop_remainder=True)
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat(10)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        cost = 100
        step = 0
        accuracy = 0

        while True:
            feed_dict = {}
            try:
                x, y = sess.run(next_element)
                feed_dict[model.xs] = x
                feed_dict[model.ys] = y

                if IS_TRAINING:
                    step = step + 1
                    if step % 10 == 0:
                        accuracy = sess.run(model.accuracy, feed_dict=feed_dict)
                        print('step %d accuracy = %f' % (step, accuracy))
                        if accuracy > TRAIN_TARGET_ACCURACY:
                            break
                    _, cost, = sess.run([model.train_op, model.cost], feed_dict=feed_dict)
                    print('cost = ', cost)
                else:
                    accuracy = sess.run(model.accuracy, feed_dict=feed_dict)
                    print('accuracy = %f' % accuracy)

            except tf.errors.OutOfRangeError:
                break

        if IS_TRAINING:
            saver = tf.train.Saver()
            saver.save(sess, MODEL_DIR)
Main()
