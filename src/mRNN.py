import tensorflow as tf

feature_desc = {
    'x': tf.io.VarLenFeature(tf.float32),
    'y': tf.io.VarLenFeature(tf.float32)
}

TIME_STEPS = 5000     # backpropagation through time 的 time_steps
BATCH_SIZE = 50
INPUT_SIZE = 8      # sin 数据输入 size
OUTPUT_SIZE = 55     # cos 数据输出 size
CELL_SIZE = 64      # RNN 的 hidden unit size
LR = 0.0006          # learning rate


def _parse_function(example_proto):
    record = tf.io.parse_single_example(example_proto, feature_desc)
    data = record['x'].values
    label = record['y'].values
    return tf.reshape(data, [-1, 8]), label


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
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
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def getLSTMCell(self, n):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(n, forget_bias=1, initializer=tf.orthogonal_initializer(), state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=0.2, output_keep_prob=0.2)
        return cell

    def add_cell(self):
        cells = [self.getLSTMCell(n) for n in [self.cell_size, 64, 128, 32, self.cell_size]]
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(stacked_cell, self.l_in_y, time_major=False, dtype=tf.float32)

    def add_output_layer(self):
        # shape = (batch, cell_size)
        l_out_x = tf.reshape(self.cell_final_state[-1].c, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

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
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    tfRecordPaths = tf.io.match_filenames_once('data*.tfrecord')

    with tf.Session(config=config) as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        files = sess.run(tfRecordPaths)
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_function)
        dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, INPUT_SIZE), (OUTPUT_SIZE)), drop_remainder=True)

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        cost = 100
        step = 0
        while cost > 0.001:
            feed_dict = {}
            try:
                x, y = sess.run(next_element)
                feed_dict[model.xs] = x
                feed_dict[model.ys] = y
                _, cost, = sess.run([model.train_op, model.cost], feed_dict=feed_dict)
                print('cost = ', cost)
                step = step + 1
                if step % 10 == 0:
                    accuracy = sess.run(model.accuracy, feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                break

Main()
