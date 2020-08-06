import tensorflow as tf
import math

feature_desc = {
    'x': tf.io.VarLenFeature(tf.float32),
    'y': tf.io.VarLenFeature(tf.float32)
}

TIME_STEPS = 5000     # backpropagation through time 的 time_steps
BATCH_SIZE = 64
INPUT_SIZE = 8      # sin 数据输入 size
OUTPUT_SIZE = 55     # cos 数据输出 size
CELL_SIZE = 128      # RNN 的 hidden unit size
LR = 1e-01         # learning rate
CELL_DEEP = 2
FC_HIDEN_NUM = 128
MODEL_DIR = 'models/gru2lay'
TRAIN_TARGET_COST = 0.333
IS_TRAINING = True
TRAIN_COST_PREC = 1e-07


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
        self.learning_rate = tf.placeholder(tf.float32, [])
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            inputl1 = self.add_fc_layer(self.xs, self.input_size, FC_HIDEN_NUM, 'inputl1')
            inputl2 = self.add_fc_layer(inputl1, FC_HIDEN_NUM, FC_HIDEN_NUM, name='inputl2')
            self.l_in_y = self.add_fc_layer(inputl2, FC_HIDEN_NUM, self.cell_size, 'inputlf')
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            outputl1 = self.add_fc_layer(self.cell_final_state[-1], self.cell_size, FC_HIDEN_NUM, name='outputl1')
            outputl2 = self.add_fc_layer(outputl1, FC_HIDEN_NUM, FC_HIDEN_NUM, name='outputl2')
            self.pred = self.add_fc_layer(outputl2, FC_HIDEN_NUM, self.output_size, name='outputlf')
        with tf.name_scope('cost'):
            self.compute_cost()
            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def add_fc_layer(self, inputs, inputsize, outputsize, name):
        l_in_x = tf.reshape(inputs, [-1, inputsize], name=name+'2_2D')
        Ws_in1 = self._weight_variable([inputsize, outputsize], name=name+'Ws_in1')
        bs_in1 = self._bias_variable([outputsize, ], name=name+'bs_in1')
        l_in_y = tf.matmul(l_in_x, Ws_in1) + bs_in1
        nb = tf.layers.batch_normalization(l_in_y, training=IS_TRAINING)
        active = tf.tanh(nb)
        return tf.reshape(active, [-1, self.n_steps, outputsize])

    def add_cell(self):
        gru_cell = tf.nn.rnn_cell.GRUCell(self.cell_size)
        cell = tf.nn.rnn_cell.ResidualWrapper(cell=gru_cell)
        cells = tf.nn.rnn_cell.MultiRNNCell([cell] * CELL_DEEP)
        _, self.cell_final_state = tf.nn.dynamic_rnn(cells, inputs=self.l_in_y, dtype=tf.float32, time_major=False)

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
    lr = LR
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
        dataset = dataset.shuffle(1024)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        cost = 100
        step = 0
        accuracy = 0
        miniest_cost = 100
        touchDownCounter = 0
        stopDownCounter = 0
        latestsave_cost = 0
        while True:
            feed_dict = {}
            try:
                x, y = sess.run(next_element)
                feed_dict[model.xs] = x
                feed_dict[model.ys] = y
                feed_dict[model.learning_rate] = lr

                if IS_TRAINING:
                    step = step + 1
                    if step % 10 == 0:
                        accuracy = sess.run(model.accuracy, feed_dict=feed_dict)
                        print('step %d accuracy = %f' % (step, accuracy))
                    _, cost, = sess.run([model.train_op, model.cost], feed_dict=feed_dict)
                    print('cost = ', cost)
                    if math.isclose(cost, miniest_cost, rel_tol=TRAIN_COST_PREC):
                        touchDownCounter = touchDownCounter + 1
                        stopDownCounter = 0
                        print('found miniestcost times =', touchDownCounter)
                        if touchDownCounter == 3:
                            break
                    elif cost < miniest_cost:
                        miniest_cost = cost
                        print('found miniestcost = ', miniest_cost)
                        touchDownCounter = 1
                        stopDownCounter = 0
                    else:
                        stopDownCounter = stopDownCounter + 1

                    if stopDownCounter > 50:
                        lr = lr / 2.0
                        stopDownCounter = 0
                        print('set lr =', lr)

                    if step % 1000 == 0 and latestsave_cost != miniest_cost:
                        latestsave_cost = miniest_cost
                        saver = tf.train.Saver()
                        saver.save(sess, MODEL_DIR, global_step=step)
                else:
                    accuracy = sess.run(model.accuracy, feed_dict=feed_dict)
                    print('accuracy = %f' % accuracy)

            except tf.errors.OutOfRangeError:
                break

        if IS_TRAINING:
            saver = tf.train.Saver()
            saver.save(sess, MODEL_DIR, global_step=step)
Main()
