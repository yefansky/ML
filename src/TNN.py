import tensorflow as tf
import math

feature_desc = {
    'x': tf.io.VarLenFeature(tf.float32),
    'y': tf.io.VarLenFeature(tf.float32)
}

TIME_STEPS = 5000     # backpropagation through time 的 time_steps
BATCH_SIZE = 128
INPUT_SIZE = 8      # sin 数据输入 size
OUTPUT_SIZE = 55     # cos 数据输出 size
LR = 0.1         # learning rate
MODEL_DIR = 'tnnmodel'
SAVE_PATH = MODEL_DIR + '/tnn'
TRAIN_TARGET_COST = 0.333
TRAIN_COST_PREC = 1e-07
DROP_RATE = 0.02
IS_TRAINING = True


def _parse_function(example_proto):
    record = tf.io.parse_single_example(example_proto, feature_desc)
    data = record['x'].values
    label = record['y'].values
    return tf.reshape(data, [-1, 8]), label


class CNN(object):
    def __init__(self, n_steps, input_size, output_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            datain = self.xs
            datain = self.add_conv_layer(datain, 64, 2, 1, 1)
            laysize = [4, 8, 16, 32, 64]
            for i in range(len(laysize)):
                conv = self.add_conv_layer(datain, 64, laysize[i], 1, laysize[i])
                datain = datain + conv
                print('layer', i, '=', datain.get_shape())

            flat1 = tf.layers.flatten(datain)
            dense1 = tf.layers.dense(flat1, 256)
            dense2 = tf.layers.dense(dense1, 256)
            dense3 = tf.layers.dense(dense2, self.output_size)
            elu1 = tf.nn.elu(dense3)
            bn = tf.layers.batch_normalization(elu1, training=IS_TRAINING)
            output = tf.layers.dropout(bn, DROP_RATE, training=IS_TRAINING)
            self.pred = output
            self.predForAcc = bn
        with tf.name_scope('cost'):
            self.compute_cost()
            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_conv_layer(self, inputs, outChannel, kernalSize, strides, dilation):
        conv1 = tf.layers.conv1d(inputs, outChannel, kernalSize, strides=strides, padding='same', data_format='channels_last', dilation_rate=dilation)
        normal = tf.layers.batch_normalization(conv1, training=IS_TRAINING)
        relu = tf.nn.elu(normal)
        drop = tf.layers.dropout(relu, DROP_RATE, training=IS_TRAINING)
        return drop

    def compute_cost(self):
        ys = tf.reshape(self.ys, [-1], name='reshape_target')
        pred = tf.reshape(self.pred, [-1], name='reshape_pred')
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=pred))
        predForAcc = tf.reshape(self.predForAcc, [-1], name='reshape_pred_acc')
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.greater_equal(tf.sigmoid(predForAcc), 0.5), tf.cast(ys, tf.bool)), tf.float32))
        PT = tf.greater_equal(tf.sigmoid(predForAcc), 0.5)
        RT = tf.cast(ys, tf.bool)
        T = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(PT, RT), PT), tf.float32))
        PE = tf.reduce_sum(tf.cast(PT, tf.float32))
        TE = tf.reduce_sum(ys)
        P = T / PE
        R = T / TE
        self.F1 = 2 * P * R / (P + R + 1e-07)


def Main():
    model = CNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tfRecordPaths = tf.io.match_filenames_once('data*.tfrecord')

    print('BATCH_SIZE = %d, LR = %f' % (BATCH_SIZE, LR))

    with tf.Session(config=config) as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        saver = tf.train.Saver()
        if not IS_TRAINING:
            path = tf.train.latest_checkpoint(MODEL_DIR)
            saver.restore(sess, path)

        files = sess.run(tfRecordPaths)
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_function)
        dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, INPUT_SIZE), (OUTPUT_SIZE)), drop_remainder=True)
        dataset = dataset.shuffle(20000)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        cost = 100
        step = 0
        accuracy = 0
        miniest_cost = 100
        touchDownCounter = 0
        while True:
            feed_dict = {}
            try:
                x, y = sess.run(next_element)
                feed_dict[model.xs] = x
                feed_dict[model.ys] = y

                if IS_TRAINING:
                    step = step + 1
                    if step % 10 == 0:
                        accuracy, F1 = sess.run([model.accuracy, model.F1], feed_dict=feed_dict)
                        print('step %d accuracy = %f, F1 = %f' % (step, accuracy, F1))

                    _, cost = sess.run([model.train_op, model.cost], feed_dict=feed_dict)
                    print('cost = ', cost)

                    '''T, PE, TE = sess.run([model.T, model.PE, model.TE], feed_dict=feed_dict)
                    print('T = %f, PE = %f, TE = %f' % (T, PE, TE))'''

                    if math.isclose(cost, miniest_cost, rel_tol=TRAIN_COST_PREC):
                        touchDownCounter = touchDownCounter + 1
                        print('found miniestcost times =', touchDownCounter)
                        if touchDownCounter == 300:
                            break
                    elif cost < miniest_cost:
                        miniest_cost = cost
                        print('found miniestcost = ', miniest_cost)
                        touchDownCounter = 1
                        if step > 200:
                            saver.save(sess, SAVE_PATH, global_step=step)
                else:
                    accuracy = sess.run(model.accuracy, feed_dict=feed_dict)
                    print('accuracy = %f' % accuracy)

            except tf.errors.OutOfRangeError:
                break

        if IS_TRAINING:
            saver.save(sess, SAVE_PATH, global_step=step)
Main()
