import tensorflow as tf
import math
#from tensorflow.keras import datasets, layers, models

feature_desc = {
    'x': tf.io.VarLenFeature(tf.float32),
    'y': tf.io.VarLenFeature(tf.float32)
}

TIME_STEPS = 5000     # backpropagation through time 的 time_steps
BATCH_SIZE = 32
INPUT_SIZE = 8      # sin 数据输入 size
OUTPUT_SIZE = 55     # cos 数据输出 size
LR = 1e-01         # learning rate
MODEL_DIR = 'models/gru2lay'
TRAIN_TARGET_COST = 0.333
IS_TRAINING = True
TRAIN_COST_PREC = 1e-07
kernalNum = 2
kernalSize = 5
poolsize = 2
poolstep = 2

def _parse_function(example_proto):
    record = tf.io.parse_single_example(example_proto, feature_desc)
    data = record['x'].values
    label = record['y'].values
    return tf.reshape(data, [-1, 8]), label


class CNNMODEL(object):
    def __init__(self, n_steps, input_size, output_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
        self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
        conv1 = tf.layers.conv1d(inputs=self.xs, filters=18, kernel_size=2, strides=1,
        padding='same', activation = tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=4, strides=4, padding='same')
        # (batch, 32, 18) -> (batch, 8, 36)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1,
        padding='same', activation = tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=4, strides=4, padding='same')
        # (batch, 8, 36) -> (batch, 2, 72)
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1,
        padding='same', activation = tf.nn.relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=4, strides=4, padding='same')
        flat = tf.reshape(max_pool_3, [batch_size, 144])
        logits = tf.layers.dense(flat, output_size)

        self.pred = tf.reshape(logits, [batch_size, output_size])
        self.compute_cost()
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def compute_cost(self):
        ys = tf.reshape(self.ys, [-1], name='reshape_target')
        pred = tf.reshape(self.pred, [-1], name='reshape_pred')
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=pred))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.greater_equal(tf.sigmoid(pred), 0.5), tf.cast(ys, tf.bool)), tf.float32))        


def Main():
    model = CNNMODEL(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tfRecordPaths = tf.io.match_filenames_once('./tianchi/data*.tfrecord')

    print('BATCH_SIZE = %d, LR = %f' % (BATCH_SIZE, LR))

    with tf.Session(config=config) as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        saver = tf.train.Saver()
        if not IS_TRAINING:
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
        latestsave_cost = 0
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
                    _, cost, = sess.run([model.train_op, model.cost], feed_dict=feed_dict)
                    print('cost = ', cost)
                    if math.isclose(cost, miniest_cost, rel_tol=TRAIN_COST_PREC):
                        touchDownCounter = touchDownCounter + 1
                        print('found miniestcost times =', touchDownCounter)
                        if touchDownCounter == 30:
                            break
                    elif cost < miniest_cost:
                        miniest_cost = cost
                        print('found miniestcost = ', miniest_cost)
                        touchDownCounter = 1

                    if step % 1000 == 0 and latestsave_cost != miniest_cost:
                        latestsave_cost = miniest_cost
                        saver.save(sess, MODEL_DIR, global_step=step)
                else:
                    accuracy = sess.run(model.accuracy, feed_dict=feed_dict)
                    print('accuracy = %f' % accuracy)

            except tf.errors.OutOfRangeError:
                break

        if IS_TRAINING:
            saver.save(sess, MODEL_DIR, global_step=step)
Main()
