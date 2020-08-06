import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.random_normal([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
x_data = np.linspace(-math.pi * 15, math.pi * 15, 300 * 15)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.sin(x_data) + noise#加入噪声

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
l1 = add_layer(xs, 1, 30, activation_function=tf.nn.tanh)
l2 = add_layer(l1, 30, 30, activation_function=tf.nn.tanh)
out = add_layer(l2, 30, 30, activation_function=tf.nn.tanh)
# add output layer
prediction = add_layer(out, 30, 1, activation_function=tf.nn.tanh)

loss = tf.reduce_mean(tf.square(ys - prediction))
#train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
train_step = tf.train.AdamOptimizer(0.05).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)# 绘制基准图
# Interactive mode on
plt.ion()
plt.show()

for i in range(50000 * 50):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and remove the previous line
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)# 绘图间隔，防止cpu过载
