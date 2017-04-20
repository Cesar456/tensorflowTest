import numpy as np
import tensorflow as tf

xy = np.loadtxt("./train.txt", unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
hypothesis = tf.matmul(W, x_data)
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

learn_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learn_rate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(cost))
