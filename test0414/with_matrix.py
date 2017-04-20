import tensorflow as tf

x_data = [[1.0, 0.0, 3.0, 0.0, 5.0], [0.0, 2.0, 0.0, 4.0, 0.0]]
y_data = [5.0, 5.0, 5.0, 5.0, 5.0]

# w为一行二列
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = tf.matmul(W, X) + b
# hypothesis = tf.matmul(W, x_data) + b

# 矩阵相乘
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# cost = tf.reduce_mean(tf.square(hypothesis - y_data))

learn_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    # sess.run(train)
    if step % 20 == 0:
        print(sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))
        # print(sess.run(cost), sess.run([W, b]))
