import tensorflow as tf

x_data = [1, 2, 3, 4]
y_data = [2, 3, 4, 5]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

lr = 0.1

descent = W - tf.multiply(lr, tf.reduce_mean(tf.multiply(hypothesis - Y, X)))

# 将descent的值赋值给
train = W.assign(descent)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(cost, feed_dict={X: x_data, Y: y_data}))
