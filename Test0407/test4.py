import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

ss = tf.Session()
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(tf.square(W))

ss.run(tf.global_variables_initializer())

for step in range(2001):
    ss.run(train)
    if step % 20 == 0:
        print(step, ss.run(W))
