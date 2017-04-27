from __future__ import print_function
import tensorflow as tf

x = tf.constant([[3.0, 1.0]])
y = tf.constant([[20.0], [15.0]])
mul = tf.matmul(x, y)
mat = tf.multiply(x, y)
sess = tf.Session()
print(sess.run(mul), sess.run(mat))
