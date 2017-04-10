import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

print(add)
print(a + b)
print(mul)
print(a * b)

with tf.Session() as sess:
    print(sess.run(add, feed_dict={a: 2, b: 3}))
    feed = {a: 3, b: 5}
    print(sess.run(mul, feed_dict=feed))
