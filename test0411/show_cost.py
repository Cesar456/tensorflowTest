import tensorflow as tf
from matplotlib import pyplot as plt

# Graph Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_smaples = len(X)

# model weight
W = tf.placeholder(tf.float32)

hypothesis = tf.multiply(X, W)

# Cost function
"""
tf.pow(x, y, name=None)	幂次方 
# tensor ‘x’ is [[2, 2], [3, 3]]
# tensor ‘y’ is [[8, 16], [2, 3]]
tf.pow(x, y) ==> [[256, 65536], [9, 27]]

tf.reduce_sum(input_tensor, reduction_indices=None, 
keep_dims=False, name=None)	计算输入tensor元素的和，或者安照reduction_indices指定的轴进行求和
# ‘x’ is [[1, 1, 1]
# [1, 1, 1]]
tf.reduce_sum(x) ==> 6
tf.reduce_sum(x, 0) ==> [2, 2, 2]
tf.reduce_sum(x, 1) ==> [3, 3]
tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
tf.reduce_sum(x, [0, 1]) ==> 6
"""
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / m
# 等价于cost = tf.reduce_mean(tf.square(hypothesis - Y))

init = tf.global_variables_initializer()

# for graphs
W_val = []
cost_val = []

# Launch the graphs
sess = tf.Session()
sess.run(init)

for i in range(-30, 50):
    x = sess.run(cost, feed_dict={W: i * 0.1})
    print(i * 0.1, x)
    W_val.append(i * 0.1)
    cost_val.append(x)

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('cost')
plt.xlabel('W')
plt.show()
