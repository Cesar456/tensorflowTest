import tensorflow as tf

x_data = [1, 2, 3]
y_data = [12, 13, 14]

"""
tf.random_normal | tf.truncated_normal | tf.random_uniform

tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None) 
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) 
tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None) 
这几个都是用于生成随机数tensor的。尺寸是shape 
random_normal: 正太分布随机数，均值mean,标准差stddev 
truncated_normal:截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数 
random_uniform:均匀分布随机数，范围为[minval,maxval]
"""
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W * x_data + b

# 计算平方的平均值
# reduce_mean(data,op)
# op为空，mean等于所有值的平均值
# op为0， mean等于每列平均值列表
# op为1，mean等于每行平均值
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
a = tf.Variable(0.1)
# 梯度下降优化
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# 初始化变量
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
