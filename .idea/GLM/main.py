import tensorflow as tf
import numpy as np
import sys
import os
import time

# os.system("python ./ps0.py")
# time.sleep(10)
# os.system("python ./ps1.py")
# time.sleep(10)
# os.system("python ./input0.py")
# time.sleep(10)
# os.system("python ./input1.py")

#“https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10”
x=tf.placeholder(dtype=tf.float32,shape=[None,3])
linear_model=tf.layers.Dense(units=2)
y=linear_model(x)

batch_size = 5
ones = tf.ones([batch_size,8,20])
print(ones)
logits = tf.layers.dense(ones,10)
print(logits.get_shape())

var = tf.Variable(tf.random_normal([3]), dtype=tf.float32) # 生成一个变量
var2= tf.get_variable(name='var2',shape=[3],initializer=tf.random_uniform_initializer(maxval=-1., minval=1., seed=0))
tf.add_to_collection('losses', var) # add_to_collection()函数将新生成变量的L2正则化损失加入集合losses
tf.add_to_collection('losses', var2)

print("---------------------------")
c=tf.constant(value=1)
#print(assert c.graph is tf.get_default_graph())
print(c.graph)
print(tf.get_default_graph())


g=tf.Graph()
print("g:",g)
with g.as_default():
     d=tf.constant(value=2)
     print(d.graph)
     #print(g)

g2=tf.Graph()
print("g2:",g2)
g2.as_default()
e=tf.constant(value=15)
print(e.graph)
print("---------------------------")

c=tf.constant(value=1)
#print(assert c.graph is tf.get_default_graph())
print(c.graph)
print(tf.get_default_graph())

init = tf.global_variables_initializer()
with tf.Session() as sess:
     sess.run(init)
     vc = tf.get_collection('losses')
     print(vc)
     print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))


