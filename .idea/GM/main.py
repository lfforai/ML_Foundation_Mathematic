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
a=tf.constant([[4.6361e+04,2.5900e+02,1.7900e+02],
             [0.0000e+00,0.0000e+00,0.0000e+00]])
arr_len_sum=10
w=tf.get_variable(name='tweedie_var', shape=[arr_len_sum], initializer=tf.random_normal_initializer(mean=0, stddev=1))
with tf.Session() as sess:
     sess.run(tf.initialize_all_variables())
     print(sess.run(tf.slice(a,[0,0],[1,-1])))
     print(sess.run(w))
