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

a=tf.constant([[4.6361e+04,2.5900e+02,1.7900e+02],
             [0.0000e+00,0.0000e+00,0.0000e+00]])
b=tf.constant([1,2,3])
c=tf.constant([2,3,4])
d=tf.multiply(b,c)
print(b)
arr_len_sum=10
w_poisson=tf.Variable(name='poisson_var',initial_value=tf.zeros([17]))
with tf.train.MonitoredTrainingSession(master='',
                                       is_chief=None,
                                       checkpoint_dir="/temp/lf",
                                       hooks=None) as sess:
     print(sess.run(w_poisson))
