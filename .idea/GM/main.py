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
with tf.Session() as sess:
     print(sess.run(tf.slice(a,[0,0],[1,-1])))