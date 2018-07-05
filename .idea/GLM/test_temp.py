from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import tensorflow as tf

# 1000个4维输入向量，每个数取值为1-10之间的随机数
data = 10 * np.random.randn(1000, 4) + 1
# 1000个随机的目标值，值为0或1
target = np.random.randint(0, 2, size=1000)

# 创建Queue，队列中每一项包含一个输入数据和相应的目标值
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])

# 批量入列数据（这是一个Operation）
enqueue_op = queue.enqueue_many([data, target])
# 出列数据（这是一个Tensor定义）
data_sample, label_sample = queue.dequeue()

# 创建包含4个线程的QueueRunner
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)


with tf.Session() as sess:
    # 创建Coordinator
    coord = tf.train.Coordinator()
    # 启动QueueRunner管理的线程
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    print(enqueue_threads)
    # 主线程，消费100个数据
    for step in range(100):
        if coord.should_stop():
            break
        data_batch, label_batch = sess.run([data_sample, label_sample])
    # 主线程计算完成，停止所有采集数据的进程
    coord.request_stop()
    coord.join(enqueue_threads)