from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

import numpy as np
import random as random
# a=tf.constant([1,2,3,4,5,6])
# b=tf.constant([0.5,0.5,0.5,0.5,0.5])
# list_a=[]
# with tf.Session() as sess:
#      for i in range(5):
#          list_a.append(a)
#      print(sess.run(tf.reduce_min(tf.abs(tf.cast(tf.stack(list_a,axis=1),dtype=tf.float32)-b),axis=1)))

#-*- coding: UTF-8 -*-
info_n={"att_modle_risk":"","att_model_all":"","att_modle_nh":"","att_modle_base":"","pei":"","weight":"","att_num":""}
f = open("/home/mapd/dumps/output/att_name.txt")             # 返回一个文件对象
line = f.readline()
print(line.split(":")[0])
info_n[line.split(":")[0]]=line.split(":")[1]
print(info_n[line.split(":")[0]])
i=1
while line and i<6:
    line = f.readline()
    print(line.split(":")[0])
    info_n[line.split(":")[0]]=line.split(":")[1]
    print(info_n[line.split(":")[0]])
    i=i+1
f.close()
print(info_n)

print("constant,mileage_1958_2947,mileage_2947_4052,mileage_4052_5776,mileage_5776_g,duration_268806_383555,duration_383555_505016,duration_505016_684012,duration_684012_g,maxspeed_0_39,maxspeed_39_53,maxspeed_53_63,maxspeed_63_74,a_0_0,a_3_9,a_9_27,a_27_g,d_0_20,d_20_44,d_83_171,d_171_g,isf_1_2,isf_2_g,ish_0_2,ish_2_5,ish_5_9,ish_17_g,isn_0_g".split(",").__len__())
