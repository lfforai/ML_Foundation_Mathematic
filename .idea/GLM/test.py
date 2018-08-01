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
while line and i<7:
    line = f.readline()
    print(line.split(":")[0])
    info_n[line.split(":")[0]]=line.split(":")[1]
    print(info_n[line.split(":")[0]])
    i=i+1
f.close()
# print(info_n)

# print("mileage_0_1971,mileage_1971_2957,mileage_2957_4063,mileage_4063_5781,mileage_5781_g,maxspeed_0_39,maxspeed_39_53,maxspeed_53_63,maxspeed_63_74,maxspeed_74_g,a_0_0,a_0_3,a_3_9,a_9_27,a_27_g,d_0_20,d_20_44,d_44_83,d_83_171,d_171_g,isf_0_0,isf_0_1,isf_1_3,isf_3_g,ish_0_2,ish_2_7,ish_7_13,ish_13_25,ish_25_g,isn_0_0,isn_0_g".split(",").__len__())
