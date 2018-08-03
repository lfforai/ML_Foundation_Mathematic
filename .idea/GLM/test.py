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
# info_n={"att_modle_risk":"","att_model_all":"","att_modle_nh":"","att_modle_base":"","pei":"","weight":"","att_num":""}
# f = open("/home/mapd/dumps/output/att_name.txt")             # 返回一个文件对象
# line = f.readline()
# print(line.split(":")[0])
# info_n[line.split(":")[0]]=line.split(":")[1]
# print(info_n[line.split(":")[0]])
# i=1
# while line and i<7:
#     line = f.readline()
#     print(line.split(":")[0])
#     info_n[line.split(":")[0]]=line.split(":")[1]
#     print(info_n[line.split(":")[0]])
#     i=i+1
# f.close()
# # print(info_n)

print("constant,mileage_1960_2948,mileage_2948_4052,mileage_4052_5779,mileage_5779_g,maxspeed_0_39,maxspeed_53_63,maxspeed_63_74,maxspeed_74_g,a_0_0,a_0_3,a_9_27,a_27_g,d_0_20,d_20_44,d_82_171,d_171_g,isf_0_0,isf_1_3,isf_3_g,ish_0_2,ish_2_7,ish_7_13,ish_13_25,isn_0_0".split(",").__len__())
import random
import numpy as np

def random_range(num=6):
    max_pitch=1.0/num
    while True:
          distant=[]
          a=np.sort(np.random.rand(num-1))
          out=True
          for i in range(a.__len__()):
              if a[0]>0.20:
                 out=False
                 break
              if i<a.__len__()-1:
                 distant.append(a[i+1]-a[i])
                 if a[i+1]-a[i]>max_pitch*0.518 and a[i+1]-a[i]<max_pitch*1.615:
                    out=True
                 else:
                    out=False
                    break
          if min(a)>0.10 and out==True and 1.0-a[a.__len__()-1]>max_pitch*0.515 and 1.0-a[a.__len__()-1]<max_pitch*1.615:
             break
    return list(a)+[1.0]

print(random_range(num=4))

a='mileage_503_795'
b=1.1164757
c='aaaa'
d="cccc"
new=c+" "*(a.__len__()-c.__len__())+"|"
new_len=new.__len__()
print(" "*int(((new_len-d.__len__())/2))+d+" "*(new_len-d.__len__()-int((new_len-d.__len__())/2)))
print(new)
print(round(b,6))


def write2txt(txtName = "codingWord.txt",des_txt="",att='w'):
    import os
    if att=='a':
        if os.path.exists(txtName):
            pass
        else:
            with open(txtName,'w') as f:
                f.write(des_txt)
                f.close()
                return 0

    with open(txtName,att) as f:
        f.write(des_txt)
        f.close()
    return 0

write2txt(txtName = "/home/mapd/dumps/att_range/codingWord.txt",des_txt="a",att='a')
list_a=[[(1.0,1.0),(1.0,2.0)],[]]
print(list(map(lambda x:tuple(map(eval,(str(x).replace("(","").replace(")","").replace(" ","").split(",")))),"[(0.0, 1.0), (1.0, 4.0), (4.0, 11.0), (11.0, 2373.0)]".replace("]","").replace("[","").split("),"))))