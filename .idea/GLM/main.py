import mapd_jdbc
import pandas
import matplotlib.pyplot as plt

#在数据处理阶段应用mapd的gpu数据库，用标准sql语言对数据进行处理
#作者：罗锋
import tensorflow as tf
import numpy as np
import random as random
import os
import argparse
import sys

import os
#import 内部文件
import data_pre_deal
import k_means


# data_pre_deal.Day_30_att_random()
#数据预处理
for i in range(100):
    if i>44:
       data_pre_deal.Day_30_fun()
       k_means.k_means(model_type="tweedie",day_mark="30Days",step_model=6,step_k_means=10,inputfilename="",save_filename="/home/mapd/dumps/result/tweedie_30Days/result_"+str(i)+".txt")

# for i in range(100):
#     if i>88:
#        data_pre_deal.Day_30_fun()
#        k_means.k_means(model_type="tweedie",day_mark="30Days",step_model=6,step_k_means=10,inputfilename="",save_filename="/home/mapd/dumps/result/tweedie_30Days/result_"+str(i)+".txt")


#data_pre_deal.Day_90_basedata_run(all_risk='90.0',limit_risk='88.0',seg_range='90Days')
# data_pre_deal.Day_90_att_random()

# for i in range(100):
#     if i>-1:
#        data_pre_deal.Day_90_fun()
#        k_means.k_means(model_type="tweedie",day_mark="90Days",step_model=6,step_k_means=10,inputfilename="",save_filename="/home/mapd/dumps/result/tweedie_90Days/result_"+str(i)+".txt")
