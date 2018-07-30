from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

import numpy as np
import random as random

FLAGS = None

batch_size=500
def read_my_file_format_csv(filename_queue,skip_header_lines=1):
    str_temp="constant,mileage_0_2000,mileage_4000_6000,mileage_6000_8000,mileage_8000_10000,mileage_10000_g,duration_0_400000,duration_800000_1200000,duration_1200000_g,maxspeed_0_24,maxspeed_24_48,maxspeed_72_96,maxspeed_96_g,a_10_20,a_20_30,a_30_40,a_40_g,d_100_200,d_200_300,d_300_400,d_400_g,isf_2_g,ish_9_18,ish_18_27,ish_27_36,ish_36_g,isn_0_g"
    list_temp=str_temp.split(",")
    len=list_temp.__len__()

    reader = tf.TextLineReader(skip_header_lines=skip_header_lines)
    key, value = reader.read(filename_queue)
    loss_total,risk,constant,mileage_0_2000,mileage_4000_6000,mileage_6000_8000,mileage_8000_10000,mileage_10000_g,duration_0_400000,duration_800000_1200000,duration_1200000_g,maxspeed_0_24,maxspeed_24_48,maxspeed_72_96,maxspeed_96_g,a_10_20,a_20_30,a_30_40,a_40_g,d_100_200,d_200_300,d_300_400,d_400_g,isf_2_g,ish_9_18,ish_18_27,ish_27_36,ish_36_g,isn_0_g = tf.decode_csv(value, record_defaults=[[1.0]]*(len+2))  #['null']解析为string类型 ，[1]为整型，[1.0]解析为浮点。
    featrues=tf.stack([constant,mileage_0_2000,mileage_4000_6000,mileage_6000_8000,mileage_8000_10000,mileage_10000_g,duration_0_400000,duration_800000_1200000,duration_1200000_g,maxspeed_0_24,maxspeed_24_48,maxspeed_72_96,maxspeed_96_g,a_10_20,a_20_30,a_30_40,a_40_g,d_100_200,d_200_300,d_300_400,d_400_g,isf_2_g,ish_9_18,ish_18_27,ish_27_36,ish_36_g,isn_0_g])
    label=loss_total
    weight=risk
    return label,weight,featrues                      #weight是每个样本的权重


def input_pipeline_csv(filenames, batch_size, num_epochs=None,file_config="/home/mapd/dumps/output/att_name.txt"):
    info_n={"att_modle_risk":"","att_modle_nh":"","att_modle_base":"","pei":""}
    # #-*- coding: UTF-8 -*-
    # f = open(file_config)             # 返回一个文件对象
    # line = f.readline()
    # print(line.split(":")[0])
    # info_n[line.split(":")[0]]=line.split(":")[1]
    # print(info_n[line.split(":")[0]])
    # i=1
    # while line and i<4:
    #     line = f.readline()
    #     print(line.split(":")[0])
    #     info_n[line.split(":")[0]]=line.split(":")[1]
    #     print(info_n[line.split(":")[0]])
    #     i=i+1
    # f.close()

    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True,seed=random.randint(1, 254))
    label_a,weight_batch,featrues_a=read_my_file_format_csv(filename_queue,skip_header_lines=1)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    y_batch,weight_batch,x_batch=tf.train.shuffle_batch(
        [label_a,weight_batch,featrues_a], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue,seed=random.randint(1, 254))
    return y_batch,weight_batch,x_batch

def tweedie_model(y,weight,x,w,p=tf.constant(1.5)):
    with tf.device('/gpu:1'):
        y_total_loss=y#确定y
        u=tf.exp(tf.reduce_sum(x*w,axis=1))
        theta=(-1.0)/(p-1.0)*tf.pow(u,(-1.0)*(p-1.0))
        K_theta=(-1.0)/(p-2.0)*tf.pow(((-(p-1.0))*theta),(p-2.0)/(p-1.0))
        loss=tf.reduce_sum(tf.multiply(weight,tf.multiply(y_total_loss,theta)-K_theta),axis=0)/tf.reduce_sum(weight,axis=0)
        return loss

def Poisson_model(y,weight,x,w):
    y_total_time=y#确定y
    u=tf.exp(tf.reduce_sum(tf.multiply(x,w),axis=1))
    theta=tf.log(u)
    K_theta=u
    loss=tf.reduce_sum(tf.multiply(weight,tf.multiply(y_total_time,theta)-K_theta),axis=0)/tf.reduce_sum(weight,axis=0)
    return loss


def main():
    # with tf.device('/gpu:0'):
        arr_len_sum=27
        learning_rate=1
        learning_rate_1=0.01
        learning_rate_2=0.001
        filenames=['/home/mapd/dumps/output/GLM_base_date_90Days_one_hot.csv']
        batch_size=3000
        num_epochs=None
        y_batch,weight_batch,x_batch=input_pipeline_csv(filenames, batch_size=batch_size, num_epochs=num_epochs)
        global_step =tf.Variable(0,trainable=False)

        # 必须要指定文件夹，保存到ckpt文件
        with tf.device('/gpu:1'):
             w_tweedie=tf.get_variable(name='tweedie_var', shape=[arr_len_sum], initializer=tf.random_normal_initializer(mean=0, stddev=1))
             loss_tweedie=-tweedie_model(y_batch,weight_batch,x_batch,w_tweedie)
        #,p=tf.constant(1.5))
        optimizer_tweedie = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tweedie,global_step=global_step)
        #optimizer_tweedie = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tweedie,global_step=global_step)
        optimizer_tweedie_1 = tf.train.AdamOptimizer(learning_rate=learning_rate_1).minimize(loss_tweedie,global_step=global_step)
        optimizer_tweedie_2 = tf.train.AdamOptimizer(learning_rate=learning_rate_2).minimize(loss_tweedie,global_step=global_step)
        #optimizer_tweedie=tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.85).minimize(loss=loss_tweedie,global_step=global_step)
        accuracy_tweedie=tf.reduce_sum(tf.abs(tf.exp(tf.reduce_sum(tf.multiply(x_batch, w_tweedie),axis=1))-y_batch)*weight_batch,axis=0)/tf.reduce_sum(weight_batch,axis=0)

        # w_poisson=tf.get_variable(name='poisson_var', shape=[arr_len_sum], initializer=tf.random_normal_initializer(mean=0, stddev=1))
        # loss_poisson=(-1.0)*Poisson_model(y_batch,weight_batch,x_onehot,w_poisson)
        # optimizer_poisson=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_poisson,global_step=global_step)
        # accuracy_poisson=tf.sqrt(tf.reduce_sum(tf.pow(tf.exp(tf.reduce_sum(tf.multiply(x_onehot, w_poisson),axis=1))-tf.reshape(tf.slice(y_batch,[0,2],[-1,1]),[-1]),2)*weight_batch)/tf.reduce_sum(weight_batch))

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            sess.run(init)
            # saver.restore(sess, '/temp/lf2/model.ckpt-300')
            coord = tf.train.Coordinator()#创建一个协调器，管理线程
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)#启动QueueRunner，此时文件名队列已经进队
            i=0
            mark=1000
            # saver.restore(sess, "/temp/lf2/100.ckpt")
            while(True):
                 if mark<0.5:
                    break
                 if i<=100:
                     sess.run(optimizer_tweedie)
                 if i>100 and i<=500:
                     sess.run(optimizer_tweedie_1)
                 if i>500:
                     sess.run(optimizer_tweedie_2)

                 if  i%20==0:
                      mark=sess.run(accuracy_tweedie)
                      print("accuracy_tweedie_:=",mark)
                      print("global_step:=",sess.run(global_step))
                 if i%300==0:
                     save_path=saver.save(sess, "/temp/lf2/model.ckpt-"+str(i))
                 i=i+1
            coord.request_stop()
            coord.join(threads)
main()