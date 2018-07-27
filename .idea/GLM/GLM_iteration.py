#使用迭代法求解GLM模型参数
import csv
import tensorflow as tf
import numpy as np

row_num=26 #参数个数，含常数项1

def readcsv(filename="/home/mapd/dumps/output/GLM_base_date_90Days_one_hot.csv"):
    y=[]
    w=[]
    x=[]
    csv_reader = csv.reader(open(filename))
    i=0
    len=0
    for row in csv_reader:
        if i==0:
           len=list(row).__len__()
        if i!=0:
           print(row)
           y.append(row[0])
           x.append(row[2:len])
           w.append(row[1])
        i=i+1
    return np.array(y,dtype=float),np.array(w,dtype=float),np.array(x,dtype=float)

#提取数据
y,w,x=readcsv()


def tweedie_model(y=y,w=w,x=x,b_1=1.0,arr_len_sum=row_num):#用迭代法计算计算tweedie的参数和方差
    g=tf.Graph()
    with g.as_default():
        len=w.shape[0]
        p=tf.constant(1.5)
        y=tf.cast(tf.convert_to_tensor(np.array(y,dtype=float)),dtype=tf.float32)
        w=tf.cast(tf.convert_to_tensor(np.array(w,dtype=float)),dtype=tf.float32)
        x=tf.cast(tf.convert_to_tensor(np.array(x,dtype=float)),dtype=tf.float32)
        w_eye=tf.eye(len)*w
        print(b_1)
        b_1=tf.convert_to_tensor(b_1,dtype=tf.float32)

    with tf.Session(graph=g) as sess:
            u=tf.exp(tf.reduce_sum(x*b_1,axis=1))
            #计算2(dg(u)/du)=d(ln(u))/du=1/u^2,b(Q)的二次偏倒数
            G=tf.eye(len)*(1.0/u)
            b_Q_2=1.0/tf.pow(u,p)
            det_gu_2=tf.multiply(tf.pow(u,2),b_Q_2)
            W=tf.eye(len)*det_gu_2*w_eye
            x_t=tf.transpose(x)
            g_G=(tf.log(u)-tf.reduce_sum(tf.multiply(G,y-u)))
            b_tweedie_2=tf.reduce_sum(tf.multiply(tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.matmul(x_t,W),x)),x_t),W),g_G),axis=1)
            b_2=sess.run(b_tweedie_2)
    sess.close()
    return  b_2

b_1=np.zeros(26)
b_2=tweedie_model(y=y,w=w,x=x,b_1=b_1,arr_len_sum=row_num)
while True:
    b_1=b_2
    print(b_1)
    b_2=tweedie_model(y=y,w=w,x=x,b_1=b_1,arr_len_sum=row_num)
    print("---------")
    print(b_2)
    i=i+1
    if i>10:
       break