#使用迭代法求解GLM模型参数,未计算检验参数，相关参数可以通过sas做检验效率更高
#目前迭代法的效率远高于
#迭代法GLM的推导来自https://max.book118.com/html/2015/0212/12389590.shtm
import csv
import tensorflow as tf
import numpy as np

row_num=28 #参数个数，含常数项1

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
           y.append(row[0])
           x.append(row[2:len])
           w.append(row[1])
        i=i+1
    return np.array(y,dtype=float),np.array(w,dtype=float),np.array(x,dtype=float)

#--------------tweedie_model------------------------------------------------------
def tweedie_model(y=np.zeros(1),w=np.zeros(1),x=np.zeros(1),arr_len_sum=row_num,loop_times=5):#用迭代法计算计算tweedie的参数和方差
    g=tf.Graph()
    with g.as_default():
        len=w.shape[0]
        p=tf.constant(1.5)
        y=tf.cast(tf.convert_to_tensor(np.array(y,dtype=float)),dtype=tf.float32)
        w=tf.cast(tf.convert_to_tensor(np.array(w,dtype=float)),dtype=tf.float32)
        x=tf.cast(tf.convert_to_tensor(np.array(x,dtype=float)),dtype=tf.float32)
        w_eye=tf.eye(len)*w

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=g,config=config) as sess:
        #计算出y的均值，用来替换缩小初始值的大小，避免出现非奇异矩阵
        init_constant=tf.log(tf.reduce_mean(y))
        b_1=np.random.normal(size=[arr_len_sum])/5.0#除了常数项以外,其余项比较小
        b_1[0]=sess.run(init_constant)
        b_1=tf.convert_to_tensor(b_1,dtype=tf.float32)
        i=1
        while True:
            u=tf.exp(tf.reduce_sum(x*b_1,axis=1))
            #计算2(dg(u)/du)=d(ln(u))/du=1/u^2,b(Q)的二次偏倒数
            G=tf.eye(len)*(1.0/u)
            b_Q_2=1.0/tf.pow(u,p)
            det_gu_2=tf.multiply(tf.pow(u,2),b_Q_2)
            W=tf.eye(len)*det_gu_2*w_eye
            x_t=tf.transpose(x)
            g_G=tf.reduce_sum(x*b_1,axis=1)+tf.reduce_sum(tf.multiply(G,y-u),axis=1)
            b_tweedie_2=tf.reduce_sum(tf.multiply(tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.matmul(x_t,W),x)),x_t),W),g_G),axis=1)
            if i>loop_times:
               break
            i=i+1
            b_1=b_tweedie_2
        b_2=sess.run(b_tweedie_2)
    sess.close()
    #初始化参数
    return  b_2

def train_tweedie_model2b_1(filename="/home/mapd/dumps/output/GLM_base_date_90Days_one_hot.csv",arr_len_sum=row_num,loop_times=50):
    y_1,w_1,x_1=readcsv(filename=filename)
    b_2=tweedie_model(y=y_1,w=w_1,x=x_1,arr_len_sum=arr_len_sum,loop_times=loop_times)
    #result=",".join(list(map(lambda x:str(x),b_2)))
    return b_2

#---------------possion_modle----------------------------------------------------------
def possion_model(y=np.zeros(1),w=np.zeros(1),x=np.zeros(1),arr_len_sum=row_num,loop_times=5):#用迭代法计算计算tweedie的参数和方差
    g=tf.Graph()
    with g.as_default():
        len=w.shape[0]
        p=tf.constant(1.5)
        y=tf.cast(tf.convert_to_tensor(np.array(y,dtype=float)),dtype=tf.float32)
        w=tf.cast(tf.convert_to_tensor(np.array(w,dtype=float)),dtype=tf.float32)
        x=tf.cast(tf.convert_to_tensor(np.array(x,dtype=float)),dtype=tf.float32)
        w_eye=tf.eye(len)*w

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=g,config=config) as sess:
        #计算出y的均值，用来替换缩小初始值的大小，避免出现非奇异矩阵
        init_constant=tf.log(tf.reduce_mean(y))
        b_1=np.random.normal(size=[arr_len_sum])/5.0#除了常数项以外,其余项比较小
        b_1[0]=sess.run(init_constant)
        b_1=tf.convert_to_tensor(b_1,dtype=tf.float32)
        i=1
        while True:
            u=tf.exp(tf.reduce_sum(x*b_1,axis=1))
            #计算2(dg(u)/du)=d(ln(u))/du=1/u^2,b(Q)的二次偏倒数
            G=tf.eye(len)*(1.0/u)#poisson模型和tweedie模型都采用ln链接
            det_gu_2=u
            W=tf.eye(len)*det_gu_2*w_eye
            x_t=tf.transpose(x)
            g_G=tf.reduce_sum(x*b_1,axis=1)+tf.reduce_sum(tf.multiply(G,y-u),axis=1)
            b_possion_2=tf.reduce_sum(tf.multiply(tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.matmul(x_t,W),x)),x_t),W),g_G),axis=1)
            if i>loop_times:
                break
            i=i+1
            b_1=b_possion_2
        b_2=sess.run(b_possion_2)
    sess.close()
    #初始化参数
    return  b_2

def train_possion_model2b_1(filename="/home/mapd/dumps/output/GLM_base_date_90Days_one_hot.csv",arr_len_sum=row_num,loop_times=50):
    y_1,w_1,x_1=readcsv(filename=filename)
    b_2=possion_model(y=y_1,w=w_1,x=x_1,arr_len_sum=arr_len_sum,loop_times=loop_times)
    #result=",".join(list(map(lambda x:str(x),b_2)))
    return b_2
