import tensorflow as tf
import numpy as np
import random as random
import os
import argparse
import sys
#基于GLM的车联网保险数据定价模型

#---------------------------------第一部分 GLM模型中需要数据预处理-------------------------------
#1、tensorflow从csv文件中批量读取数据
def read_my_file_format(filename_queue,skip_header_lines=1):
    reader = tf.TextLineReader(skip_header_lines=skip_header_lines)
    key, value = reader.read(filename_queue)
    loss_total,loss_ave,frequency,weight,age,car_type,car_year= tf.decode_csv(value, record_defaults=[[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]])#['null']解析为string类型 ，[1]为整型，[1.0]解析为浮点。
    featrues=tf.stack([age,car_type,car_year])    #输出指标x
    label=tf.stack([loss_total,loss_ave,frequency]) #输出总赔款、案均赔款、出险次数
    return label,weight,featrues                    #weight是每个样本的权重


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True,seed=random.randint(1, 254))
    label_a,weight_batch,featrues_a=read_my_file_format(filename_queue)
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

#2、连续数据切割为分段数据，数据中点为新标记
def cut2pitch(value=tf.constant(0.0),std=tf.constant([0.0,100.0,150.0,300.0]),arr=0):
    result=tf.constant(0.0)
    if arr==1:
        for e in range(std.shape[0]):
            if e>0:
                result=tf.cond(tf.reduce_all(tf.concat([tf.reshape(tf.less_equal(value,std[e]),[-1,1]),tf.reshape(tf.greater(value,std[e-1]),[-1,1])],axis=0))
                               ,lambda :(std[e-1]+std[e])/tf.constant(2.0),lambda :result)
            else:
                result=tf.cond(tf.equal(value,std[e]),lambda:(std[e]+std[e+1])/tf.constant(2.0),lambda :result)
    else:
        result=value
    return result

#3、输出一组向量两个点之间的std=[0,100,150,300]中间值
def medie_tensor(std=tf.constant([0.0,100.0,150.0,300.0])):
    result=tf.constant(0)
    std_1=tf.reshape(tf.slice(std,[1],[std.shape[0]-1]),[-1,1])
    std_2=tf.reshape(tf.slice(std,[0],[std.shape[0]-1]),[-1,1])
    std_zip=tf.concat([std_1,std_2],1)
    with tf.Session() as sess:
        result=tf.map_fn(lambda x:tf.cast(x[0]+x[1],dtype=tf.float32)/tf.constant(2.0),std_zip)
    return result

#4、输出一组list（每个元素表示一个指标的分段区间）
def medie_tensor_list(std_list=[tf.constant([0.0,100.0,150.0,300.0]),tf.constant([1.0,2.0,3.0]),tf.constant(['A','B','C','D'])],arr_mark=[1,1,1,0]):
    result=[]
    assert(std_list.__len__()==arr_mark.__len__())
    for e in range(std_list.__len__()):
        if arr_mark[e]==0:
            result.append(std_list[e])
        else:
            result.append(medie_tensor(std_list[e]))
    return result

#tweedie分布拟合总赔款模型 w是需要拟合的变量
def tweedie_model(y,weight,x,w,p=tf.constant(1.5)):
    y_total_loss=tf.slice(y,[0,0],[-1,1])#确定y
    u=tf.exp(tf.reduce_sum(tf.multiply(x,w)))
    theta=(-1.0)/(p-1.0)*tf.pow(u,(-1.0)*(p-1.0))
    K_theta=(-1.0)/(p-2.0)*tf.pow(((-(p-1.0))*theta),(p-2.0)/(p-1.0))
    loss=tf.reduce_mean(tf.multiply(weight,tf.multiply(y_total_loss,theta)-K_theta),axis=0)
    return loss

#gamma分布拟合案均赔款模型
def gamma_model(y=tf.constant(0),weight=tf.constant(0),x=tf.constant(0)):
    y_total_loss=tf.slice(y,[0,0],[-1,1])#确定y
    # print("y_total_loss:=",y_total_loss)
    #计算loss
    u=tf.exp(tf.reduce_sum(tf.multiply(x,w)))
    theta=(-1.0)/(p-1.0)*tf.pow(u,(-1.0)*(p-1.0))
    K_theta=(-1.0)/(p-2.0)*tf.pow(((-(p-1.0))*theta),(p-2.0)/(p-1.0))
    loss=tf.reduce_mean(tf.multiply(weight,tf.multiply(y_total_loss,theta)-K_theta))
    return loss

#Poisson拟合索赔次数
def Poisson_model(y,weight,x,w):
    y_total_time=tf.slice(y,[0,2],[-1,1])#确定y
    u=tf.exp(tf.reduce_sum(tf.multiply(x,w)))
    loss=tf.reduce_mean(tf.multiply(tf.multiply(tf.log(u),y_total_time)-u,weight),axis=0)
    return loss

#5、开始模型测算
def main(_):
    print(FLAGS)
    #一、参数设置和文件路径
    filenames=['./data.csv', './data1.csv', './data2.csv']
    batch_size=50
    num_epochs=None
    std_list=[tf.constant([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]), \
              tf.constant([1.0,2.0,3.0,4.0]), \
              tf.constant([1.0,2.0,3.0,4.0])]

    arr_mark=[0,0,0]#每个指标是否为需要离散化的指标
    if_constant=True#是否需要常数项
    learning_rate=0.01

    arr_len_sum=0#指标的总长度
    for e in std_list:
        arr_len_sum=arr_len_sum+e.shape[0]
    if if_constant==True:
        arr_len_sum=arr_len_sum+1#增加一个常数项

    gama_mark=0;#拟合案均赔款
    poisson_mark=0;#拟合出险次数
    tweedie_mark=1;#拟合总赔款

    #二、设置服务器
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,config=config)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" %FLAGS.task_index,
                cluster=cluster)):
            y_batch,weight_batch,x_batch=input_pipeline(filenames, batch_size=batch_size, num_epochs=num_epochs)
            if std_list.__len__()!=x_batch.shape[1]:
                print("需要离散化的指标个数与离散化区间（std_list）个数不一致！")
                exit()

            #----------------------------1、开始指标进入和切分为[1,0,0,0]预处理部分------------------------------------------------
            def map_func(x_func=tf.constant(1),std_list=std_list,arr_mark=[]):
                temp=[]
                for j in range(x_func.shape[0]):
                    temp.append(cut2pitch(x_func[j],std_list[j],arr_mark[j]))
                return tf.stack(temp,axis=0)

            #把离散化指标转化为GLM模型的[1,0,0]这类格式
            #比如指标[A,B,C],[A:1,0,0][B:0,1,0][C:0,0,1],arr_mark:1表示它是连续区间需要切割区间：0表示本身就是离散变量不用切割区间
            #把分段区间标准化
            x_std=medie_tensor_list(std_list=std_list,arr_mark=arr_mark)

            #把每一个指标的每个指标，转换为标准的【0，1】
            def map_func_2(x_n=tf.constant([1.25,1.575,2.50,3.00])):
                def map_func_e(x=x_n,std_list=x_std):
                    def map_func_in(value=tf.constant(0.0)):
                        def in_func(value=value,temp=temp_mid):
                            result=tf.cond(tf.equal(value,temp),lambda :tf.constant(1.0),lambda :tf.constant(0.0))
                            return result
                        return in_func(value=value,temp=temp_mid)
                    result=[]
                    #遍历每一个指标[1.25,1.575,2.50,3.00]
                    for i in range(x.shape[0]):
                        temp_mid=x[i]
                        result.append(tf.map_fn(map_func_in,std_list[i]))
                    result=tf.concat(result,axis=0)
                    return result
                return map_func_e(x=x_n,std_list=x_std)

            x=tf.map_fn(lambda x:map_func(x_func=x,std_list=std_list,arr_mark=arr_mark),x_batch)

            if if_constant==True:#在指标中添加一个常数项b0
                x_onehot=tf.map_fn(map_func_2,x)
                def map_func_add_one(x=tf.constant(0)):
                    return tf.concat([tf.constant([1.0]),x],axis=0)
                x_onehot=tf.map_fn(map_func_add_one,x_onehot)
            else:
                x_onehot=tf.map_fn(map_func_2,x)
            #----------------------------结束指标进入和切分为[1,0,0,0]预处理部分------------------------------------------------

            #----------------------------2、拟合模型--------------------------------------------------------------------------
            hooks=[tf.train.StopAtStepHook(last_step=1000000)]
            global_step = tf.contrib.framework.get_or_create_global_step()

            # The StopAtStepHook handles stopping after running given steps.
            w_tweedie=tf.get_variable(name='tweedie_var', shape=[arr_len_sum], initializer=tf.random_normal_initializer(mean=0, stddev=1))
            loss_tweedie=(-1.0)*tweedie_model(y_batch,weight_batch,x_onehot,w_tweedie,p=tf.constant(1.5))
            optimizer_tweedie = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tweedie,global_step=global_step)
            accuracy_tweedie=tf.sqrt(tf.reduce_mean(tf.pow(tf.exp(tf.reduce_sum(tf.multiply(x_onehot, w_tweedie)))-tf.slice(y_batch,[0,0],[-1,1]),2)))

            w_poisson=tf.get_variable(name='poisson_var', shape=[arr_len_sum], initializer=tf.random_normal_initializer(mean=0, stddev=1))
            loss_poisson=(-1.0)*Poisson_model(y_batch,weight_batch,x_onehot,w_poisson)
            optimizer_poisson=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_poisson,global_step=global_step)
            accuracy_poisson=tf.sqrt(tf.reduce_mean(tf.pow(tf.exp(tf.reduce_sum(tf.multiply(x_onehot,w_poisson)))-tf.slice(y_batch,[0,2],[-1,1]),2)))
            #----------------------------2、结束拟合模型-----------------------------------------------------------------------


            # The StopAtStepHook handles stopping after running given steps.
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   checkpoint_dir="c:/baoxian/",
                                                   hooks=hooks) as mon_sess:
                # mon_sess.run(init)
                while not mon_sess.should_stop():
                    coord = tf.train.Coordinator()#创建一个协调器，管理线程
                    threads = tf.train.start_queue_runners(sess=sess,coord=coord)#启动QueueRunner，此时文件名队列已经进队
                    for i in  range(40000):
                        if tweedie_mark==1:
                            sess.run(optimizer_tweedie)
                        if poisson_mark==1:
                            sess.run(optimizer_poisson)
                        if  i%100==0:
                            if tweedie_mark==1:
                                print("accuracy_tweedie_:=",sess.run(accuracy_tweedie))
                            if poisson_mark==1:
                                print("accuracy_poisson_:=",sess.run(accuracy_poisson))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="localhost:2222,localhost:2223",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="localhost:2224,localhost:2225,localhost:2226",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="ps",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)