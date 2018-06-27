import tensorflow as tf
import numpy as np
import random as random
#基于GLM的车联网保险数据定价模型

#---------------------------------第一部分 GLM模型中需要数据预处理-------------------------------
#1、tensorflow从csv文件中批量读取数据
def read_my_file_format(filename_queue,skip_header_lines=1):
    reader = tf.TextLineReader(skip_header_lines=skip_header_lines)
    key, value = reader.read(filename_queue)
    id,label1,label2,label3= tf.decode_csv(value, record_defaults=[[1.0],[1.0],[1.0],[1.0]])#['null']解析为string类型 ，[1]为整型，[1.0]解析为浮点。
    featrues=tf.stack([id,label1,label2,label3])
    return featrues


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True,seed=random.randint(1, 254))
    featrues_a= read_my_file_format(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch= tf.train.shuffle_batch(
        [featrues_a], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue,seed=random.randint(1, 254))
    return example_batch

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

#5、传入一组离散和非离散指标数据，对其中连续数据进行离散化处理使用cut2pitch()
def att_cut2pitch(filenames=['D:\\baoxian\\01.csv', 'D:\\baoxian\\02.csv', 'D:\\baoxian\\03.csv'],
                  batch_size=20,num_epochs=None,
                  std_list=[tf.constant([0.0,100.0,150.0,300.0]),tf.constant([1.0,2.0,3.0]),tf.constant([1,2,3,4])],
                  arr_mark=[1,1,1,0]):

    featrues_a=input_pipeline(filenames, batch_size=batch_size, num_epochs=num_epochs)
    result_N=tf.constant(0)
    with tf.Session() as sess:
        if std_list.__len__()!=featrues_a.shape[1]:
           print("需要离散化的指标个数与离散化区间（std_list）个数不一致！")
           exit()

        coord = tf.train.Coordinator()#创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)#启动QueueRunner，此时文件名队列已经进队

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

        for i in  range(4):
            x=tf.map_fn(lambda x:map_func(x_func=x,std_list=std_list,arr_mark=arr_mark),featrues_a)
            x_onehot=tf.map_fn(map_func_2,x)
            print(sess.run(x_onehot))
        coord.request_stop()
        coord.join(threads)
    return np.array(result_N)

#测试代码
# a=att_cut2pitch(filenames=['D:\\baoxian\\01.csv', 'D:\\baoxian\\02.csv', 'D:\\baoxian\\03.csv'], \
#                 batch_size=20,num_epochs=None, \
#                 std_list=[tf.constant([0.0,25.0,50.0,100.0,200.0]), \
#                               tf.constant([-1000000,0.0,1500,30000]), \
#                               tf.constant([-3000.0,-200.0,0.0,50.0,150.0,500.0]), \
#                               tf.constant(['A','B','C','D'])])
att_cut2pitch(filenames=['D:\\baoxian\\01.csv', 'D:\\baoxian\\02.csv', 'D:\\baoxian\\03.csv'],
              batch_size=2000,num_epochs=None,std_list=[tf.constant([0.0,25.0,50.0,100.0,200.0]), \
                                         tf.constant([-1000000,0.0,1500,30000]), \
                                         tf.constant([-3000.0,-200.0,0.0,50.0,150.0,500.0]), \
                                         tf.constant([1.0,2.0,3.0,4.0])],arr_mark=[1,1,1,0])
# with tf.Session() as sess:
#     std=tf.constant([1,2,3,4])
#
#     def map_in(x=tf.constant(0)):
#         def map_func(x=x,std_e=std_e):
#             return tf.cond(tf.equal(x,std_e),lambda :tf.constant(1),lambda :tf.constant(0))
#         return map_func(x,std_e)
#
#     std_e=tf.constant(3)
#     print(sess.run(tf.map_fn(map_in,std)))

#      a=[1,23,23]
#      i = tf.constant(0)
#      i_cpu=0
#      c = lambda i,i_cpu: tf.less(i, 10)
#      def b(i,i_cpu):
#            return tf.add(i, 1),i_cpu+1
#
#      r1,r2 = tf.while_loop(c, b, [i,i_cpu])
#      print(sess.run(r2))
#生成一个先入先出队列和一个Queuerunner，生成文件名队列
