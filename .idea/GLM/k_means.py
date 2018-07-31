import csv
import tensorflow as tf
import numpy as np
import GLM_iteration
import random

#此聚类算法只用来做单因子聚类means_group_num预先设置的聚类种类
def k_means_core(u=tf.constant(0),y=tf.constant(0),w=tf.constant(0),means_group_num=5,row_size=4000,step=5):
    means_group=tf.convert_to_tensor(np.array(list(range(means_group_num)),dtype=float))#设置的分类个数
    row_size_id=tf.convert_to_tensor(np.array(list(range(row_size)),dtype=float))#样本序列号id
    u_vs_y=tf.concat([tf.reshape(u,[-1,1]),tf.reshape(y,[-1,1]),tf.reshape(w,[-1,1])],axis=1)

    #随机生成初始化的中心位置
    id_random=list(np.random.randint(0, high=row_size-1, size=row_size-1, dtype='l'))#生成分组初始值index
    id_random=random.sample(id_random,means_group_num)
    print("random id:",id_random)

    #init_u=[]
    temp_list=[]
    for i in range(means_group_num):
        temp_list.append(u[int(id_random[i])])
    init_u_a=tf.stack(temp_list,axis=0)

    #处理u=[u1,u2,u3,u4,u5]
    u_list=[]
    for i in range(means_group_num):
        u_list.append(u)
    init_u=tf.stack(u_list,axis=1)

    #计算距离u到每一组：means_group=[1,2,3,4,5],init_u_a=[u1,u2,u3,u4,u5]
    def map_func(x):
        def map_func_core(x,means_group=means_group):
            min_value=tf.constant(10000.0)
            min_index=tf.constant(-2.0)
            for i in range(means_group.shape[0]):
                min_value=tf.cond(tf.greater(x[i],min_value),lambda:min_value,lambda:x[i])
                min_index=tf.cond(tf.greater(x[i],min_value),lambda:min_index,lambda:tf.constant(i,dtype=tf.float32))
            return min_index
        return map_func_core(x,means_group=means_group)

    abs_temp=tf.abs(init_u-init_u_a)
    group_u=tf.map_fn(map_func,abs_temp)#
    #point=[u,y,group_id]
    point=tf.concat([u_vs_y,tf.reshape(group_u,[-1,1])],axis=1)#已经为每个u分好类别了
    with tf.Session() as sess:#计算出结果以后再进行筛选
         point_sess=sess.run(point)
         sess.close()
    #--------------init 结束------------

    #--计算每个分组内部的中心距离，也就是所有点的质心=avg（所有点）
    def re_divide(point_sess=point_sess,means_group=means_group,u_vs_y=u_vs_y):
        #计算point中每组的质心
        center_list_tensor=tf.constant(0)
        center_list=[]

        for i  in range(means_group.shape[0]):#按groupid进行过滤
            temp_tensor=np.array(list(filter(lambda x:x[3]==float(i),point_sess)))
            temp_center=tf.reduce_mean(tf.reshape(tf.slice(tf.convert_to_tensor(temp_tensor),[0,0],[-1,1]),[-1]))
            center_list.append(temp_center)

        center_list_tensor=tf.stack(center_list,axis=0)

        #重新计算新分组下的每个记录到质心的距离，并重新分组
        #处理u=[u1,u2,u3,u4,u5]
        u_list=[]
        for i in range(means_group_num):
            u_list.append(u)
        init_u=tf.stack(u_list,axis=1)

        #计算距离u到每一组：means_group=[1,2,3,4,5],init_u_a=[u1,u2,u3,u4,u5]
        def map_func(x):
            def map_func_core(x,means_group=means_group):
                min_value=tf.constant(10000.0)
                min_index=tf.constant(-2.0)
                for i in range(means_group.shape[0]):
                    min_value=tf.cond(tf.greater(x[i],min_value),lambda:min_value,lambda:x[i])
                    min_index=tf.cond(tf.greater(x[i],min_value),lambda:min_index,lambda:tf.constant(i,dtype=tf.float32))
                return min_index
            return map_func_core(x,means_group=means_group)

        abs_temp=tf.abs(init_u-center_list_tensor)
        group_u=tf.map_fn(map_func,abs_temp)#
        #point=[u,y,group_id]
        point=tf.concat([u_vs_y,tf.reshape(group_u,[-1,1])],axis=1)#已经为每个u分好类别了
        with tf.Session() as sess:#计算出结果以后再进行筛选
            point_sess=sess.run(point)
            sess.close()
        return point_sess
    #反复迭代
    for i in range(step):
        point_sess=re_divide(point_sess=point_sess,means_group=means_group,u_vs_y=u_vs_y)
    return point_sess

#基于tensorflow的k_means,model_tpye="posssion" or "tweedie"
def k_means(filename="",parameter=[],model_type="tweedie",step_model=15,step_k_means=50):
    #计算参数
    g=tf.Graph()
    #计算参数GLM模型参数
    if model_type=="tweedie":
       b_1=GLM_iteration.train_tweedie_model2b_1(filename="/home/mapd/dumps/output/GLM_base_date_90Days_one_hot.csv",arr_len_sum=GLM_iteration.row_num,loop_times=step_model)
    if model_type=="possion":
       b_1=GLM_iteration.train_possion_model2b_1(filename="/home/mapd/dumps/output/GLM_base_date_90Days_one_hot.csv",arr_len_sum=GLM_iteration.row_num,loop_times=step_model)
    print("GLM模型参数：",b_1)
    y,w,x=GLM_iteration.readcsv(filename="/home/mapd/dumps/output/GLM_base_date_90Days_one_hot.csv")
    row_size=y.shape[0]#总共有多少样本
    print("样本总计数：",row_size)
    with g.as_default():
         b_1_tf=tf.convert_to_tensor(b_1)
         y=tf.cast(tf.convert_to_tensor(np.array(y,dtype=float)),dtype=tf.float32)
         w=tf.cast(tf.convert_to_tensor(np.array(w,dtype=float)),dtype=tf.float32)
         x=tf.cast(tf.convert_to_tensor(np.array(x,dtype=float)),dtype=tf.float32)
         u=tf.exp(tf.reduce_sum(x*b_1,axis=1))
         point=k_means_core(u,y,w,means_group_num=5,row_size=row_size,step=step_k_means)
         import pandas as pd
         df=pd.DataFrame({"y_GLM":point[:,0],"y_real":point[:,1],"w":point[:,2],"group_id":point[:,3]})
         df.eval('sum_y_GLM=y_GLM*w',inplace=True)
         df.eval('sum_y_real=y_real*w',inplace=True)
         print("--------------系数----------------")
         print("系数：=",np.exp(np.array(b_1)))
         print("-----风险暴露数-----")
         print(df.groupby('group_id')['w'].sum())

         print("-----按分组数据差异差异-------")
         print(df.groupby('group_id')['sum_y_GLM'].sum()/df.groupby('group_id')['w'].sum())
         print(df.groupby('group_id')['sum_y_real'].sum()/df.groupby('group_id')['w'].sum())
    return 0
k_means(filename="",parameter=[],model_type="possion",step_model=15,step_k_means=50)
#print(k_means())


