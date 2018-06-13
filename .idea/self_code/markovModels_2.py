import tensorflow as tf
import numpy as np

#基础数据
#[1,1,0.5,0.5,6]=【状态，策略，转移到1，转移到2，报酬函数】
base_data=tf.constant([[1,1,0.5,0.5,6],[1,2,0.8,0.2,4],[2,1,0.4,0.6,-3],[2,2,0.7,0.3,-5]])
#状态空间
s=tf.constant([1,2])
#每个状态空间可以有不一样的策略空间A（i）！=A（j）
a=tf.constant([[1,2],[1,2]])
a_num=tf.constant([2,2])#每个s的action个数

#生成tensor的转移概率,再状态p(s_j|s_i,a_index)
def b2P_each(base_date=base_data,a_num=a_num,a_index=2,s_i=1,s_j=1):
    result=0
    tf.get_default_graph()
    index_s=tf.reduce_sum(tf.slice(a_num,[0],[s_i-1]),axis=0)#状态s块的
    with tf.Session() as sess:
        result=sess.run(base_data[index_s+a_index-1][1+s_j])
    return result

#生成概率矩阵
print("test_b2P_each:=",b2P_each(base_date=base_data,a_num=a_num,a_index=1,s_i=2,s_j=1))

#生成tensor的转移概率,再状态p(vector|s_i,a_index)#返回s_i条件下所有的
def b2P_vector(base_date=base_data,a_num=a_num,a_index=2,s_i=1):
    result=0
    len=a_num.shape[0]
    tf.get_default_graph()
    index_s=tf.reduce_sum(tf.slice(a_num,[0],[s_i-1]),axis=0)#状态s块的
    with tf.Session() as sess:
         result=sess.run(tf.slice(base_data[index_s+a_index-1],[2],[len]))
    return result

#生成概率矩阵
print("test_b2P_vector:=",b2P_vector(base_date=base_data,a_num=a_num,a_index=1,s_i=2))

#生成概率矩阵fun=[1,1],在状态1选择策略1，在状态2选择策略1
def b2p(base_date=base_data,fun=[1,1]):
    tf.get_default_graph()
    result=0
    s_len=fun.__len__()
    temp_np=np.zeros([fun.__len__(),2])
    #生成对应矩阵位置
    for i in range(fun.__len__()):
        temp_np[i]=[i+1,fun[i]]

    list_result=[]
    with tf.Session() as sess:
        for i in range(fun.__len__()):
            index_s=tf.reduce_sum(tf.slice(a_num,[0],[int(temp_np[i][0])-1]),axis=0)#状态s块的
            list_result.append([list(sess.run(tf.slice(base_data[index_s+int(temp_np[i][1])-1],[2],[s_len])))])
        result=sess.run(tf.concat(list_result,axis=0))
    return result

print("test_b2p:=",b2p(base_date=base_data,fun=[1,1]))

#输出报酬函数r向量
def  b2r(base_date=base_data,fun=[1,1]):
    tf.get_default_graph()
    result=0
    s_len=fun.__len__()
    temp_np=np.zeros([fun.__len__(),2])
    #生成对应矩阵位置
    for i in range(fun.__len__()):
        temp_np[i]=[i+1,fun[i]]

    list_result=[]
    with tf.Session() as sess:
        for i in range(fun.__len__()):
            index_s=tf.reduce_sum(tf.slice(a_num,[0],[int(temp_np[i][0])-1]),axis=0)#状态s块的
            list_result.append([list(sess.run(tf.slice(base_data[index_s+int(temp_np[i][1])-1],[2+s_len],[1])))])
        result=sess.run(tf.concat(list_result,axis=0))
    return result

print("test_b2r:=",b2r(base_date=base_data,fun=[1,1]))

#输出报酬函数r（i，a）单个
def  b2r_each(base_date=base_data,s_len=2,s_i=1,a_i=1):
    tf.get_default_graph()
    result=0
    with tf.Session() as sess:
        index_s=tf.reduce_sum(tf.slice(a_num,[0],[s_i-1]),axis=0)#状态s块的
        result=sess.run(tf.slice(base_data[index_s+a_i-1],[2+s_len],[1]))
    return result

print("test_b2r_each:=",b2r_each(base_date=base_data,s_len=2,s_i=2,a_i=1))

#输入每个状态的决策集，构造决策序列,只能构造每个状态策略都相同的
def active_road(active_vector=tf.zeros([4,4])):
    tf.get_default_graph()
    a=tf.constant(1)
    result=tf.zeros(5)
    len=active_vector.shape[0] #状态的个数
    len_colume=active_vector.shape[1] #每个状态下的action
    with tf.Session() as sess:
        for i in range(len-1):#遍历每个状态
            if i==0:
                s_a_matrix_x,s_a_matrix_y=tf.meshgrid(active_vector[i],active_vector[i+1])
                result=tf.concat([tf.reshape(s_a_matrix_x,[-1,1]),tf.reshape(s_a_matrix_y,[-1,1])],1)
            else:
                result_temp=tf.tile(result,[len_colume,1])
                temp_vector=tf.zeros(result.shape[0],dtype=tf.int32)
                s_a_matrix_x,s_a_matrix_y=tf.meshgrid(temp_vector,active_vector[i+1])
                # print(sess.run(s_a_matrix_y))
                result=tf.concat([result_temp,tf.reshape(s_a_matrix_y,[-1,1])],1)
                # print(sess.run(result))
        result=sess.run(result)
    return np.array(result)

print("test_active_road:=",active_road(active_vector=np.array([[1,2],[1,2]])))

#开始结束
def max_find_3_1(base_data=base_data,s=s,a=a,a_num=a_num):
    f=active_road(a)
    #f=[1,1]
    def init_v(f=f[0],base_data=base_data,s=s,a=a,a_num=a_num):
        tf.get_default_graph()
        f_num=f.shape[0]
        r=tf.reshape(tf.convert_to_tensor(b2r(base_date=base_data,fun=f)),[-1])
        p=tf.convert_to_tensor(b2p(base_date=base_data,fun=f))
        I=tf.eye(f_num)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            v=tf.reduce_sum(tf.multiply(tf.matrix_inverse(I-0.9*p),r),axis=1)
            return np.array(f),np.array(sess.run(v))

    #最优迭代
    def max_arg(v=[],base_data=base_data,s=s,a=a,a_num=a_num):
        tf.get_default_graph()
        s_len=v.__len__()#状态数
        v_tensor=tf.convert_to_tensor(v)
        f_now=[]
        v_now=[]
        with tf.Session() as sess:
            for i in range(s_len):
                max_value=-1
                max_action=-1
                a_len=a[i].shape[0]#在第i个状态下的
                for j in range(a_len):
                    # print("1:=",sess.run(tf.convert_to_tensor(b2P_vector(base_date=base_data,a_num=a_num,a_index=j+1,s_i=i+1))*v_tensor))
                    # print("2:=",b2r_each(base_date=base_data,s_len=s_len,s_i=i+1,a_i=j+1))
                    temp=sess.run(tf.convert_to_tensor(b2r_each(base_date=base_data,s_len=s_len,s_i=i+1,a_i=j+1))+tf.reduce_sum(0.9*tf.convert_to_tensor(b2P_vector(base_date=base_data,a_num=a_num,a_index=j+1,s_i=i+1))*v_tensor))
                    if temp>max_value:
                       max_value=temp
                       max_action=j+1
                    else:
                       pass
                f_now.append(max_action)
                v_now.append(max_value)
        return np.array(f_now),np.array(v_now)

    #迭代求解
    def equle(A=np.zeros(1),B=np.zeros(1)):
        result=True
        A_len=A.shape[0]
        for i in range(A_len):
            if A[i]!=B[i]:
               result=False
               break
        return result

    f_pre=0
    f_now=0
    init_v_value=0

    f_pre,init_v_value=init_v(f=f[0],base_data=base_data,s=s,a=a,a_num=a_num)
    while(True):
        f_now,v_now=max_arg(v=init_v_value,base_data=base_data,s=s,a=a,a_num=a_num)
        if  equle(f_now,f_pre):
            break
        else:
            f_pre,init_v_value=init_v(f=f_now,base_data=base_data,s=s,a=a,a_num=a_num)
    return f_now,v_now

print("max_find_3_1:=",max_find_3_1(base_data=base_data,s=s,a=a,a_num=a_num))