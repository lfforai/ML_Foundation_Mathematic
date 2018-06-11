
#作者：罗锋
#目的：采用马尔科夫链解决自动编码的问题，这个文档验证一些迭代算法求解马尔可夫的可行性
#书本来源：马尔可夫决策过程理论与应用54页，算法3.1的例3.2
import numpy as np
import tensorflow as tf

#例子3.1，平稳马尔决策类
#策略集合，也就是一组路径,共有2个状态，f(i)=a,当状态确定时候
#f1=（1,1),当状态为1时候选者1策略吗，当状态为2时候选者1策略
#f2=(1,2),当状态为1时候选者1策略吗，当状态为2时候选者2策略
#f3=(2,1),当状态为1时候选者2策略吗，当状态为2时候选者1策略
#f4=(2,2),当状态为1时候选者2策略吗，当状态为2时候选者2策略
#当s状态空间和策略数较多的时候，组合出来的策略会由于排列组合的数量太大而无法用常规算法实现
#需要考虑第10章大规模数据问题的近似计算

#状态策略树
     #       策略1：状态1
     # 状态1
     #       策略2：状态1
     #
     #
     #       策略1：状态2
     # 状态2
     #
     #       策略2：状态2
     #

#输入每个状态的决策集，构造决策序列,只能构造每个状态策略都相同的
def active_road(active_vector=tf.zeros([4,4])):
     tf.get_default_graph()
     a=tf.constant(1)
     result=tf.zeros(5)
     len=active_vector.shape[0]
     len_colume=active_vector.shape[1]
     with tf.Session() as sess:
         for i in range(len-1):
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


#返回不同策略组合下的转移概率矩阵[1,1,0.5,0.5]状态1在行动1下，跳转到1和2状态的概率分别是0.5和0.5
def P_fun(a=tf.zeros(10),s=tf.zeros(10),f=np.zeros((10,10)),p=np.array([[1,1,0.5,0.5],[1,2,0.8,0.2],[2,1,0.4,0.6],[2,2,0.7,0.3]])):
    tf.get_default_graph()
    result=[]
    s_len=s.shape[0]#s.shape[0]==a.shape[0],状态个数
    f_len=f.shape[0]#策略路径的个数
    with tf.Session() as sess:
        # p=sess.run(p)
        for i in range(f_len):#每个策略路径构成一个转移矩阵
              temp_matrix=np.zeros((s_len,s_len))
              for j in range(f[i].shape[0]): #转移矩阵的行数
                    for x in p:
                        if x[0]==j+1 and x[1]==f[i][j]:
                           temp_matrix[j]=x[2:np.array(x).__len__()+1]
                        #矩阵行数对应       #对应策略
              result.append(temp_matrix)
    return f,result

#迭代计算最优结果,f是策略函数集合，p_matrix_list转移矩阵list,v报酬函数
def max_find_3_1(a,f,p_matrix_list,r):
    tf.get_default_graph()
    s_num=f[0].shape[0]#状态数
    f_num=f.shape[0]
    s_temp=np.zeros((s_num,2))#每个状态会选择一个action
    for i in range(s_num):
        s_temp[i]=[i+1,f[0][i]]

    #提取v的数值
    list_temp=[]
    v_temp=tf.Variable(tf.zeros([s_num]))
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print("r:=",sess.run(r))
        print("f=",f,f_num)
        print("s_temp:=",s_temp)
        print("p_matrix_list:=",p_matrix_list)
        r_h=sess.run(r)
        a_h=sess.run(a)

        for i in range(r_h.shape[0]):
            for j in range(s_temp.shape[0]):
                # print(r_h[i][0],s_temp[j][0],r_h[i][1],s_temp[j][1])
                if r_h[i][0]==s_temp[j][0] and r_h[i][1]==s_temp[j][1]:
                   # print("ok:=",r_h[i][0],s_temp[j][0],r_h[i][1],s_temp[j][1])
                   list_temp.append(i)

        # print("list_temp:=",list_temp)
        #初始化第一个状态
        r_m=tf.reshape(tf.slice(tf.gather(tf.cast(r,dtype=tf.float32),list_temp),[0,2],[2,1]),[-1])
        p=tf.convert_to_tensor(tf.cast(p_matrix_list[0],dtype=tf.float32))
        I=tf.eye(s_num)
        v=tf.reduce_sum(tf.multiply(tf.matrix_inverse(I-0.9*p),r_m),axis=1)
        v_1=tf.reduce_max(v,axis=0)

        print("v_1:=",sess.run(v_1))
        print("------------开始迭代-------------------")

        max_f_index_1=0#n次迭代中最大函数f的index

        #查找在f[i]满足，条件f[i][index]=a,其余f[i][j]=f_value=[j],j!=index,返回i
        def e_A_B_list(index,a,f_value=[],f=[]):
            result=-1
            for i in range(f.__len__()):
                true=0
                for j in range(list(f[0]).__len__()):
                    if (f[i][j]==f_value[j] and j!=index) or (j==index and f[i][index]==a):
                       true=true+1
                if true==list(f[0]).__len__():
                   result=i
                   break
        return result

        #循环迭代求最优的v值
        while(True):
            #计算使得v最大的f
            temp_value=0
            max_f_index_2=0
            list=[]
            for index_f in range(s_num): #每个状态的
               for e in range(list(a_h[index_f]).__len__()): #每个策略都进行一次遍历，找到最优值
                 #找到对应的f中的元素f[max]
                 max_f=e_A_B_list(e,a_h[index_f][e],f)
                 list_temp=[]
                 for i in range(s_num):
                     s_temp[i]=[i+1,f[max_f][i]]
                 # print("s_temp:=",s_temp)
                 for i in range(r_h.shape[0]):
                     for j in range(s_temp.shape[0]):
                         # print(r_h[i][0],s_temp[j][0],r_h[i][1],s_temp[j][1])
                         if r_h[i][0]==s_temp[j][0] and r_h[i][1]==s_temp[j][1]:
                            # print("ok:=",r_h[i][0],s_temp[j][0],r_h[i][1],s_temp[j][1])
                            list_temp.append(i)

                 # print("list_temp:=",list_temp)
                 r_m=tf.reshape(tf.slice(tf.gather(tf.cast(r,dtype=tf.float32),list_temp),[0,2],[2,1]),[-1])
                 p=tf.convert_to_tensor(tf.cast(p_matrix_list[max_f],dtype=tf.float32))
                 v_temp=tf.add(r_m,tf.reduce_sum(tf.multiply(0.9*p,v),axis=1))

                 v_2=tf.reduce_max(v_temp,axis=0)
                 print("v_temp：=",sess.run(v_temp))

                 if max_f==0:
                    temp_value=sess.run(v_2)

                 temp_media=sess.run(v_2)
                 if temp_media>temp_value and max_f!=0:
                    max_f_index_2=max_f
                    temp_value=temp_media

               print("n+1次最好的策略是：=",f[max_f_index_2],"n最好的策略是：=",f[max_f_index_1])
               print("n+1次最好值：=",sess.run(v_temp))
               if max_f_index_1==max_f_index_2:
                   print("break out!")
                   break
               else:
                   max_f_index_1=max_f_index_2

            #回到步骤2
            list_temp=[]
            for i in range(s_num):
                s_temp[i]=[i+1,f[max_f_index_1][i]]

            for i in range(r_h.shape[0]):
                for j in range(s_temp.shape[0]):
                    # print(r_h[i][0],s_temp[j][0],r_h[i][1],s_temp[j][1])
                    if r_h[i][0]==s_temp[j][0] and r_h[i][1]==s_temp[j][1]:
                        # print("ok:=",r_h[i][0],s_temp[j][0],r_h[i][1],s_temp[j][1])
                        list_temp.append(i)

            #初始化第一个状态
            r_m=tf.reshape(tf.slice(tf.gather(tf.cast(r,dtype=tf.float32),list_temp),[0,2],[2,1]),[-1])
            p=tf.convert_to_tensor(tf.cast(p_matrix_list[max_f_index_1],dtype=tf.float32))
            I=tf.eye(s_num)
            v=tf.reduce_sum(tf.multiply(tf.matrix_inverse(I-0.9*p),r_m),axis=1)
            # v_1=tf.reduce_max(v,axis=0)

        print("max f :=",max_f_index_2)
        print(sess.run(v_temp))
    return 1

#（状态，收益）矩阵
#状态向量
s=tf.constant([1,2]);

#决策函数序列f1,f2,f3,f4
a=tf.constant([[1,2],[1,2]])
f=active_road(a)

#依据f生成不同的概率转移矩阵
#转移到            状态1 状态2       状态1 状态2         状态1 状态2          状态1 状态2
#不同状态下选者的策略   [1 1]            [2 1]             [1 2]               [2 2]
#状态转移矩阵 p(1|1,1),p(1|2,1) p(1|1,2),p(2|2,1)   p(1|1,1),p(2|1,2)   p(1|1,2),p(2|2,2)

p=tf.constant([[1,1,0.5,0.5],[1,2,0.4,0.6],[2,1,0.8,0.2],[2,2,0.7,0.3]])

#收益报酬[1,1,6]表示i=1,a=1,状态在1,策略为1下的收益为6
r=tf.constant([[1,1,6],[1,2,4],[2,1,-3],[2,2,-5]])

#总报酬函数、最优值函数v（i，pi）

#第一次迭代，取方案
#迭代求解方程最优解法
f,p_matrix_list=P_fun(a,s,f)
max_find_3_1(f,p_matrix_list,r)


with tf.Session() as sess:
     pass