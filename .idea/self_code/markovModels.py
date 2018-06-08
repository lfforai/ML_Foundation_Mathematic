
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
def max_find(f,p_matrix_list,v):
    s_num=f[0].shape[0]
    s_temp=np.zeros(s_num)
    for i in range(s_num):
        s_temp
    v=tf.reshape(tf.slice(tf.gather(tf.cast(r,dtype=tf.float32),[0,2]),[0,2],[2,1]),[-1])

    print(s_num)

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
max_find(f,p_matrix_list,r)


with tf.Session() as sess:
     pass

      # print("---------------------------------")
      # v=tf.reshape(tf.slice(tf.gather(tf.cast(r,dtype=tf.float32),[0,2]),[0,2],[2,1]),[-1])
      # p=tf.convert_to_tensor(tf.cast(p_matrix_list[0],dtype=tf.float32))
      # I=tf.eye(2)
      # result=tf.reduce_sum(tf.multiply(tf.matrix_inverse(I-0.9*p),v),axis=1)
      # print(sess.run(result))


      #print(sess.run(tf.reduce_sum(tf.multiply(tf.matrix_inverse(tf.subtract(tf.eye(2,dtype=tf.float32),0.9*tf.convert_to_tensor(tf.cast(p_matrix_list[1],dtype=tf.float32)))),,axis=1)))



      # c = tf.matmul(a, b)
      # print(sess.run(c))