
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

#输入每个状态的决策集，构造决策序列
def active_road(active_vector=tf.zeros([4,4])):
    g=tf.Graph()
    with g.as_default():
         a=tf.constant(1)
         result=tf.zeros(0)
         len=active_vector.shape[0]
         for i in range(len-1):
             if i==0:
                s_a_matrix_x,s_a_matrix_y=tf.meshgrid(active_vector[i],active_vector[i+1])
                result=tf.concat([tf.reshape(s_a_matrix_x,[-1,1]),tf.reshape(s_a_matrix_y,[-1,1])],1)
             else:
                temp_vector=tf.zeros(result.shape[0])
                s_a_matrix_x,s_a_matrix_y=tf.meshgrid(temp_vector,active_vector[i+1])
                result=tf.concat([tf.reshape(s_a_matrix_x,[-1,1]),tf.reshape(s_a_matrix_y,[-1,1])],1)

    sess=tf.Session(graph=g)
    print(sess.run(a))
    sess.close()
    return 1

#（状态，收益）矩阵
#状态向量
s=tf.constant([3,4]);

#策略向量
a=tf.constant([1,2]);

#决策函数序列f1,f2,f3,f4
s_a_matrix_x,s_a_matrix_y=tf.meshgrid(s,a)
f=tf.concat([tf.reshape(s_a_matrix_x,[4,1]),tf.reshape(s_a_matrix_y,[4,1])],1)
# [[1 1]
#  [2 1]
# [1 2]
# [2 2]]

#依据f生成不同的概率转移矩阵
#转移到            状态1 状态2       状态1 状态2         状态1 状态2          状态1 状态2
#不同状态下选者的策略   [1 1]            [2 1]             [1 2]               [2 2]
#状态转移矩阵 p(1|1,1),p(1|2,1) p(1|1,2),p(2|2,1)   p(1|1,1),p(2|1,2)   p(1|1,2),p(2|2,2)
p=tf.constant([[0.5,0.5],[0.4,0.6],[0.8,0.2],[0.7,0.3]])

#收益报酬
r=tf.constant([6,4,-3,-5])

#总报酬函数、最优值函数v（i，pi）

#第一次迭代，取方案
#迭代求解方程最优解法
with tf.Session() as sess:

     print(np.array(sess.run(tf.meshgrid(s,a))))

active_road(tf.constant([[1,2],[3,4]]))