import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    a=tf.constant([[2,1],[3,4]])
    b=tf.constant([[5,6],[7,8]])
    c=tf.constant([[9,10],[11,12]])
    d=tf.concat([a,b,c],axis=0)
    print(d.shape)
    e=tf.stack([a,b,c],axis=0)
    # print(sess.run(tf.fill([5,2],[2,3])))
    print(e.shape)
    print(sess.run(d))
    print(sess.run(e))
    print("---------------")
    a=tf.constant([[[1],[1]]])
    b=tf.constant([[[5],[6]]])
    c=tf.constant([[[11],[12]]])
    print(sess.run(tf.stack([a,b,c],axis=1)))
    print(tf.stack([a,b,c],axis=1))
    print(a)
    #总结 concat和stack差异:
    #1、stack会增加tensor的维度，而concat不会，stack相当于[a,b]链接再axis=n，第n维度tensor扩展2，相当于[a,b,c]链接再axis=n，第n维度tensor扩展3
                                             #concat维度不变，但是会扩展的数值，数值等于[a,b]链接再该维度下所有元素的个数之和
    #2、concat可以要求最后一个维度不相等如果ax=0

#tf.fill()
    print("---------------------------------------------------------")
    print(sess.run(tf.fill([2,2],1)))

#onehot()
    print("-------------------tf.one_hot-----------------------------")
    print(sess.run(tf.one_hot(tf.constant([0,1,2]),depth=3)))
    print(sess.run(tf.one_hot(tf.constant([[0,1,2],[0,1,2]]),depth=3)))

#tf.expand_dims()
    print("-------------------tf.expand_dims----------------------------------")
    print("tf.expand_dims:=",sess.run(tf.expand_dims(tf.constant([[2,2],[2,2]]),dim=0)))
    print("tf.expand_dims:=",sess.run(tf.expand_dims(tf.constant([[2,2],[2,2]]),dim=1)))
    print("tf.expand_dims:=",sess.run(tf.expand_dims(tf.constant([[2,2],[2,2]]),dim=2)))

#tf.unstack()
    print("-------------------------------------------------------------------")

