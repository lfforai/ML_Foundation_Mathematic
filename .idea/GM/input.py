import tensorflow as tf

#连续数据切割为分段数据，数据中点为新标记
def cut2pitch(value=tf.constant(0.0),std=tf.constant([0.0,100.0,150.0,300.0])):
    result=tf.constant(0.0)
    for e in range(std.shape[0]):
       if e>0:
          result=tf.cond(tf.reduce_all(tf.concat([tf.reshape(tf.less_equal(value,std[e]),[-1,1]),tf.reshape(tf.greater(value,std[e-1]),[-1,1])],axis=0))
                         ,lambda :(std[e-1]+std[e])/tf.constant(2.0),lambda :result)
       else:
          result=tf.cond(tf.equal(value,std[e]),lambda:(std[e]+std[e+1])/tf.constant(2.0),lambda :result)
    return result

#输出一组向量两个点之间的std=[0,100,150,300]中间值
def medie_tensor(std=tf.constant([0.0,100.0,150.0,300.0])):
    tf.get_default_graph()
    result=0
    std_1=tf.reshape(tf.slice(std,[1],[std.shape[0]-1]),[-1,1])
    std_2=tf.reshape(tf.slice(std,[0],[std.shape[0]-1]),[-1,1])
    std_zip=tf.concat([std_1,std_2],1)
    with tf.Session() as sess:
         result=sess.run(tf.map_fn(lambda x:tf.cast(x[0]+x[1],dtype=tf.float32)/tf.constant(2.0),std_zip))
    return result

with tf.Session() as sess:
     print(sess.run(cut2pitch(value=tf.constant(160.0),std=tf.constant([0.0,100.0,150.0,300.0]))))
     # print(medie_tensor(std=tf.constant([0.0,100.0,150.0,300.0])))
