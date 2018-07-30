import tensorflow as tf

a=tf.constant([1,2,3,4,5,6])
b=tf.constant([0.5,0.5,0.5,0.5,0.5])
list_a=[]
with tf.Session() as sess:
     for i in range(5):
         list_a.append(a)
     print(sess.run(tf.reduce_min(tf.abs(tf.cast(tf.stack(list_a,axis=1),dtype=tf.float32)-b),axis=1)))