import csv
import tensorflow as tf
import numpy as np
import GLM_iteration
import random
import data_pre_deal

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
def k_means(filename="",parameter=[],model_type="tweedie",day_mark="30Days",step_model=15,step_k_means=50):
    #计算参数
    #读取有多少个参数
    info_n={"att_modle_risk":"","att_model_all":"","att_modle_nh":"","att_modle_base":"","pei":"","weight":"","att_num":"","day_mark":"","att_car":""}
    f = open("/home/mapd/dumps/output/att_name_"+day_mark+".txt")             # 返回一个文件对象
    line = f.readline()
    print(line.split(":")[0])
    info_n[line.split(":")[0]]=line.split(":")[1].replace("\n","")
    print(info_n[line.split(":")[0]])
    i=1
    while line and i<9:
        line = f.readline()
        print(line.split(":")[0])
        info_n[line.split(":")[0]]=line.split(":")[1].replace("\n","")
        print(info_n[line.split(":")[0]])
        i=i+1
    f.close()
    colums_num=int(info_n["att_num"])
    print("参数个数:=",colums_num)

    g=tf.Graph()
    #计算参数GLM模型参数
    if model_type=="tweedie":
       b_1=GLM_iteration.train_tweedie_model2b_1(filename="/home/mapd/dumps/output/GLM_base_date_"+info_n["day_mark"]+"_one_hot.csv",arr_len_sum=colums_num,loop_times=step_model)
    if model_type=="possion":
       b_1=GLM_iteration.train_possion_model2b_1(filename="/home/mapd/dumps/output/GLM_base_date_"+info_n["day_mark"]+"_one_hot.csv",arr_len_sum=colums_num,loop_times=step_model)

    #把参数与属性名称合并到一起，方便输出查看
    b_1_bnk=b_1
    b_1=np.exp(np.array(b_1))#计算glm模型的系数
    att_name=list(["constant"])+list(info_n["att_model_all"].replace("\n","").split(","))
    # print(info_n["att_model_all"])
    att_value=[1.0]*(info_n["att_model_all"].replace("\n","").split(",").__len__()+1)
    att_name_vs_value_all=dict(zip(att_name,att_value))
    # print(att_name_vs_value_all)

    att_name_vs_value_nh=dict(zip(info_n["att_modle_nh"].replace("\n","").split(","),b_1))
    for e in att_name_vs_value_nh: #遍历所有拟合属性
        att_name_vs_value_all[e]=att_name_vs_value_nh[e]
    print("GLM模型参数：",att_name_vs_value_all)

    y,w,x=GLM_iteration.readcsv(filename="/home/mapd/dumps/output/GLM_base_date_"+info_n["day_mark"]+"_one_hot.csv")
    row_size=y.shape[0]#总共有多少样本
    print("样本总计数：",row_size)

    with g.as_default():
         b_1_tf=tf.convert_to_tensor(b_1)
         y=tf.cast(tf.convert_to_tensor(np.array(y,dtype=float)),dtype=tf.float32)
         w=tf.cast(tf.convert_to_tensor(np.array(w,dtype=float)),dtype=tf.float32)
         x=tf.cast(tf.convert_to_tensor(np.array(x,dtype=float)),dtype=tf.float32)
         u=tf.exp(tf.reduce_sum(x*b_1_bnk,axis=1))
         point=k_means_core(u,y,w,means_group_num=5,row_size=row_size,step=step_k_means)
         import pandas as pd
         df=pd.DataFrame({"y_GLM":point[:,0],"y_real":point[:,1],"w":point[:,2],"group_id":point[:,3]})
         df.eval('sum_y_GLM=y_GLM*w',inplace=True)
         df.eval('sum_y_real=y_real*w',inplace=True)
         # print("--------------系数----------------")
         # print("GLM模型系数=：",att_name_vs_value_all)
         # print("-----风险暴露数-----")
         level=["A","B","C","D","E"]
         df_risk=df.groupby('group_id')['w'].sum()
         # print("-----按分组数据差异差异-------")
         df_y_GLM=df.groupby('group_id')['sum_y_GLM'].sum()/df.groupby('group_id')['w'].sum()
         df_y_real=df.groupby('group_id')['sum_y_real'].sum()/df.groupby('group_id')['w'].sum()
         result=pd.DataFrame({"real_y":np.array(list(df_y_real)),"GLM_y":np.array(list(df_y_GLM)),"car_num":np.array(list(df_risk))})
         result=result.sort_values(by='real_y')
         result.insert(0,"等级",level)
         # 车辆数结果

         #按标准格式输出系数
         o_list=info_n["att_car"].split(",")#mileage,maxspeed,a,d,isf,ish,isn
         o_list=dict(list(zip(o_list,[""]*o_list.__len__())))
         # print("o_list:",o_list)

         #把att_name_vs_value_all中key设置为最大长度，并且规定value_float的长度为5位小数
         max_len=0
         for e in att_name_vs_value_all:
             now_len=str(e).__len__()
             if max_len<now_len:
                max_len=now_len

         #替换所有的att_name_vs_value_all中的key的len长度为最大长度，不足位补“ ”
         key_list=[]
         for e in att_name_vs_value_all:
             key_list.append(e+" "*(max_len-str(e).__len__()))
         value_list=[" "]*key_list.__len__()
         new_att_name_vs_value_all=dict(list(zip(key_list,value_list)))
         #复制round(value,6)到new_att_name_vs_value_all中去
         for e in new_att_name_vs_value_all:
             temp=str(round(float(att_name_vs_value_all[e.replace(" ","")]),6))
             new_att_name_vs_value_all[e]=temp+"0"*('1.121628'.__len__()-temp.__len__())
         att_name_vs_value_all=new_att_name_vs_value_all

         #把相同的数据合并到
         max_list_len=0
         for  e in o_list:
              temp_list=[]
              for b in att_name_vs_value_all:
                  if str(b.replace(" ","")).split("_")[0].__eq__(str(e).replace(" ","")):
                     temp_list.append(b+":"+att_name_vs_value_all[b]+"|")
              o_list[e]=temp_list
              # print(temp_list.__len__())
              if temp_list.__len__()>max_list_len:
                 max_list_len=temp_list.__len__()

         # print("o_list:",o_list)
         # print("max_list_len",max_list_len)

         #计算o_list的value中最大的list长度，对不足长度的key的value=list中添加“-”
         stand_len=('1.121628'.__len__()+1+max_len)#|之间的标准长度
         add_item="-"*stand_len+'|'#添加“-------------”
         for e in o_list:
             if o_list[e].__len__()!=max_list_len:
                o_list[e]=o_list[e]+[add_item]*(max_list_len-o_list[e].__len__())
         # print(o_list)

         #打印结果
         text="一、GLM模型系数("+day_mark+")\n"+"指标说明：mileage-公里数,maxspeed-最大速度,a-急加速,d-急减速,isf-疲劳驾驶/驾驶天数占比,ish-高速行驶/驾驶天数占比,isn-夜间驾驶/驾驶天数占比\n\n"
         #因子名称

         for e in o_list:
             text=text+" "*int((stand_len-e.__len__())/2)+e+" "*int((stand_len-int((stand_len-e.__len__())/2)-e.__len__()))+"|"

         text=text+"\n"

         for i in range(max_list_len):
             temp_text=""
             for e in o_list:
                 temp_text=temp_text+str(o_list[e][i])
             text=text+temp_text+"\n"
         text=text+"二、各风险等级车辆\n"+str(result)
         print(text)
    return 0
k_means(filename="",parameter=[],model_type="tweedie",day_mark="30Days",step_model=5,step_k_means=5)
#print(k_means())


