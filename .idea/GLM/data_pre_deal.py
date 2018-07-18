# Note: The following example should be run in the same directory as
# map_jdbc.py and mapdjdbc-1.0-SNAPSHOT-jar-with-dependencies.jar
import mapd_jdbc
import pandas
import matplotlib.pyplot as plt

#在数据处理阶段应用mapd的gpu数据库，用标准sql语言对数据进行处理
#作者：罗锋
import tensorflow as tf
import numpy as np
import random as random
import os
import argparse
import sys
from tensorflow.python.client import device_lib as _device_lib


#功能描述：对指标进行排序\Mappartion处理数据\
def is_gpu_available(cuda_only=True):
    if cuda_only:
        return any((x.device_type == 'GPU')
                   for x in _device_lib.list_local_devices())
    else:
        return any((x.device_type == 'GPU' or x.device_type == 'SYCL')
                   for x in _device_lib.list_local_devices())

def total_menory():
    menory_sum=0
    for x in _device_lib.list_local_devices():
        pass

print("是否可以使用GPU：",is_gpu_available(cuda_only=False))


#将csv文件导入到mapd中去进行处理
dbname = 'mapd'
user = 'mapd'
host = 'localhost:9091'
password = 'HyperInteractive'

#一、参数设定
import_path_dir="/home/mapd/dumps/" #导入数据文件目录
#一类文件有统一表格格式
imput_file_style=[#导入车基数据
                  {"input_file_path":import_path_dir,
                   "input_file_name":
                              ["month_201606.csv","month_201607.csv","month_201608.csv","month_201609.csv",
                              "month_201610.csv","month_201611.csv"],
                   "mapd_talbe_type":
                   "("+
                   "carid TEXT  NOT NULL ENCODING DICT,"+
                   "c2_date SMALLINT,"+
                   "mileage FLOAT,"+
                   "fuel  FLOAT,"+
                   "duration INT,"+
                   "maxspeed FLOAT,"+
                   "ignition SMALLINT,"+
                   "flameout SMALLINT,"+
                   "insertNum SMALLINT,"+
                   "accelerateTimes SMALLINT,"+
                   "decelerateTimes SMALLINT,"+
                   "sharpTurnTimes SMALLINT,"+
                   "connectNum SMALLINT,"+
                   "isreplace SMALLINT,"+
                   "maxsatellite SMALLINT,"+
                   "lVoltage FLOAT,"+
                   "hVoltage FLOAT,"+
                   "isdrive SMALLINT,"+
                   "isfatigue SMALLINT,"+
                   "ishighspeed SMALLINT,"+
                   "pulloutTimes INT,"+
                   "isnonlocal SMALLINT,"+
                   "isnightdrive SMALLINT,"+
                   "vbigtime INT,"+
                   "maxmileage FLOAT,"+
                   "ismissing SMALLINT,"+
                   "interactTimes SMALLINT,"+
                   "totalmile FLOAT,"+
                   "fee FLOAT,"+
                   "onoff FlOAT,"+
                   "traces SMALLINT,"+
                   "deltrace SMALLINT,"+
                   "uptime TIMESTAMP)"
                  }
                  #导入carid和vin对应码
                  ,
                  {"input_file_path":import_path_dir,
                   "input_file_name":
                       ["carid2vin_2.csv","carid2vin.csv"],
                   "mapd_talbe_type":
                       "("+
                       "carid TEXT  NOT NULL ENCODING DICT,"+
                       "vin   TEXT"+")"
                   }
                  #导入赔付数据
                  ,
                  {"input_file_path":import_path_dir,
                   "input_file_name":
                       ["pei.csv"],
                   "mapd_talbe_type":
                       "("+
                       "vin TEXT  NOT NULL ENCODING DICT,"+
                       "happen TIMESTAMP,"
                       "c3_over TIMESTAMP,"
                       "claims INT,"
                       "c5_get TIMESTAMP"+")"
                   }
                  ]
#删除的table名字
delete_filename_list=[]

# 二、mapd数据库链接设置
mapd_con = mapd_jdbc.connect(
    dbname=dbname, user=user, host=host, password=password)
mapd_cursor = mapd_con.cursor()

#三、不同功能的数据库操函数
#1\删除数据
def pitch_delete(delete_file_list=delete_filename_list):
    query_delete="DROP table"
    for e in delete_file_list:
        query_delete_temp=str(query_delete+" "+e)
        mapd_cursor.execute(query_delete_temp)
    print("clear table：="+str(delete_file_list))

#2\在mapd上创建表格
#备注：mapd用户需要使用读入的数据需要放到/home/mapd文件夹下，不然会报错permission错误
def create_csv2table(imput_file_style=imput_file_style,if_clear=True,if_input=[0,0,1]):
    create_talbe_mapd="CREATE TABLE IF NOT EXISTS "
    i=0
    for e in imput_file_style:
      if if_input[i]==1:
        #每个e都是一组相同类型的表
        for value in e["input_file_name"]:
            value_temp=e["input_file_path"]+value
            query_table=create_talbe_mapd+str(value).replace(".csv","")+e["mapd_talbe_type"]
            print("creat table:",query_table)
            mapd_cursor.execute(query_table)
            if  if_clear==True:
                print("need to TRUNCATE TABLE "+str(value).replace(".csv",""))
                mapd_cursor.execute("TRUNCATE TABLE "+str(value).replace(".csv",""))
            print("input cvs to table:",str(value_temp))
            print("copy "+str(value).replace(".csv","")+" from "+"\'"+value_temp+"\'")
            mapd_cursor.execute("copy "+str(value).replace(".csv","")+" from "+"\'"+value_temp+"\'")
            print("-----------------------------------------------------------------------")
      i=i+1

#3、sql代码 create table  month_201606_vin2 as (select m.*,c.vin from month_201606 m,carid2vin_2 c where m.carid =c.carid)




# 三、执行数据预计处理
if len(delete_filename_list)>0:
   pitch_delete()
create_csv2table()

# Get the results

# results = mapd_cursor.fetchall()

# Make the results a Pandas DataFrame
#
# df = pandas.DataFrame(results)
# print("df:=",df)
# print(type(df[1]))

# Make a scatterplot of the results

# plt.scatter(df[1], df[2])
#
# plt.show()
