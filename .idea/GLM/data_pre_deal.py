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
#（一）导入车基数据
import_path_dir="/home/mapd/dumps/"#入数据文件目录
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
                       "vin   TEXT  NOT NULL ENCODING DICT"+")"
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

#（二）
#为month_2016×中的每条记录打标所属日期,"2016-06-xx"="2016-06-01"+month_201607.day-1
select_columns="carid,c2_date,mileage,fuel,duration,maxspeed,accelerateTimes,decelerateTimes,sharpTurnTimes,isdrive,isfatigue,ishighspeed,pulloutTimes,isnonlocal,isnightdrive,vbigtime,maxmileage,totalmile "
date_list_style={"database":["month_201606","month_201607","month_201608","month_201609",
                       "month_201610","month_201611"],
           "first_date":["\'2016-06-01\'","\'2016-07-01\'","\'2016-08-01\'","\'2016-09-01\'","\'2016-10-01\'","\'2016-11-01\'"],
           "colume_name":"y_m_d",#记录日期新指标的名称
           "select_columns":select_columns
           }

#每30天划分为一个风险暴露区间
select_columns="y_m_d,carid,c2_date,mileage,fuel,duration,maxspeed,accelerateTimes,decelerateTimes,sharpTurnTimes,isdrive,isfatigue,ishighspeed,pulloutTimes,isnonlocal,isnightdrive,vbigtime,maxmileage,totalmile "
when_case_style_30day={"database":["month_201606","month_201607","month_201608","month_201609",
                                    "month_201610","month_201611"],
                       "time_range":[("\'2016-06-01 00:00:00\'","\'2016-6-30 00:00:00\'"),("\'2016-07-01 00:00:00\'","\'2016-07-30 00:00:00\'"),("\'2016-08-01 00:00:00\'","\'2016-8-30 00:00:00\'"),("\'2016-09-01 00:00:00\'","\'2016-9-30 00:00:00\'"),("\'2016-10-01 00:00:00\'","\'2016-10-30 00:00:00\'"),("\'2016-11-01 00:00:00\'","\'2016-11-30 00:00:00\'")],
                       "month_num":['6','7','8','9','10','11'],
                       "select_columns":select_columns
                      }#按一定时间区间对数据进行切分，划分为不同的风险暴露

#把赔款数据匹配到每个车基数据上去 LFV2A1159C3567437 17 LVSHFFAL1FS337046 16
pei2car_style={"database":["pei"]}

#数据校验的规则
data_check_rule={"database":["month_201606","month_201607","month_201608","month_201609",
                                   "month_201610","month_201611"],
                }#按一定时间区间对数据进行切分，划分为不同的风险暴露


#（三）需要删除的table名字
delete_filename_list=["month_201606","month_201607","month_201608","month_201609",
                      "month_201610","month_201611"]
# delete_filename_list=[]
delete_filename_list=["carid2vin","carid2vin_2"]
delete_filename_list=["pei_of_carid"]
# delete_filename_list=[]

# 二、mapd数据库链接设置
mapd_con = mapd_jdbc.connect(
    dbname=dbname, user=user, host=host, password=password)
mapd_cursor = mapd_con.cursor()

#三、不同功能的数据库操函数
#1、删除数据
def pitch_delete(delete_file_list=delete_filename_list):
    query_delete="DROP table"
    for e in delete_file_list:
        query_delete_temp=str(query_delete+" "+e)
        mapd_cursor.execute(query_delete_temp)
    print("clear table：="+str(delete_file_list))

#2、在mapd上创建表格
#备注：mapd用户需要使用读入的数据需要放到/home/mapd文件夹下，不然会报错permission错误
def create_csv2table(imput_file_style=imput_file_style,if_clear=True,if_input=[0,1,0]):
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

#3、when-case给每条记录标记其所属的风险暴露区间
#在每条记录上添加一个标记，记录当前记录所属的天数
#例如：select DATEADD('DAY',c2_date-1,DATE'2016-07-01') as y_m_d,count(*) from month_201607 group by y_m_d order by y_m_d
def add_date(date_list_style=date_list_style):
    i=0
    for e in date_list_style["database"]:#遍历每一个需要打标的数据库
        print("CREATE TABLE "+e+"_temp"+" AS SELECT DATEADD('DAY',c2_date-1,DATE"+date_list_style["first_date"][i]+") as y_m_d,"+date_list_style["select_columns"]+"FROM "+e)
        mapd_cursor.execute("CREATE TABLE "+e+"_temp"+" AS SELECT DATEADD('DAY',c2_date-1,DATE"+date_list_style["first_date"][i]+") as y_m_d,"+date_list_style["select_columns"]+"FROM "+e)
        print("DROP TABLE "+e)
        mapd_cursor.execute("DROP TABLE "+e)
        print("ALTER TABLE "+e+"_temp"+" RENAME TO "+e)
        mapd_cursor.execute("ALTER TABLE "+e+"_temp"+" RENAME TO "+e)
        print("---------------------------------------")
        i=i+1
    return 0

#例如：select CASE accelerateTimes WHEN 0 THEN 1 WHEN 1 THEN 2 ELSE 3 END from temp_2
#例如：select CASE WHEN accelerateTimes BETWEEN 0 AND 5 THEN 1 WHEN accelerateTimes BETWEEN 6 AND 10 THEN 2 ELSE 3 END from temp_2
def when_case_risk_range(when_case_style=when_case_style_30day,time_column="y_m_d",range_name='30d'):
    CASE_WHEN="CASE WHEN "
    WHEN="WHEN "
    THEN="THEN "
    SELECT="SELECT "
    FROM="FROM "
    BETWEEN="BETWEEN "
    DATA="TIMESTAMP"
    ElSE_END="ELSE -1 END AS seg "
    len_time=when_case_style["time_range"].__len__()
    for e in when_case_style["database"]:#遍历每一个需要切分打标的数据库
          query=""
          for j in range(len_time):
              if j==0:#初始化
                 query="CREATE TABLE "+e+"_temp AS "+SELECT+when_case_style["select_columns"]+','+CASE_WHEN+time_column+" "+BETWEEN+\
                    DATA+when_case_style["time_range"][j][0]+" AND "+DATA+when_case_style["time_range"][j][1]+" "+THEN+ \
                    when_case_style["month_num"][j]+" "
              else:
                 if j==len_time-1:#最后when case 添加from
                    query=query+WHEN+time_column+" "+BETWEEN+ \
                          DATA+when_case_style["time_range"][j][0]+" AND "+DATA+when_case_style["time_range"][j][1]+" "+THEN+ \
                          when_case_style["month_num"][j]+" "+ElSE_END+FROM+e
                 else:
                    query=query+WHEN+time_column+" "+BETWEEN+ \
                    DATA+when_case_style["time_range"][j][0]+" AND "+DATA+when_case_style["time_range"][j][1]+" "+THEN+ \
                    when_case_style["month_num"][j]+" "
          print(query)
          mapd_cursor.execute(query)
          print("DROP TABLE "+e)
          mapd_cursor.execute("DROP TABLE "+e)
          print("ALTER TABLE "+e+"_temp"+" RENAME TO "+e)
          mapd_cursor.execute("ALTER TABLE "+e+"_temp"+" RENAME TO "+e)
          print("---------------------------------------")
    return 0

#4、对赔付数据进行处理，用carid替换vin代码
def pei_vin2car():
    #vin替换为carid
    pei_query="create table pei_of_carid as select carid2vin_2.carid,case when happen is NULL then 0 else 1 end as time_n,pei.happen,pei.claims from pei,carid2vin_2 where pei.vin=carid2vin_2.vin"
    mapd_cursor.execute(pei_query)


    #把出险次数和函数

#4、sql 对指标的准确性进行校验的规则
# def check_rule():
#     select isdrive,mileage from month_201611 where (isdrive=1 and mileage=0) or (isdrive=0 and mileage>0)
#select carid2vin_2.carid,pei.happen,pei.claims from pei,carid2vin_2 where pei.vin=carid2vin_2.vin

# 三、执行数据预计处理,不执行的步骤用#标记后跳过
if len(delete_filename_list)>0:
   pitch_delete()
# create_csv2table()
# add_date()
# when_case_risk_range()
# pei_vin2car()














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
