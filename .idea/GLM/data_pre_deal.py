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

import os

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
export_path_dir="/home/mapd/dumps/output/"#输出文件目录
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

imput_file_90_style=[#导入车基数据
    {"input_file_path":import_path_dir,
     "input_file_name":
         ["month_201608.csv","month_201609.csv",
          "month_201610.csv"],
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

date_list_90_style={"database":["month_201608","month_201609",
                             "month_201610"],
                 "first_date":["\'2016-08-01\'","\'2016-09-01\'","\'2016-10-01\'"],
                 "colume_name":"y_m_d",#记录日期新指标的名称
                 "select_columns":select_columns
                 }

#每30天划分为一个风险暴露区间
select_columns="y_m_d,carid,c2_date,mileage,fuel,duration,maxspeed,accelerateTimes,decelerateTimes,sharpTurnTimes,isdrive,isfatigue,ishighspeed,pulloutTimes,isnonlocal,isnightdrive,vbigtime,maxmileage,totalmile "
when_case_style_30day={"database":["month_201606","month_201607","month_201608","month_201609",
                                   "month_201610","month_201611"],
                       "time_range":[("\'2016-06-01 00:00:00\'","\'2016-6-30 23:59:59\'"),("\'2016-07-01 00:00:00\'","\'2016-07-30 23:59:59\'"),("\'2016-08-01 00:00:00\'","\'2016-8-30 23:59:59\'"),("\'2016-09-01 00:00:00\'","\'2016-9-30 23:59:59\'"),("\'2016-10-01 00:00:00\'","\'2016-10-30 23:59:59\'"),("\'2016-11-01 00:00:00\'","\'2016-11-30 23:59:59\'")],
                       "month_num":['6','7','8','9','10','11'],
                       "select_columns":select_columns
                       }#按一定时间区间对数据进行切分，划分为不同的风险暴露

when_case_style_15day={"database":["month_201606","month_201607","month_201608","month_201609",
                                   "month_201610","month_201611"],
                       "time_range":[("\'2016-06-01 00:00:00\'","\'2016-6-15 23:59:59\'"),
                                     ("\'2016-06-16 00:00:00\'","\'2016-06-30 23:59:59\'"),
                                     ("\'2016-07-01 00:00:00\'","\'2016-07-15 23:59:59\'"),
                                     ("\'2016-07-16 00:00:00\'","\'2016-07-30 23:59:59\'"),
                                     ("\'2016-07-31 00:00:00\'","\'2016-08-14 23:59:59\'"),
                                     ("\'2016-08-15 00:00:00\'","\'2016-08-29 23:59:59\'"),
                                     ("\'2016-08-30 00:00:00\'","\'2016-09-13 23:59:59\'"),
                                     ("\'2016-09-14 00:00:00\'","\'2016-09-28 23:59:59\'"),
                                     ("\'2016-09-29 00:00:00\'","\'2016-10-13 23:59:59\'"),
                                     ("\'2016-10-14 00:00:00\'","\'2016-10-28 23:59:59\'"),
                                     ("\'2016-10-29 00:00:00\'","\'2016-11-12 23:59:59\'"),
                                     ("\'2016-11-13 00:00:00\'","\'2016-11-27 23:59:59\'")],
                       "month_num":['1','2','3','4','5','6','7','8','9','10','11','12'],
                       "select_columns":select_columns
                       }#按一定时间区间对数据进行切分，划分为不同的风险暴露

#将30天数据归纳在一起
when_case_style_90day={"database":["month_201608","month_201609","month_201610"],
                       "time_range":[("\'2016-08-02 00:00:00\'","\'2016-10-30 23:59:59\'")],
                       "month_num":['1'],
                       "select_columns":select_columns
                       }#按一定时间区间对数据进行切分，划分为不同的风险暴露

#4、把赔款数据匹配到每个车基数据上去 LFV2A1159C3567437 17 LVSHFFAL1FS337046 16
select_columns_1="y_m_d,carid,c2_date,mileage,fuel,duration,maxspeed,accelerateTimes,decelerateTimes,sharpTurnTimes,isdrive,isfatigue,ishighspeed,pulloutTimes,isnonlocal,isnightdrive,vbigtime,maxmileage,totalmile,seg "
pei2car_style={"database":["month_201606","month_201607","month_201608","month_201609",
                           "month_201610","month_201611"],
               "select_column_1":select_columns_1, #month_temp_1=month jion distinct_vin
               }

pei2car_90_style={"database":["month_201608","month_201609",
                           "month_201610"],
               "select_column_1":select_columns_1, #month_temp_1=month jion distinct_vin
               }

#5、need_att提取需要的指标,并按照carid,seg进行分组汇总处理,其中risk是风险暴露数
select_columns_1="carid,seg,count(*) as risk,sum(claims_t) as claims_t,sum(time_t) as time_t,sum(mileage) as mileage,sum(duration) as duration,avg(maxspeed) as maxspeed,sum(accelerateTimes) as a,sum(decelerateTimes) as d,sum(sharpTurnTimes) as s,sum(isfatigue) as isf,sum(ishighspeed) as ish,sum(isnonlocal) as isnl,sum(isnightdrive) as isn "
#剔除不合理的极值数据 120公里×24小时、      24小时                  平均时速200公里/小时
check_rule="where mileage>=0 and mileage<=2880 and duration>=0 and duration<=846000 and mileage/(duration/3600)<250 and accelerateTimes<=100 and sharpTurnTimes<=100 and decelerateTimes<=100 and maxspeed>=0 "
check_rule="where mileage>=0 and mileage<=2880 and duration>=0 and duration<=846000 and accelerateTimes<=150 and sharpTurnTimes<=100 and decelerateTimes<=150 and maxspeed>=0 "
#将数据指标扩展到满风险暴露数上去
select_columns_2="carid,seg,risk,claims_t,time_t,mileage,duration,maxspeed,a,d,s,isf,ish,isnl,isn"
select_columns_3="carid,seg,claims_t,time_t,mileage,duration,maxspeed,accelerateTimes,decelerateTimes,sharpTurnTimes,isfatigue,ishighspeed,isnonlocal,isnightdrive"
GLM_need_att_style={"database":["month_201606_base","month_201607_base","month_201608_base","month_201609_base",
                           "month_201610_base","month_201611_base"],
               "select_column_1":select_columns_1, #month_temp_1=month jion distinct_vin
               "check_row":check_rule,
               "mapd_talbe_type":#将所有数据库的数据导出为csv，union以后，从新导入回mapd
                        "("+
                        "carid TEXT  NOT NULL ENCODING DICT,"+
                        "seg SMALLINT,"+
                        "risk SMALLINT,"+
                        "claims_t FLOAT,"+
                        "time_t SMALLINT,"+
                        "mileage FLOAT,"+
                        "duration FLOAT,"+
                        "maxspeed FLOAT,"+
                        "accelerateTimes FLOAT,"+
                        "decelerateTimes FLOAT,"+
                        "sharpTurnTimes FLOAT,"+
                        "isfatigue FLOAT,"+
                        "ishighspeed FLOAT,"+
                        "isnonlocal FLOAT,"+
                        "isnightdrive FLOAT)",
               "select_column_2":select_columns_2,
               "select_columns_3":select_columns_3,
               "ex_pei_range":"",
               "pei_columns_groupby":""
              }

#90天用
pei_columns="carid,claims_t as claims_use,time_t as time_use"
#对聚集以后的csv再进行一次grouby操作
pei_columns_risk="claims_use,time_use,claims_t,time_t,mileage,duration,maxspeed,a,d,s,isf,ish,isnl,isn "
pei_columns_leftjoin="m.risk,m.claims_t,m.time_t,m.mileage,m.duration,m.maxspeed,m.a,m.d,m.s,m.isf,m.ish,m.isnl,m.isn "
pei_columns_groupby="carid,seg,sum(risk) as risk,sum(claims_t) as claims_t,sum(time_t) as time_t,sum(mileage) as mileage,sum(duration) as duration,avg(maxspeed) as maxspeed,sum(accelerateTimes) as a,sum(decelerateTimes) as d,sum(sharpTurnTimes) as s,sum(isfatigue) as isf,sum(ishighspeed) as ish,sum(isnonlocal) as isnl,sum(isnightdrive) as isn "
GLM_need_att_90_style={"database":["month_201608_base","month_201609_base","month_201610_base"],
                    "select_column_1":select_columns_1, #month_temp_1=month jion distinct_vin
                    "check_row":check_rule,
                    "mapd_talbe_type":#将所有数据库的数据导出为csv，union以后，从新导入回mapd
                        "("+
                        "carid TEXT  NOT NULL ENCODING DICT,"+
                        "seg SMALLINT,"+
                        "risk SMALLINT,"+
                        "claims_t FLOAT,"+
                        "time_t SMALLINT,"+
                        "mileage FLOAT,"+
                        "duration FLOAT,"+
                        "maxspeed FLOAT,"+
                        "accelerateTimes FLOAT,"+
                        "decelerateTimes FLOAT,"+
                        "sharpTurnTimes FLOAT,"+
                        "isfatigue FLOAT,"+
                        "ishighspeed FLOAT,"+
                        "isnonlocal FLOAT,"+
                        "isnightdrive FLOAT)",
                    "select_column_2":select_columns_2,
                    "select_columns_3":select_columns_3,
                    "ex_pei_range":[('\'2016-01-01 00:00:00\'','\'2016-12-31 23:59:59\'')],
                        #,('\'2015-01-01 00:00:00\'','\'2015-12-31 23:59:59\'')]#需要额外计算赔付区间
                    "pei_column":pei_columns,
                    "pei_columns_groupby":pei_columns_groupby,
                    "pei_columns_leftjoin":pei_columns_leftjoin,
                    "pei_columns_risk":pei_columns_risk
                    }

#（三）需要删除的table名字
delete_filename_list=["month_201606","month_201607","month_201608","month_201609",
                      "month_201610","month_201611"]

# delete_filename_list=[]
delete_filename_list=["carid2vin","carid2vin_2"]
delete_filename_list=["pei_of_carid"]
delete_filename_list=["month_201606_l","month_201607_l","month_201608_l","month_201609_l",
                      "month_201610_l","month_201611_l"]
delete_filename_list=["month_201606_temp","month_201607_temp","month_201608_temp","month_201609_temp",
                      "month_201610_temp","month_201611_temp","month_201606_l","month_201607_l","month_201608_l","month_201609_l",
                      "month_201610_l","month_201611_l","month_201606","month_201607","month_201608","month_201609",
                      "month_201610","month_201611"]
# delete_filename_list=["month_201606_temp_1","month_201607_temp_1","month_201608_temp_1","month_201609_temp_1",
#                       "month_201610_temp_1","month_201611_temp_1"]
# delete_filename_list=["month_201606_temp_use","month_201607_temp_use","month_201608_temp_use","month_201609_temp_use",
#                       "month_201610_temp_use","month_201611_temp_use"]
# delete_filename_list=["GLM_base_date"]
delete_filename_list=[]

# 二、mapd数据库链接设置
mapd_con = mapd_jdbc.connect(
    dbname=dbname, user=user, host=host, password=password)
mapd_cursor = mapd_con.cursor()

#三、不同功能的数据库操函数
#1、删除数据
def pitch_delete(delete_file_list=delete_filename_list):
    query_delete="DROP table"
    # print(delete_filename_list)
    for e in delete_file_list:
        query_delete_temp=str(query_delete+" "+e)
        mapd_cursor.execute(query_delete_temp)
    print("clear table：="+str(delete_file_list))

#2、在mapd上创建表格
#备注：mapd用户需要使用读入的数据需要放到/home/mapd文件夹下，不然会报错permission错误
def create_csv2table(imput_file_style=imput_file_style,if_clear=True,if_input=[1,0,0]):
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
        print("CREATE TABLE "+e+"_medie"+" AS SELECT DATEADD('DAY',c2_date-1,DATE"+date_list_style["first_date"][i]+") as y_m_d,"+date_list_style["select_columns"]+"FROM "+e)
        mapd_cursor.execute("CREATE TABLE "+e+"_medie"+" AS SELECT DATEADD('DAY',c2_date-1,DATE"+date_list_style["first_date"][i]+") as y_m_d,"+date_list_style["select_columns"]+"FROM "+e)
        print("DROP TABLE "+e)
        mapd_cursor.execute("DROP TABLE "+e)
        print("ALTER TABLE "+e+"_medie"+" RENAME TO "+e)
        mapd_cursor.execute("ALTER TABLE "+e+"_medie"+" RENAME TO "+e)
        print("---------------------------------------")
        i=i+1
    return 0

#按计算分段标志seg
#例如：select CASE accelerateTimes WHEN 0 THEN 1 WHEN 1 THEN 2 ELSE 3 END from temp_2
#例如：select CASE WHEN accelerateTimes BETWEEN 0 AND 5 THEN 1 WHEN accelerateTimes BETWEEN 6 AND 10 THEN 2 ELSE 3 END from temp_2
def when_case_risk_range(when_case_style=when_case_style_30day,time_column="y_m_d"):
    print("start when_case_risk_range!")
    CASE_WHEN="CASE WHEN "
    WHEN="WHEN "
    THEN="THEN "
    SELECT="SELECT "
    FROM="FROM "
    BETWEEN="BETWEEN "
    DATA="TIMESTAMP"
    ElSE_END="ELSE -1 END AS seg "
    len_time=when_case_style["time_range"].__len__()

    if len_time>1:#多余2个区间
        for e in when_case_style["database"]:#遍历每一个需要切分打标的数据库
            query=""
            for j in range(len_time):
                if j==0:#初始化
                    query="CREATE TABLE "+e+"_medie AS "+SELECT+when_case_style["select_columns"]+','+CASE_WHEN+time_column+" "+BETWEEN+ \
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
            print("ALTER TABLE "+e+"_medie"+" RENAME TO "+e)
            mapd_cursor.execute("ALTER TABLE "+e+"_medie"+" RENAME TO "+e)
            print("---------------------------------------")

    else:#如果只有一个元素
        for e in when_case_style["database"]:
            query="CREATE TABLE "+e+"_medie AS "+SELECT+when_case_style["select_columns"]+','+CASE_WHEN+time_column+" "+BETWEEN+ \
                  DATA+when_case_style["time_range"][0][0]+" AND "+DATA+when_case_style["time_range"][0][1]+" "+THEN+ \
                  when_case_style["month_num"][0]+" "+ElSE_END+FROM+e

            print(query)
            mapd_cursor.execute(query)
            print("DROP TABLE "+e)
            mapd_cursor.execute("DROP TABLE "+e)
            print("ALTER TABLE "+e+"_medie"+" RENAME TO "+e)
            mapd_cursor.execute("ALTER TABLE "+e+"_medie"+" RENAME TO "+e)
            print("---------------------------------------")
    return 0

#4、对赔付数据进行处理，用carid替换vin代码,p_merge_e=True表示是否将赔付数据链接到每一天上
def pei_vin2car(if_drop_carid=True,if_drop_dis=True,if_drop_happen=True,pei2car_style=pei2car_style,if_deal_pei=False):
    create_talbe_mapd="CREATE TABLE "
    if if_deal_pei==True:#赔付数据一般首次入库处理时候就准备好，后续不用继续处理
        #vin替换为carid
        if if_drop_carid==True:
           mapd_cursor.execute("DROP TABLE pei_of_carid")
        pei_query=create_talbe_mapd+"pei_of_carid as select carid2vin_2.carid,case when happen is NULL then 0 else 1 end as time_n,pei.happen,pei.claims from pei,carid2vin_2 where pei.vin=carid2vin_2.vin"
        mapd_cursor.execute(pei_query)

        #distinct一个唯vin代码与车基数据进行匹配剔除掉
        if if_drop_dis==True:
           mapd_cursor.execute("DROP TABLE distinct_vin")
        pei_query= create_talbe_mapd+"distinct_vin as select distinct(carid) from pei_of_carid"
        mapd_cursor.execute(pei_query)

        #构建一个只有出险记录的数据集pei_happened
        if if_drop_happen==True:
           mapd_cursor.execute("DROP TABLE pei_happen")
        pei_query=create_talbe_mapd+"pei_happen as select * from pei_of_carid  where time_n=1"
        mapd_cursor.execute(pei_query)
        #合并每一天的赔付和次数
        if if_drop_happen==True:
            mapd_cursor.execute("DROP TABLE pei_happen_group")
        pei_query="create table pei_happen_group as select carid,happen,sum(time_n) as time_t,sum(claims) as claims_t from pei_happen group by carid,happen"
        mapd_cursor.execute(pei_query)

    #筛选出month中和distinct_vin相对应的有效样本，并在车基数据month中加入出险数据
    #筛选vin对应的车基数据
    if_drop_temp=False
    for e in pei2car_style["database"]:
        #只保留在赔付表中有出现的车辆！
        if if_drop_temp==True:
           mapd_cursor.execute("drop table "+e+"_l")
        temp_list=list(map(lambda x:e+'.'+x,pei2car_style["select_column_1"].split(",")))
        temp_str=str(",".join(temp_list))
        query="create table "+e+"_l"+"  as  (SELECT "+temp_str+"FROM "+e+" left JOIN distinct_vin ON "+e+".carid=distinct_vin.carid where distinct_vin.carid is not NULL)"
        print(query)
        mapd_cursor.execute(query)

        #按天在车基数据中，加入索赔次数和赔款,并替换null为0
        if if_drop_temp==True:
           mapd_cursor.execute("drop table "+e+"_base")
        temp_list=list(map(lambda x:"m"+'.'+x,pei2car_style["select_column_1"].split(",")))
        temp_str=str(",".join(temp_list))
        query="create table "+e+"_base"+" as (SELECT "+temp_str+",CASE WHEN p.claims_t is null then 0 else  p.claims_t end as claims_t,CASE WHEN p.time_t is null then 0 else p.time_t end as time_t,p.happen FROM "+e+"_l"+" m left JOIN pei_happen_group p ON m.carid=p.carid"+ \
              " AND EXTRACT(YEAR FROM m.y_m_d)=EXTRACT(YEAR FROM p.happen) AND EXTRACT(MONTH FROM m.y_m_d)=EXTRACT(MONTH FROM p.happen) AND EXTRACT(DAY FROM m.y_m_d)=EXTRACT(DAY FROM p.happen) where m.seg<>-1)"
        print(query)
        mapd_cursor.execute(query)
        mapd_cursor.execute("drop table "+e+"_l")
        print("---------------------------------------------------------------")
#-----------基础数据导入、数据库拼接结束-----------------形成temp文件供后续数据建模使用（每日-车基-出险为一条数据）


#5、提取需要需要的指标进行glm模型拟合
#合并csv文件
def csv_merge(flle_path=export_path_dir,merge_filename="month_2016_6_11.csv"):
    import glob
    import time
    # print(flle_path+'*.csv')
    csvx_list = glob.glob(flle_path+'*.csv')
    # print(csvx_list)
    if os.path.exists(export_path_dir+merge_filename):
        print("删除",export_path_dir+merge_filename)
        os.remove(export_path_dir+merge_filename)
    print('正在处理............')
    for i in csvx_list:
        fr = open(i,'r').read()
        with open(export_path_dir+merge_filename,'a') as f:
            f.write(fr)
        print('写入成功！')
    print('写入完毕！')

#删除文件夹下面的所有内容
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

#提取需要的数据保存为数据集:
# mismatching=True ，表示车基与出险数据匹配（使用key=car，seg,使用month_y_m_base的赔付数据）
# mismatching=False，表示车基与出险不匹配  （使用key=car，seg,此时赔付数据需要另行计算）
def save2use(GLM_need_att_style=GLM_need_att_style,all_risk='15.0',limit_risk='12.0',seg_range='15Days',mismatching=True):
    #（1）创建导回用数据库
    # mapd_cursor.execute("drop table GLM_base_date")
    create_talbe_mapd="CREATE TABLE IF NOT EXISTS "
    query=create_talbe_mapd+"GLM_base_date"+GLM_need_att_style["mapd_talbe_type"]
    mapd_cursor.execute(query)
    mapd_cursor.execute("TRUNCATE TABLE GLM_base_date")
    #（2）合并前删除文件夹下历史数据
    del_file(export_path_dir)#导入数据到export_path_dir之前，先清空

    #（3）开始合并
    if mismatching==True:#所有不同seg都在一个月份的数据中，不需要跨月份很并计算key=car,seg
        #计算风险暴露数，并按暴露数扩展车基和赔付样本数据
        temp_list=GLM_need_att_style["select_column_2"].split(",")
        list_l=[]
        temp_database=""
        #统计risk风险暴露数
        for e  in temp_list:
            if (not e.__eq__("carid")) and (not e.__eq__("seg")) and (not e.__eq__("risk")):
                if  e.__eq__("maxspeed"):
                    list_l.append(str(e))
                else:
                    list_l.append("case when risk<"+all_risk+" then "+e+"*cast("+all_risk+"/risk as float) else "+"cast("+e+" as float) end as "+e)
            else:
                list_l.append(str(e))
        temp_str=str(",".join(list_l))

        #数据库读取
        for e in GLM_need_att_style["database"]:
            query="create table  "+e+"_use"+" as select "+GLM_need_att_style["select_column_1"]+"from "+e+" "+GLM_need_att_style["check_row"]+" group by carid,seg"
            mapd_cursor.execute(query)
            #按风险暴露数，扩展车基数据
            query="COPY (select "+temp_str+" from "+e+"_use"+" where "+"risk>="+limit_risk+")  to "+'\''+import_path_dir+"output/"+e+"_use.csv"+'\''+" with (header=\'False\')"
            print(query)
            mapd_cursor.execute(query)
            mapd_cursor.execute("drop table "+e+"_use")
            print("--------------------------")


        #输出结果合并
        merge_filename="month_2016_6_11.csv"
        csv_merge(merge_filename=merge_filename)

        #导入合并后的csv文件
        print("------------导入csv------------:"+export_path_dir+merge_filename)
        query="copy GLM_base_date from "+"\'"+export_path_dir+merge_filename+"\'"+" WITH (header='false')"
        mapd_cursor.execute(query)


        if  seg_range=='30Days':
            # # #从新按seg和carid汇总一个赔付表，用来左链接,只有当seg=30天时候使用
            mapd_cursor.execute("drop table pei_6_11_seg")
            query="create table pei_6_11_seg as select carid,EXTRACT(month FROM p.happen) as seg,sum(time_t) as time_t,sum(claims_t) as claims_t from pei_happen_group p where EXTRACT(YEAR FROM p.happen)=2016 and EXTRACT(month FROM p.happen) between 6 and 11 group by carid,seg"
            mapd_cursor.execute(query)

            #GLM_last是最后的数据集
            mapd_cursor.execute("drop table GLM_base_date_30Days")
            temp_list=list(map(lambda x:'m'+'.'+x,GLM_need_att_style["select_columns_3"].split(",")))
            temp_str=str(",".join(temp_list))
            # mapd_cursor.execute("drop table GLM_last_15day")
            query="create table GLM_base_date_"+seg_range+" as select "+"case when p.time_t is null then 0 else p.time_t end as time_use,case when p.claims_t  is null then 0 else p.claims_t end as claims_use,"+temp_str+" from GLM_base_date m left join pei_6_11_seg p on m.carid=p.carid and m.seg=p.seg"
            print(query)
            mapd_cursor.execute(query)

    else:
         #seg需要跨月合并，需要使用额外赔付数据,使用额外数据时候未按暴露数对车基数据进行拓展
         #计算额外赔付数据pei_group
         str_temp="case when "
         len_list=GLM_need_att_style["ex_pei_range"].__len__()
         for i in range(len_list):
             if i==0 and i!=len_list-1:
                str_temp=str_temp+"(happen between "+"TIMESTAMP"+GLM_need_att_style["ex_pei_range"][i][0]+" and TIMESTAMP"+GLM_need_att_style["ex_pei_range"][i][1]+") or "
             if i==0 and i==len_list-1:
                str_temp=str_temp+"(happen between "+"TIMESTAMP"+GLM_need_att_style["ex_pei_range"][i][0]+" and TIMESTAMP"+GLM_need_att_style["ex_pei_range"][i][1]+") then 1 else -1 end as seg "
             if i!=0 and i!=len_list-1:
                str_temp=str_temp+"(happen between "+"TIMESTAMP"+GLM_need_att_style["ex_pei_range"][i][0]+" and TIMESTAMP"+GLM_need_att_style["ex_pei_range"][i][1]+") or "
             if i!=0 and i==len_list-1:
                str_temp=str_temp+"(happen between "+"TIMESTAMP"+GLM_need_att_style["ex_pei_range"][i][0]+" and TIMESTAMP"+GLM_need_att_style["ex_pei_range"][i][1]+") then 1 else -1 end as seg "

         mapd_cursor.execute("drop table pei_nonmis")
         query="create table pei_nonmis as select "+GLM_need_att_style["pei_column"]+","+str_temp+"from pei_happen_group"
         print(query)
         mapd_cursor.execute(query)
         # mapd_cursor.execute("drop table pei_nonmis_temp")
         query="create table pei_nonmis_temp as select carid,seg,sum(claims_use) as claims_use,sum(time_use) as time_use from pei_nonmis where seg<>-1 group by carid,seg"
         print(query)
         mapd_cursor.execute(query)
         mapd_cursor.execute("drop table pei_nonmis")
         mapd_cursor.execute("ALTER TABLE pei_nonmis_temp"+" RENAME TO pei_nonmis")

         #在单个数据库中进行首次汇总
         for e in GLM_need_att_style["database"]:
            #按风险暴露数，扩展车基数据
            query="COPY (select "+GLM_need_att_style["select_column_1"]+"from "+e+" "+GLM_need_att_style["check_row"]+" group by carid,seg )  to "+'\''+import_path_dir+"output/"+e+"_use.csv"+'\''+" with (header=\'False\')"
            print(query)
            mapd_cursor.execute(query)
            #输出结果合并
         merge_filename="month_2016_6_11.csv"
         csv_merge(merge_filename=merge_filename)
         print("--------------------------")

         #导入合并后的csv文件
         print("------------导入csv------------:"+export_path_dir+merge_filename)
         query="copy GLM_base_date from "+"\'"+export_path_dir+merge_filename+"\'"+" WITH (header='false')"
         mapd_cursor.execute(query)

         #对全部文件进行一次group carid seg的汇总
         #group by
         query="create table GLM_base_date_temp as select "+GLM_need_att_style["pei_columns_groupby"]+" from GLM_base_date group by carid,seg"
         print(query)
         mapd_cursor.execute(query)
         mapd_cursor.execute("drop table GLM_base_date")
         mapd_cursor.execute("ALTER TABLE GLM_base_date_temp"+" RENAME TO GLM_base_date")

         #链接
         query="create table GLM_base_date_temp as select  case when p.claims_use is null then 0 else p.claims_use end as claims_use,case when p.time_use is null then 0 else p.time_use end as time_use,"+GLM_need_att_style["pei_columns_leftjoin"]+" from GLM_base_date m left join pei_nonmis p on m.carid=p.carid and m.seg=p.seg"
         print(query)
         mapd_cursor.execute(query)
         mapd_cursor.execute("drop table GLM_base_date")
         mapd_cursor.execute("ALTER TABLE GLM_base_date_temp"+" RENAME TO GLM_base_date")

         #对GLM_base_date_90Days按照risk>limit进行扩展(90天)
         #计算风险暴露数，并按暴露数扩展车基和赔付样本数据
         temp_list=GLM_need_att_style["pei_columns_risk"].split(",")
         list_l=[]
         #统计risk风险暴露数
         for e  in temp_list:
                if  e.__eq__("maxspeed"):#maxspeed不进行扩展
                    list_l.append(str(e))
                else:
                    if  e.__eq__("claims_use"):#只有当claim_use是落在出险区间的记录才进行扩展
                        list_l.append("case when risk<"+all_risk+" and claims_use=claims_t then "+e+"*cast("+all_risk+"/risk as float) else "+"cast("+e+" as float) end as "+e)
                    else:
                        list_l.append("case when risk<"+all_risk+" then "+e+"*cast("+all_risk+"/risk as float) else "+"cast("+e+" as float) end as "+e)
         temp_str=str(",".join(list_l))

         #进行risk扩展操作
         mapd_cursor.execute("drop table GLM_base_date_"+seg_range)
         query="create table GLM_base_date_"+seg_range+" as select case when time_use>0 then 1 else 0 end as ispei,"+temp_str+" from GLM_base_date where risk>="+limit_risk
         mapd_cursor.execute(query)
         mapd_cursor.execute("drop table GLM_base_date")
         print("--------------------------")

#6、把指定指标分段并离散化为（1,0,0,0）的onehot类型
one_hot_style_90days={"database":"GLM_base_date_90Days",#数据库
                      "att_pei":"claims_use,time_use"  ,
                      "att_car":"mileage,duration,maxspeed,a,d,isf,ish,isn",                      #指标
                      "att_range":{"mileage":[(0.0,2000.0),(2000.0,4000.0),(4000.0,6000.0),(6000.0,8000.0),(8000.0,10000.0),(10000.0,1000000.0)],
                                   "duration":[(0.0,400000.0),(400000.0,800000.0),(800000.0,1200000.0),(1200000.0,50000000.0)],
                                   "maxspeed":[(0.0,24.0),(24.0,48.0),(48.0,72.0),(72.0,96.0),(96.0,120.0)],
                                   "a":[(0.0,10.0),(10.0,20.0),(30.0,40.0),(40.0,1000000.0)],
                                   "d":[(0.0,100.0),(100.0,200.0),(200.0,300.0),(300.0,400.0),(400.0,1000000.0)],
                                   "isf":[(0.0,2.5),(2.5,100000)],
                                   "ish":[(0.0,9.0),(9.0,18.0),(18.0,27.0),(27.0,36.0),(36.0,45.0)],
                                   "isn":((0.0,0.05),(0.05,10000))#只判断有无夜间驾驶
                                  }
                     }

#根据指标名称生成相应的onehot的sql代码
def one_hot_sqltxt(dict_n=one_hot_style_90days["att_range"],key="mileage"):
    str_list=[]#记录case when
    att_list=[]#记录指标 ”a_1,a_2,a_3“
    i=0
    len_atrr=dict_n[key].__len__()
    for e  in dict_n[key]:

        if i!=len_atrr-1:#最后一项
           att_list.append(key+"_"+str(int(e[0]))+"_"+str(int(e[1])))
           str_list.append("case when "+key+">="+str(e[0])+" and "+key+"<"+str(e[1])+" then 1.0 else 0.0 end as "+key+"_"+str(int(e[0]))+"_"+str(int(e[1])))
        else:
           str_list.append("case when "+key+">="+str(e[0])+" then 1.0 else 0.0 end as "+key+"_"+str(int(e[0]))+"_g")
           att_list.append(key+"_"+str(int(e[0]))+"_g")
        i=i+1
    sql_text=str(",".join(str_list))
    att_text=str(",".join(att_list))
    return sql_text,att_text

#设置需要进行拟合的数据，并转换为one_hot(1,0,0)的形式,pei_or_time=是赔付还是次数,group_avg是否需要按车基属性进行一次group求平均
def value2one_hot(one_hot_style=one_hot_style_90days,pei_or_time="pei",group_avg=True):
    str_list=[]#记录case when
    att_list=[]#记录指标 ”a_1,a_2,a_3“
    value_car_list=one_hot_style_90days["att_car"].split(",")
    for e in value_car_list:
        a,b=one_hot_sqltxt(dict_n=one_hot_style_90days["att_range"],key=e)
        str_list.append(a)
        att_list.append(b)
    sql_text=str(",".join(str_list))
    att_text=str(",".join(att_list))
    if pei_or_time=="pei":
        pei_query=one_hot_style["att_pei"].split(",")[0]#只要出金额
    else:
        pei_query=one_hot_style["att_pei"].split(",")[1]#只要出险次数

    #计算基准车基-----------------------------------
    query="create table GLM_base_date_90Days_temp as select "+sql_text+" from GLM_base_date_90Days"#1为常数项
    #求one_hot指标的权重
    mapd_cursor.execute(query)
    #每个车基数据被分为了多少段

    list_len_each=[len(e.split(",")) for e in att_list]
    list_temp=list(zip(list(range(list_len_each.__len__())),list_len_each))
    list_temp=list(map(lambda x:[x[0]+1]*x[1],list_temp))
    list_a=[]
    for e in list_temp:
        list_a.extend(e)

    print("每个车基数据被分为了多少段:",list_len_each)
    #所有用于建立GLM模型的车基指标名称（分段后的）
    columns_name=",".join(att_list).split(",")
    print("所有用于建立GLM模型的车基指标名称（分段后的）:",columns_name)
    #每段数据的风险暴露数
    query="select "+",".join(list(map(lambda x:"sum("+x+") as "+x,",".join(att_list).split(","))))+" from GLM_base_date_90Days_temp"
    mapd_cursor.execute(query)
    results = mapd_cursor.fetchall()
    df = pandas.DataFrame(results)
    risk_num=df.values[0]
    print("风险暴露数据：",risk_num)
    #合并数据求基准车基, list_a(分组号)+columns_name（分组名字）+risk_num(分组数)，目标max(risk_num) group by columns_name
    car_base_list=[]
    i=0
    for e in list_len_each:#[6, 4, 5, 4, 5, 2, 5, 2]
          max_value=0
          max_att=""
          for j in range(e):
              if risk_num[j+i]>max_value:
                 max_value=risk_num[j+i]
                 max_att=columns_name[j+i]
          car_base_list.append(max_att)
          i=i+e
    #得到基准车基列表
    print("基准车基数据列表：",car_base_list)
    #从代拟合数据中剔除基准车基
    for e  in car_base_list:
        columns_name.remove(e)
    print("剔除基准车以后的剩余拟合车基数据表,长度：",columns_name,len(columns_name))
    mapd_cursor.execute("drop table GLM_base_date_90Days_temp")
    att_text=",".join(columns_name)#替换指标集
    #计算基准车基指标结束-----------------------------------------------

    if group_avg==False:#不求平均
        query="create table GLM_base_date_90Days_temp as select "+pei_query+","+sql_text+" from GLM_base_date_90Days" #1为常数项
        mapd_cursor.execute(query)


        query="select "+pei_query+",1.0 as constant,"+att_text+" from GLM_base_date_90Days_temp"
        print(query)
        query="COPY ("+query+")  to "+'\''+import_path_dir+"output/GLM_base_date_90Days_one_hot_norisk.csv"+'\''+" with (header=\'True\')"
        mapd_cursor.execute(query)

        mapd_cursor.execute("drop table GLM_base_date_90Days_temp")

    else:#求平均
        query="create table GLM_base_date_90Days_temp as select "+pei_query+","+sql_text+" from GLM_base_date_90Days" #1为常数项
        mapd_cursor.execute(query)

        query="select avg("+pei_query+") as "+pei_query+",count(*) as risk,1.0 as constant,"+att_text+" from GLM_base_date_90Days_temp group by "+att_text
        print(query)
        query="COPY ("+query+")  to "+'\''+import_path_dir+"output/GLM_base_date_90Days_one_hot.csv"+'\''+" with (header=\'True\')"
        mapd_cursor.execute(query)

        query="select avg("+pei_query+") as "+pei_query+",1.0 as constant,"+att_text+" from GLM_base_date_90Days_temp group by "+att_text
        print(query)
        query="COPY ("+query+")  to "+'\''+import_path_dir+"output/GLM_base_date_90Days_one_hot_norisk.csv"+'\''+" with (header=\'True\')"
        mapd_cursor.execute(query)

        mapd_cursor.execute("drop table GLM_base_date_90Days_temp")
    return att_text

a=value2one_hot(one_hot_style=one_hot_style_90days,group_avg=True)

# 三、执行数据预计处理,不执行的步骤用#标记后跳过（代码的执行顺序不能乱）
delete_filename_list=["month_201606_temp","month_201607_temp","month_201608_temp","month_201609_temp",
                      "month_201610_temp","month_201611_temp","month_201606_l","month_201607_l","month_201608_l","month_201609_l",
                      "month_201610_l","month_201611_l","month_201606","month_201607","month_201608","month_201609",
                      "month_201610","month_201611","month_201606_temp_use","month_201607_temp_use","month_201608_temp_use","month_201609_temp_use",
                      "month_201610_temp_use","month_201611_temp_use"]

delete_filename_list=[]
delete_filename_list=["month_201608",
                      "month_201610","month_201608_base","month_201609_base",
                      "month_201610_base"]
delete_filename_list=[]




#90天非对称版：使用
# if len(delete_filename_list)>0:
#    pitch_delete(delete_file_list=delete_filename_list)
# create_csv2table(imput_file_style=imput_file_90_style,if_clear=True,if_input=[1,0,0])#medie为中间
# add_date(date_list_style=date_list_90_style)#medie为中间文件
# when_case_risk_range(when_case_style=when_case_style_90day)#medie为中间文件
# pei_vin2car(if_drop_carid=False,if_drop_dis=False,if_drop_happen=False,pei2car_style=pei2car_90_style,if_deal_pei=False)
# save2use(GLM_need_att_style=GLM_need_att_90_style,all_risk='90.0',limit_risk='88.0',seg_range='90Days',mismatching=False)


#30天对齐版：使用claim_t(对齐)或者claim_use(对齐)
# if len(delete_filename_list)>0:
#     pitch_delete(delete_file_list=delete_filename_list)
# # create_csv2table(imput_file_style=imput_file_style,if_clear=True,if_input=[1,0,0])#medie为中间
# # add_date(date_list_style=date_list_style)#medie为中间文件
# # when_case_risk_range(when_case_style=when_case_style_30day)#medie为中间文件
# # pei_vin2car(if_drop_carid=False,if_drop_dis=False,if_drop_happen=False,pei2car_style=pei2car_style,if_deal_pei=False)
# save2use(GLM_need_att_style=GLM_need_att_style,all_risk='30.0',limit_risk='28.0',seg_range='30Days',mismatching=True)

# #15天对齐版：使用claim_t(对齐)或者claim_use
# if len(delete_filename_list)>0:
#     pitch_delete(delete_file_list=delete_filename_list)
# create_csv2table(imput_file_style=imput_file_style,if_clear=True,if_input=[1,0,0])#medie为中间
# # add_date(date_list_style=date_list_style)#medie为中间文件
# # when_case_risk_range(when_case_style=when_case_style_30day)#medie为中间文件
# # pei_vin2car(if_drop_carid=False,if_drop_dis=False,if_drop_happen=False,pei2car_style=pei2car_style,if_deal_pei=False)
# save2use(GLM_need_att_style=GLM_need_att_style,all_risk='15.0',limit_risk='13.0',seg_range='15Days',mismatching=True)


#------onehot车基数据



# Get the results
# mapd_cursor.execute(query)
# results = mapd_cursor.fetchall()

# Make the results a Pandas DataFrame
#
# df = pandas.DataFrame(results)
# print(df)
# print("df:=",df)
# print(type(df[1]))

# Make a scatterplot of the results

# plt.scatter(df[1], df[2])
#
# plt.show()
