# --encoding:utf-8--
#
from MySQLHelper import *
import  numpy as np

import pandas as pd
df=pd.read_excel("./计量党建学信息收集.xlsx")
arr=df.values
helper =MySQLHelper("localhost", "root", "123456")
helper.setDB("partycreatestudy")
for row in arr:
    print(row)
    helper.insertsql(
        "insert into stuInfo(id,name,number,grade,dangyuan,date,reward) values(null,'" + str(row[0]) + "','" + str(
            row[1]) + "','" + str(2017) + "','" + str(row[3]) + "','" + str(row[4]) + "','" + str(row[5]) + "')")
