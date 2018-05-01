import random
import re
from MySQLHelper import *
from docx import Document
import os
import numpy as np
import pandas as pd
from dtaidistance import dtw as dt
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root) #当前目录路径
        print(dirs) #当前路径下所有子目录
        print(files) #当前路径下所有非目录子文件
        return files
files=file_name('./计量党建')
helper =MySQLHelper("localhost", "root", "123456")
helper.setDB("partycreatestudy")
for f in files:
    print(f)
    # 源文件 test.docx
    doc = Document("./计量党建/"+str(f))
    # 遍历所有表格
    flag=1
    name=""
    number=""
    grade=""
    dangyuan=""
    for table in doc.tables:

        for row in table.rows:
            if(len(row.cells[0].text)<=0):
                print("空")
                break;
            print(flag)
            if(flag<3):
                if(flag==1):
                    print(row.cells[0].text)
                    print(row.cells[1].text)
                    name = row.cells[1].text
                    print(row.cells[2].text)

                    print(row.cells[3].text)
                    number = row.cells[3].text
                else:
                    print(row.cells[0].text)
                    print(row.cells[1].text)
                    grade = row.cells[1].text
                    print(row.cells[2].text)

                    print(row.cells[3].text)
                    dangyuan = row.cells[3].text
            elif(flag==3):
                pass
            else:
                print(row.cells[0].text)
                print(row.cells[1].text)
                helper.insertsql("insert into stuInfo(id,name,number,grade,dangyuan,date,reward) values(null,'" + str(
                    name) + "','" + str(number) + "','" + str(grade) + "','" + str(dangyuan) + "','" + str(
                    row.cells[0].text) + "','" + str(row.cells[1].text) + "')")
            flag=flag+1
