import os
import sys

datatype = "float"

os.system("rm -rf ./time")
# 删除已有记录

resultpath="./result/"
if not os.path.exists(resultpath):
    os.makedirs(resultpath)
if not os.path.exists(resultpath+"500.csv"):
    os.mknod(resultpath+"500.csv")
timepath="./time/"
if not os.path.exists(timepath):
    os.makedirs(timepath)
if not os.path.exists(timepath+"time.csv"):
    os.mknod(timepath+"time.csv")
print(timepath+"time.csv")  
# 处理相关路径

if os.system("nvcc purehigh.cu -o purehigh --std=c++11 -arch=sm_80 -w")==0:
    # purehigh.cu运行成功
    listtime=[]
    for i in range(5):
        os.system("./purehigh "+str(i)+" "+resultpath+"500.csv")
        # 执行 purehigh 程序并传入参数
        # print("###", "./purehigh "+str(i)+" "+resultpath+"500.csv")
        ftime=open("time_tmp.csv","r+")
        time=ftime.read()
        ftime.close()
        listtime.append(time)
    f=open(timepath+"time.csv","r+")
    this_time_list=f.readlines()
    f.close()
    this_time_list.append(",".join(listtime)+"\n")
    f=open(timepath+"time.csv","w+")
    f.writelines(this_time_list)
    f.close()
else:
    print(datatype,str(para))
    sys.exit(1)