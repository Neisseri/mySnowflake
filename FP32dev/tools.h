#ifndef TOOL_H
    #define TOOL_H
    #include"./paras.h"
    template<typename T>
    void writetocsv(string filename,T* data,int X,int Y){
        ofstream f(filename);
        f<<",";
        for(int x=0;x<X;x++){
            if(x==X-1)f<<x<<"\n";
            else f<<x<<",";
        }
        for(int y=0;y<Y;y++){
            f<<y<<",";
            for(int x=0;x<X;x++){
                if(x==X-1){
                    f<<data[y*X+x]<<"\n";
                }else{
                    f<<data[y*X+x]<<",";
                }
            }
        }
        f.close();
    }
    __global__ void dataprepare_high(highprecision *phi){
        int x=blockIdx.z%unitdimX*unitx+threadIdx.x;
        int y=blockIdx.z/unitdimX*unity+threadIdx.y; // 计算线程的二维坐标
        highprecision(*phid)[dimX]=(highprecision(*)[dimX])phi;
        // 指针数组 phid, 用于将一维指针 phi 解释为二维数组
        float dis1=pow(x-(dimX/2+8),2)+pow(y-(dimY/2+8),2);
        //计算当前点到中心点距离的平方
        if(dis1<seed){
            phid[y][x]=1;
        }
        // 如果距离小于 seed , 数据设为 1
    }

    __global__ void dataprepare_half(half *phi){
        int x=blockIdx.z%unitdimX*unitx+threadIdx.x;
        int y=blockIdx.z/unitdimX*unity+threadIdx.y;
        half(*phid)[dimX]=(half(*)[dimX])phi;
        float dis1=pow(x-(dimX/2+8),2)+pow(y-(dimY/2+8),2);
        if(dis1<seed){
            phid[y][x]=(half)1;
        }
    }

#endif