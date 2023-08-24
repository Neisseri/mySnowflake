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

    __global__ void motivation_monitor2_datasychr(highprecision *now,highprecision *last){
        int x=blockIdx.z%unitdimX*unitx+threadIdx.x;
        int y=blockIdx.z/unitdimX*unity+threadIdx.y;
        highprecision(*nowd)[dimX]=(highprecision(*)[dimX])now;
        highprecision(*lastd)[dimX]=(highprecision(*)[dimX])last;
        lastd[y][x]=nowd[y][x];
    }
    template<typename T>
    void BubbleSort(T *p,int length,int* ind_diff){
        for(int m=0;m<length;m++)ind_diff[m]=m;
        for(int i=0;i<length;i++){
            for(int j=0;j<length-i-1;j++){
                if(p[j]<p[j+1]){
                    T temp=p[j];
                    p[j]=p[j+1];
                    p[j+1]=temp;
                    int ind_temp=ind_diff[j];
                    ind_diff[j]=ind_diff[j+1];
                    ind_diff[j+1]=ind_temp;
                }
            }
        }
    }
    int get_neibour(int index,int direct,int r){
        int x=index%unitdimX;
        int y=index/unitdimX;
        if(direct==1)x=(x-r+unitdimX)%unitdimX;//left
        if(direct==2)x=(x+r)%unitdimX;//right
        if(direct==3)y=(y-r+unitdimY)%unitdimY;//down
        if(direct==4)y=(y+r)%unitdimY;//up
        if(direct==5){
            x=(x-r+unitdimX)%unitdimX;
            y=(y-r+unitdimY)%unitdimY;
        }
        if(direct==6){
            x=(x+r)%unitdimX;
            y=(y-r+unitdimY)%unitdimY;
        }
        if(direct==7){
            x=(x+r)%unitdimX;
            y=(y+r)%unitdimY;
        }
        if(direct==8){
            x=(x-r+unitdimX)%unitdimX;
            y=(y+r)%unitdimY;
        }
        return y*unitdimX+x;
    }
    #if ((defined HALF)||(defined HALF2))
        __global__ void purelow2high_aftercomputing(purelowprecision *hdata,float *fdata){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int x_offset=threadIdx.x ;
            int x_start;
            #ifdef HALF
                x_start=unitindex_x*unitx;
            #else
                x_start=unitindex_x*uxd2;
            #endif
            int x=x_start+x_offset;
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;
            float(*fdatad)[dimX]=(float(*)[dimX])fdata;
            purelowprecision(*hdatad)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])hdata;
            #ifdef HALF
                fdatad[y][x]=(float)hdatad[y][x];
            #else
                fdatad[y][x*2]=__low2float(hdatad[y][x]);
                fdatad[y][x*2+1]=__high2float(hdatad[y][x]);
            #endif

        }
    #endif
    #if ((defined AMSTENCIL)||(defined GRAM1||(defined GRAM2)))
        __global__ void highdata_to_low(highprecision *highdata,lowprecision *lowdata){
            highprecision(*highdatad)[dimX]=(highprecision(*)[dimX])highdata;
            lowprecision(*lowdatad)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])lowdata;
            int y=blockIdx.z/unitdimX*unity+threadIdx.y;
            #ifdef AMSTENCIL
                if(blockIdx.x==1)return;
                int x=blockIdx.z%unitdimX*uxd2+threadIdx.x;
                lowdatad[y][x]=__floats2half2_rn(highdatad[y][x*2],highdatad[y][x*2+1]);
            #endif
            #if ((defined GRAM1 )|| (defined GRAM2))
                int x=blockIdx.z%unitdimX*unitx+threadIdx.x+uxd2*blockIdx.x;
                lowdatad[y][x]=highdatad[y][x];
            #endif
        }
        __global__ void data_sychro_aftercomputation(highprecision *highdata,lowprecision *lowdata,int* typedata){
            highprecision(*highdatad)[dimX]=(highprecision(*)[dimX])highdata;
            lowprecision(*lowdatad)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])lowdata;
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int y=unitindex_y*unity+threadIdx.y;
            #ifdef AMSTENCIL
                if(blockIdx.x==1)return;
                int x=unitindex_x*uxd2+threadIdx.x;
                int(*typedatad)[unitdimX]=(int(*)[unitdimX])typedata;
                if(typedatad[unitindex_y][unitindex_x]==1){
                    highdatad[y][x*2]=__low2float(lowdatad[y][x]);
                    highdatad[y][x*2+1]=__high2float(lowdatad[y][x]);
                }
            #endif
            #ifdef GRAM1
                int x=unitindex_x*unitx+threadIdx.x+uxd2*blockIdx.x;
                if((unitindex_y*unitdimX+unitindex_x)<unitNums*ratio){
                    highdatad[y][x]=__half2float(lowdatad[y][x]);
                }
            #endif
            #ifdef GRAM2
                int x=unitindex_x*unitx+threadIdx.x+uxd2*blockIdx.x;
                if((unitindex_y*unitdimX+unitindex_x)%100<100*ratio){
                    highdatad[y][x]=__half2float(lowdatad[y][x]);
                }
            #endif
        }
    #endif
    #ifdef AMSTENCIL
        __global__ void get_max_diff1(highprecision* con,highprecision* max_diff_con){
            int unitindex_x=threadIdx.x + blockIdx.x * blockDim.x;
            int unitindex_y=threadIdx.y + blockIdx.y * blockDim.y;
            int x_start=unitindex_x*unitx;
            int y_start=unitindex_y*unity;
            highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
            highprecision(*max_diff_cond)[unitdimX]=(highprecision(*)[unitdimX])max_diff_con;
            highprecision mincon=1.0,maxcon=0.0;
            int p[4]={0,5,10,15};
            for(int j=0;j<4;j++){
                for(int i=0;i<4;i++){
                    if(cond[y_start+p[j]][x_start+p[i]]<mincon)mincon=cond[y_start+p[j]][x_start+p[i]];
                    if(cond[y_start+p[j]][x_start+p[i]]>maxcon)maxcon=cond[y_start+p[j]][x_start+p[i]];
                }
            }
            max_diff_cond[unitindex_y][unitindex_x]=maxcon-mincon;
        }
        __global__ void get_max_diff2(highprecision* test_data_last,highprecision* test_data,highprecision* max_diffs){
            int unitindex_x=threadIdx.x + blockIdx.x * blockDim.x;
            int unitindex_y=threadIdx.y + blockIdx.y * blockDim.y;
            int x_start=unitindex_x*unitx;
            int y_start=unitindex_y*unity;
            highprecision(*test_data_lastd)[dimX]=(highprecision(*)[dimX])test_data_last;
            highprecision(*test_datad)[dimX]=(highprecision(*)[dimX])test_data;
            highprecision(*max_diffsd)[unitdimX]=(highprecision(*)[unitdimX])max_diffs;
            highprecision max_diff=0.0;
            int p[4]={2,6,10,14};
            for(int j=0;j<4;j++){
                for(int i=0;i<4;i++){
                    highprecision diff_this=abs(test_data_lastd[y_start+p[j]][x_start+p[i]]-test_datad[y_start+p[j]][x_start+p[i]]);
                    if(diff_this>max_diff)max_diff=diff_this;
                }
            }
            max_diffsd[unitindex_y][unitindex_x]=max_diff;
        }
        __global__ void get_type(highprecision* max_diff_con,int *type_old,int *type_con){
            int unitindex_x=threadIdx.x + blockIdx.x * blockDim.x;
            int unitindex_y=threadIdx.y + blockIdx.y * blockDim.y;
            highprecision(*max_diff_cond)[unitdimX]=(highprecision(*)[unitdimX])max_diff_con;
            int(*type_cond)[unitdimX]=(int(*)[unitdimX])type_con;
            int(*type_oldd)[unitdimX]=(int(*)[unitdimX])type_old;
            type_oldd[unitindex_y][unitindex_x]=type_cond[unitindex_y][unitindex_x];
            highprecision max_con=max_diff_cond[unitindex_y][unitindex_x];
            max_con=unitindex_y<unitdimY-1?max(max_con,max_diff_cond[unitindex_y+1][unitindex_x]):max_con;
            max_con=unitindex_y>0?max(max_con,max_diff_cond[unitindex_y-1][unitindex_x]):max_con;
            max_con=unitindex_x<unitdimX-1?max(max_con,max_diff_cond[unitindex_y][unitindex_x+1]):max_con;
            max_con=unitindex_x>0?max(max_con,max_diff_cond[unitindex_y][unitindex_x-1]):max_con;
            if(max_con<threshold){
                type_cond[unitindex_y][unitindex_x]=1;
            }
            else if(max_con>=threshold){
                type_cond[unitindex_y][unitindex_x]=2;
            }
        }
        __global__ void data_sychro_duringcomputation(highprecision* eta1,highprecision* eta2,half2* heta1,half2* heta2,int* old_con,int* new_con){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int(*old_cond)[unitdimX]=(int(*)[unitdimX])old_con;
            int(*new_cond)[unitdimX]=(int(*)[unitdimX])new_con;
            int x=unitindex_x*uxd2+threadIdx.x;
            int y=unitindex_y*unity+threadIdx.y;
            if(old_cond[unitindex_y][unitindex_x]==2&&new_cond[unitindex_y][unitindex_x]==1){
                highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
                highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
                half2(*heta1d)[dimXd2]=(half2(*)[dimXd2])heta1;
                half2(*heta2d)[dimXd2]=(half2(*)[dimXd2])heta2;
                heta1d[y][x]=__floats2half2_rn(eta1d[y][x*2],eta1d[y][x*2+1]);
                heta2d[y][x]=__floats2half2_rn(eta2d[y][x*2],eta2d[y][x*2+1]);
            }
        }
    #endif
#endif