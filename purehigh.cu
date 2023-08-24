#include "function.h"
#include "tools.h"
int main(int argc,char* argv[]){
    highprecision *phi,*phi_lap,*tempr,*tempr_lap,*phidx,*phidy,*epsilon,*epsilon_deri;
    CHECK_ERROR(cudaMallocManaged((void**)&phi,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phi_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&tempr,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&tempr_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phidx,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&phidy,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&epsilon,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&epsilon_deri,sizeof(highprecision)*dimX*dimY));
    // 分配大小为 dimX * dimY 的二维数组空间
    dim3 blocks(unitx,unity);
    // 二维线程网络中的线程块
    dim3 grids(1,1,unitdimX*unitdimY);
    // 线程网络维度
    dataprepare_high<<<grids,blocks>>>(phi); // 启动 CUDA kernal
    // dataprepare_high 函数定义在 tools.h 中
    cudaDeviceSynchronize(); // CPU 等待 GPU 上所有操作全都完成
    
    #ifdef End2end
        cudaEvent_t start,stop;float elapsed;
        // start, stop : 记录CUDA事件, 用于测量GPU代码执行时间
        // elapsed : 存储代码执行时间
    #endif
    for(int i=0;i<timesteps;i++){
        #ifdef End2end
            if(i==5){
                CHECK_ERROR(cudaEventCreate(&start));
                CHECK_ERROR(cudaEventCreate(&stop)); // 创建 CUDA 事件
                CHECK_ERROR(cudaEventRecord(start,0));
                // 记录 start 事件时间戳到 stream 0 中
            }
        #endif
        kernel1_pure<<<grids,blocks>>>(phi,phi_lap,tempr,tempr_lap,phidx,phidy,epsilon,epsilon_deri);
        kernel2_pure<<<grids,blocks>>>(phi,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap);
        cudaDeviceSynchronize();
    }
    #ifdef End2end
        if(timesteps>5){
            CHECK_ERROR(cudaEventRecord(stop,0));
            CHECK_ERROR(cudaEventSynchronize(stop));
            // 确保之前记录的 CUDA 事件同步完成
            CHECK_ERROR(cudaEventElapsedTime(&elapsed,start,stop));
            // 计算时间间隔
            CHECK_ERROR(cudaEventDestroy(start));
            CHECK_ERROR(cudaEventDestroy(stop));
            // 销毁 CUDA 事件
        }
        ofstream ftime("time_tmp.csv"); // 创建输出文件流对象
        ftime<<elapsed; // 数据计算时间(ms)
        ftime.close();
    #endif
    #ifdef End2end
        if(string(argv[1])=="4"){
            string path=string(argv[2]);
            writetocsv(path,phi,dimX,dimY);
        }
    #endif
    
    CHECK_ERROR(cudaFree(phi));
    CHECK_ERROR(cudaFree(phi_lap));
    CHECK_ERROR(cudaFree(tempr));
    CHECK_ERROR(cudaFree(tempr_lap));
    CHECK_ERROR(cudaFree(phidx));
    CHECK_ERROR(cudaFree(phidy));
    CHECK_ERROR(cudaFree(epsilon));
    CHECK_ERROR(cudaFree(epsilon_deri));
    // 释放 GPU 内存
    return 0;
}