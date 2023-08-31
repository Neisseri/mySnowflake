#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main()
{
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量: " << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个SM的最大线程数: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
}

/* 输出结果为:
使用GPU device 0: NVIDIA A100-PCIE-40GB
SM的数量: 108
每个线程块的共享内存大小：48 KB
每个线程块的最大线程数：1024
每个SM的最大线程数: 2048
每个SM的最大线程束数: 64
*/