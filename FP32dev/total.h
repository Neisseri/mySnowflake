#ifndef TOTAL_H
   #define TOTAL_H
   #include "stdio.h"
   #include <stdlib.h>
   #include <iostream>
   #include <cuda.h>
   #include <cuda_runtime.h>
   #include <unistd.h>
   #include <fstream>
   #include <string>
   #include <sstream>
   #include <mma.h>
   #include "device_launch_parameters.h"
   #include <cuda_fp16.h>
   #include <math.h>
   using namespace std;
   using namespace nvcuda;
   typedef float highprecision;
   typedef half lowprecision;
   int timesteps=500;
   const int dimX=512,dimY=512;
   const int unitx=16,unity=16,unitdimX=dimX/unitx,unitdimY=dimY/unity,uxd2=unitx/2,uxd2s1=uxd2-1,uxs1=unitx-1,uys1=unity-1,dimXd2=dimX/2,unitNums=unitdimX*unitdimY;
   const highprecision threshold=1e-05;
   const highprecision ratio=1.0;

   typedef half lowprecision;
   const int lowprecison_dimX=dimX;

   #define pi 3.1415926
   #define CHECK_ERROR(error) checkCudaError(error, __FILE__, __LINE__)
   inline void checkCudaError(cudaError_t error, const char *file, const int line)
   {
      if (error != cudaSuccess) {
         std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
         exit(EXIT_FAILURE);
      }
   }
   
#endif
