#ifndef FUNCTION_H
    #define FUNCTION_H
    #include"./paras.h"

    __global__ void kernel1_pure(
            highprecision *phi,
            highprecision* phi_lap,
            lowprecision* tempr,
            lowprecision* tempr_lap,
            lowprecision* phidx,
            lowprecision* phidy,
            highprecision* epsilon,
            highprecision* epsilon_deri
    ){
        int unitindex_x = blockIdx.z % unitdimX;
        int unitindex_y = blockIdx.z / unitdimX;
        // 计算线程块在二维线程网络中的索引
        int x_offset = threadIdx.x;
        int x_start = unitindex_x * unitx;
        int x = x_start + x_offset;
        // 计算线程在当前线程块内的 x 轴方向的索引
        // 和在整个线程网格中的 x 轴方向的绝对索引
        int y_offset = threadIdx.y;
        int y = unitindex_y * unity + y_offset;

        int xs1 = x > 0 ? (x - 1) : (dimX - 1);
        int ys1 = y > 0 ? (y - 1) : (dimY - 1); // s1: (x - 1, y - 1)
        int xa1 = x < (dimX - 1) ? (x + 1) : 0;
        int ya1 = y < (dimY - 1) ? (y + 1) : 0; // a1: (x + 1, y + 1)
        int xs2 = (x -2) < 0 ? (x - 2) + dimX : (x - 2);
        int xa2 = (x + 2) > (dimX - 1) ? (x + 2 - dimX) : (x + 2);
        int ys2 = (y - 2) < 0 ? (y - 2 + dimY) : (y - 2); // s2: (x - 2, y - 2)
        int ya2 = (y + 2) > (dimY - 1) ? (y + 2 - dimY) :(y + 2); // a2: (x + 2, y + 2)
        
        highprecision(*phid)[dimX]=(highprecision(*)[dimX])phi;
        // 将一维数组转换成二维数组，后续修改 phid 而非 phi
        highprecision(*phi_lapd)[dimX]=(highprecision(*)[dimX])phi_lap;
        // 计算像素的拉普拉斯值
        lowprecision(*phidxd)[dimX]=(lowprecision(*)[dimX])phidx;
        lowprecision(*phidyd)[dimX]=(lowprecision(*)[dimX])phidy;
        // 计算梯度
        lowprecision(*temprd)[dimX]=(lowprecision(*)[dimX])tempr;
        lowprecision(*tempr_lapd)[dimX]=(lowprecision(*)[dimX])tempr_lap; // half
        highprecision(*epsilond)[dimX]=(highprecision(*)[dimX])epsilon;
        highprecision(*epsilon_derid)[dimX]=(highprecision(*)[dimX])epsilon_deri;
        // 将一维指针转换为二维数组
        
        phi_lapd[y][x] = 
            phid[y][xs1] / dxdy
            + phid[y][xs2] / dxdy
            + phid[y][xa1] / dxdy
            + phid[y][xa2] / dxdy
            + phid[ys1][x] / dxdy
            + phid[ys2][x] / dxdy
            + phid[ya1][x] / dxdy
            + phid[ya2][x] / dxdy
            - 8.0 * phid[y][x] / dxdy;

        half delta_tempr_lapd;
        half dxdy16 = __float2half(dxdy);

        delta_tempr_lapd = temprd[y][xs1] / dxdy16
            + temprd[y][xs2] / dxdy16
            + temprd[y][xa1] / dxdy16
            + temprd[y][xa2] / dxdy16
            + temprd[ys1][x] / dxdy16
            + temprd[ys2][x] / dxdy16
            + temprd[ya1][x] / dxdy16
            + temprd[ya2][x] / dxdy16
            - __float2half(8.0) * temprd[y][x] / dxdy16
            - __float2half(tempr_lapd[y][x]);

        tempr_lapd[y][x] += delta_tempr_lapd;

        phidxd[y][x] = __float2half((phid[y][xa1] - phid[y][xs1]) / (2.0 * dx));

        phidyd[y][x] = __float2half((phid[ya1][x] - phid[ys1][x]) / (2.0 * dy));

        highprecision theta = atan2(__half2float(phidyd[y][x]), __half2float(phidxd[y][x]));
        epsilond[y][x] = __half2float(
            __float2half(epsilonb) * 
            (__float2half(1.0) + __float2half(delta) * __float2half(cos(aniso * (theta - theta0)))));
        epsilon_derid[y][x] = -epsilonb * aniso * delta * sin(aniso * (theta - theta0));
    }

    __global__ void kernel2_pure(
            highprecision* phi,
            highprecision* phi_lap,
            highprecision* epsilon,
            highprecision *epsilon_deri,
            lowprecision* phidx,
            lowprecision* phidy,
            lowprecision* tempr,
            lowprecision* tempr_lap
    ){
        int unitindex_x = blockIdx.z % unitdimX;
        int unitindex_y = blockIdx.z / unitdimX;
        int x_offset = threadIdx.x;
        int x_start = unitindex_x * unitx;
        int x = x_start + x_offset;
        int y_offset = threadIdx.y;
        int y = unitindex_y * unity + y_offset;
        
        int xs1 = x > 0 ? (x - 1) : (dimX - 1);
        int ys1 = y > 0 ? (y - 1) : (dimY - 1);
        int xa1 = x < (dimX - 1) ? (x + 1) : 0;
        int ya1 = y < (dimY - 1) ? (y + 1) : 0;
        
        highprecision(*phid)[dimX] = (highprecision(*)[dimX])phi;
        highprecision(*phi_lapd)[dimX] = (highprecision(*)[dimX])phi_lap;
        highprecision(*epsilond)[dimX] = (highprecision(*)[dimX])epsilon;
        highprecision(*epsilon_derid)[dimX] = (highprecision(*)[dimX])epsilon_deri;
        lowprecision(*phidxd)[dimX] = (lowprecision(*)[dimX])phidx;
        lowprecision(*phidyd)[dimX] = (lowprecision(*)[dimX])phidy;
        lowprecision(*temprd)[dimX] = (lowprecision(*)[dimX])tempr;
        lowprecision(*tempr_lapd)[dimX] = (lowprecision(*)[dimX])tempr_lap;
        highprecision phi_old = phid[y][x];
        highprecision deltaphi;

        highprecision term1 = (
            epsilond[ya1][x] * epsilon_derid[ya1][x] * __half2float(phidxd[ya1][x]) 
            - epsilond[ys1][x] * epsilon_derid[ys1][x] * __half2float(phidxd[ys1][x])
        ) / (2.0 * dy);
        highprecision term2 = - __half2float((
            __float2half(epsilond[y][xa1]) * __float2half(epsilon_derid[y][xa1]) * phidyd[y][xa1]
            - __float2half(epsilond[y][xs1]) * __float2half(epsilon_derid[y][xs1]) * phidyd[y][xs1]
        ) / (__float2half(2.0) * __float2half(dx)));
        
        highprecision m = __half2float(
            __float2half(alpha) / __float2half(pi) 
            * __float2half(atan(gama * (teq - __half2float(temprd[y][x]))))
        );
        
        deltaphi = (dtime / tau) * 
            (term1 + term2 
                + pow(epsilond[y][x], 2) * phi_lapd[y][x]
                + phi_old * (1.0 - phi_old) * (phi_old - 0.5 + m)
            );
        phid[y][x] = phid[y][x] + deltaphi; 
            
        half delta_temprd = __float2half(dtime) * tempr_lapd[y][x]
            + __float2half(kappa * (phid[y][x] - phi_old));

        temprd[y][x] = temprd[y][x] + delta_temprd;
    }    

#endif