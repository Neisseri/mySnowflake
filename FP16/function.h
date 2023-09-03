#ifndef FUNCTION_H
    #define FUNCTION_H
    #include"./paras.h"

    __global__ void kernel1_pure(lowprecision *phi,lowprecision* phi_lap,lowprecision* tempr,lowprecision* tempr_lap,lowprecision* phidx,lowprecision* phidy,lowprecision* epsilon,lowprecision* epsilon_deri){
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
        
        lowprecision(*phid)[dimX]=(lowprecision(*)[dimX])phi;
        // 将一维数组转换成二维数组，后续修改 phid 而非 phi
        lowprecision(*phi_lapd)[dimX]=(lowprecision(*)[dimX])phi_lap;
        // 计算像素的拉普拉斯值
        lowprecision(*phidxd)[dimX]=(lowprecision(*)[dimX])phidx;
        lowprecision(*phidyd)[dimX]=(lowprecision(*)[dimX])phidy;
        // 计算梯度
        lowprecision(*temprd)[dimX]=(lowprecision(*)[dimX])tempr;
        lowprecision(*tempr_lapd)[dimX]=(lowprecision(*)[dimX])tempr_lap;
        lowprecision(*epsilond)[dimX]=(lowprecision(*)[dimX])epsilon;
        lowprecision(*epsilon_derid)[dimX]=(lowprecision(*)[dimX])epsilon_deri;
        // 将一维指针转换为二维数组

        /*phi_lapd[y][x] = phid[y][xs1] - phid[y][x];
        phi_lapd[y][x] += phid[y][xs2] - phid[y][x];
        phi_lapd[y][x] += phid[y][xa1] - phid[y][x];
        phi_lapd[y][x] += phid[y][xa2] - phid[y][x];
        phi_lapd[y][x] += phid[ys1][x] - phid[y][x];
        phi_lapd[y][x] += phid[ys2][x] - phid[y][x];
        phi_lapd[y][x] += phid[ya1][x] - phid[y][x];
        phi_lapd[y][x] += phid[ya2][x] - phid[y][x];*/

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
        // dxdy = 0.0009

        tempr_lapd[y][x] = 
            temprd[y][xs1]
            + temprd[y][xs2]
            + temprd[y][xa1]
            + temprd[y][xa2]
            + temprd[ys1][x]
            + temprd[ys2][x]
            + temprd[ya1][x]
            + temprd[ya2][x]
            - __double2half(8.0) * temprd[y][x];

        phidxd[y][x] = phid[y][xa1] - phid[y][xs1];
        phidyd[y][x] = phid[ya1][x] - phid[ys1][x];


        lowprecision theta = atan2(__half2float(phidyd[y][x]), __half2float(phidxd[y][x]));
        epsilond[y][x] = epsilonb * (__double2half(1.0) + delta * __float2half(cos(__half2float(aniso * (theta - theta0)))));
        epsilon_derid[y][x] = -epsilonb * aniso * delta * __float2half(sin(__half2float(aniso * (theta - theta0))));
    }

    __global__ void kernel2_pure(lowprecision* phi,lowprecision* phi_lap,lowprecision* epsilon,lowprecision *epsilon_deri,lowprecision* phidx,lowprecision* phidy,lowprecision* tempr,lowprecision* tempr_lap){
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
        
        lowprecision(*phid)[dimX] = (lowprecision(*)[dimX])phi;
        lowprecision(*phi_lapd)[dimX] = (lowprecision(*)[dimX])phi_lap;
        lowprecision(*epsilond)[dimX] = (lowprecision(*)[dimX])epsilon;
        lowprecision(*epsilon_derid)[dimX] = (lowprecision(*)[dimX])epsilon_deri;
        lowprecision(*phidxd)[dimX] = (lowprecision(*)[dimX])phidx;
        lowprecision(*phidyd)[dimX] = (lowprecision(*)[dimX])phidy;
        lowprecision(*temprd)[dimX] = (lowprecision(*)[dimX])tempr;
        lowprecision(*tempr_lapd)[dimX] = (lowprecision(*)[dimX])tempr_lap;
        lowprecision phi_old = phid[y][x];

        lowprecision term1 = epsilond[ya1][x] * epsilon_derid[ya1][x] * phidxd[ya1][x]
            - epsilond[ys1][x] * epsilon_derid[ys1][x] * phidxd[ys1][x];
        lowprecision term2 = - epsilond[y][xa1] * epsilon_derid[y][xa1] * phidyd[y][xa1]
            + epsilond[y][xs1] * epsilon_derid[y][xs1] * phidyd[y][xs1];
        
        lowprecision m = alpha / __double2half(pi * atan(__half2float(gama * (teq - temprd[y][x]))));

        phid[y][x] = phid[y][x] + 
            (dtime / tau) * 
            ((term1 + term2) / (__double2half(4.0) * dxdy) 
                + __double2half(pow(__half2float(epsilond[y][x]), 2.0)) * phi_lapd[y][x] / dxdy
                + phi_old * (__double2half(1.0) - phi_old) * (phi_old - __double2half(0.5) + m)
            );
            
        temprd[y][x] = temprd[y][x] 
            + dtime * tempr_lapd[y][x] / dxdy
            + kappa * (phid[y][x] - phi_old);
    }

#endif